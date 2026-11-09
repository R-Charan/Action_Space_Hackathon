"""Attackers as the learning agents, against the classical bus defender (see the cfg docstring)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch

from . import room_layout
from .bus_pilot import AttackerTracker, BusPilot, BusPilotCfg, TelemetryCfg
from .tello_attack_face_env_cfg import TelloAttackFaceEnvCfg
from .tello_defend_face_env import TelloDefendFaceEnv


def _with(cfg, overrides: dict, label: str):
    names = {f.name for f in dataclasses.fields(cfg)}
    for key, value in overrides.items():
        if key not in names:
            raise ValueError(f"unknown {label} field: {key}")
        setattr(cfg, key, value)
    return cfg


class TelloAttackFaceEnv(TelloDefendFaceEnv):
    cfg: TelloAttackFaceEnvCfg

    def __init__(self, cfg: TelloAttackFaceEnvCfg, render_mode: str | None = None, **kwargs):
        n, a, dev = cfg.scene.num_envs, cfg.num_attackers, cfg.sim.device
        self._att_cmd = torch.zeros(n, a, 3, device=dev)
        self._phi = torch.zeros(n, a, device=dev)
        self._ep_def_down = torch.zeros(n, device=dev)
        super().__init__(cfg, render_mode, **kwargs)
        for i in range(cfg.num_defenders):
            self._actions[f"defender_{i}"] = torch.zeros(self.num_envs, 4, device=self.device)
        self._pilot_every = max(1, round(1.0 / (cfg.pilot_hz * self.step_dt)))
        pcfg = BusPilotCfg()
        if cfg.room_map:
            pcfg.y_min, pcfg.y_max = room_layout.pilot_y_range(cfg)
        self.pilot = BusPilot(_with(pcfg, cfg.pilot_set, "BusPilotCfg"), self.face, self.num_envs,
                              cfg.num_defenders, a, self._pilot_every * self.step_dt, self.device, tags=self.room_tags)
        self.tracker = AttackerTracker(_with(TelemetryCfg(), cfg.telemetry_set, "TelemetryCfg"), self.num_envs, a,
                                       self.step_dt, self.device)
        self._steps = 0
        self._defender_cmd = {f"defender_{i}": torch.zeros(self.num_envs, 4, device=self.device)
                              for i in range(cfg.num_defenders)}

    # -- actions -------------------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        c = self.cfg
        cmd = torch.stack([actions[f"attacker_{i}"] for i in range(c.num_attackers)], 1).clamp(-1.0, 1.0)
        if c.speed_cap_norm:
            cmd = cmd / torch.linalg.norm(cmd, dim=-1, keepdim=True).clamp(min=1.0)
        self._att_cmd = cmd

        # the defence: bus pilot on delayed, noisy attacker telemetry, at its own command rate
        if self._steps % self._pilot_every == 0:
            d_pos = self._local(self._defenders)
            d_vel = torch.stack([o.data.root_lin_vel_w for o in self._defenders], 1)
            d_quat = torch.stack([o.data.root_quat_w for o in self._defenders], 1)
            a_est, a_vel, a_alive = self.tracker.estimate()
            vel, yaw_rate = self.pilot.step(d_pos, d_vel, d_quat, self._defender_alive.clone(), a_est, a_vel, a_alive)
            act = torch.cat([vel / c.command.max_lin_speed, (yaw_rate / c.command.max_yaw_rate).unsqueeze(-1)], -1)
            self._defender_cmd = {f"defender_{i}": act[:, i].clamp(-1.0, 1.0) for i in range(c.num_defenders)}
        self._steps += 1
        super()._pre_physics_step(self._defender_cmd)  # stores them; downed defenders land

    def _scripted_attacker_actions(self) -> list[torch.Tensor]:
        c = self.cfg
        vel = self._att_cmd * c.command.max_lin_speed
        z = self._local(self._attackers)[..., 2]
        landing = torch.zeros_like(vel)
        landing[..., 2] = -c.downed_descent_speed * (z > c.downed_floor_height)
        vel = torch.where(self._attacker_alive.unsqueeze(-1), vel, landing)
        act = torch.cat([vel / c.command.max_lin_speed, torch.zeros_like(vel[..., :1])], -1).clamp(-1.0, 1.0)
        return [act[:, i] for i in range(c.num_attackers)]

    # -- MDP -----------------------------------------------------------------------------

    def _potential(self, a_pos: torch.Tensor, d_pos: torch.Tensor, d_alive: torch.Tensor) -> torch.Tensor:
        """Per attacker (N, A): -w_d * face distance - w_m * clipped best defender margin."""
        c = self.cfg
        margin = self.face.margin(d_pos.unsqueeze(2), a_pos.unsqueeze(1), self.pilot.pts)  # (N, D, A)
        margin = margin.masked_fill(~d_alive.unsqueeze(-1), -1e3).max(1).values
        margin = margin.clamp(*c.margin_clip)  # no live defender: the low clip
        return -c.rew_face_dist * self.face.distance(a_pos) - c.rew_margin * margin

    def _get_observations(self) -> dict[str, torch.Tensor]:
        c = self.cfg
        a_pos, d_pos = self._local(self._attackers), self._local(self._defenders)
        a_vel = torch.stack([o.data.root_lin_vel_w for o in self._attackers], 1)
        d_vel = torch.stack([o.data.root_lin_vel_w for o in self._defenders], 1)
        a_alive, d_alive = self._attacker_alive.float(), self._defender_alive.float()
        centre = torch.tensor([0.0, 0.0, c.face_bottom + 0.5 * c.face_height], device=self.device)
        left = 1.0 - self.episode_length_buf.float() / self.max_episode_length
        obs = {}
        for i in range(c.num_attackers):
            j = 1 - i if c.num_attackers == 2 else (i + 1) % c.num_attackers
            me = a_alive[:, i : i + 1]
            d_rel = ((d_pos - a_pos[:, i : i + 1]) * d_alive.unsqueeze(-1)).clamp(-c.obs_clip_pos, c.obs_clip_pos)
            d_rv = ((d_vel - a_vel[:, i : i + 1]) * d_alive.unsqueeze(-1)).clamp(-c.obs_clip_vel, c.obs_clip_vel)
            mate = a_alive[:, j : j + 1]
            obs[f"attacker_{i}"] = torch.cat([
                (a_pos[:, i] - centre) * me, a_vel[:, i] * me, me,
                (a_pos[:, j] - a_pos[:, i]) * mate * me, (a_vel[:, j] - a_vel[:, i]) * mate * me, mate,
                torch.cat([d_rel, d_rv, d_alive.unsqueeze(-1)], -1).reshape(self.num_envs, -1) * me,
                self.face.distance(a_pos[:, i : i + 1]) * me, left.unsqueeze(-1),
            ], -1)
        return obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        c = self.cfg
        a_before, d_before = self._attacker_alive.clone(), self._defender_alive.clone()
        super()._get_rewards()  # contacts, breach, out of play; updates alive flags and episode tallies
        a_pos = self._local(self._attackers)
        self.tracker.step(a_pos, self._attacker_alive)

        newly = a_before & ~self._attacker_alive
        a_down = newly.float().sum(-1)
        d_down = (d_before & ~self._defender_alive).float().sum(-1)
        left = self._out_of_arena(a_pos) | ((a_pos[..., 0] < 0.0) & ~self.face.inside(a_pos))
        out = (newly & left).float().sum(-1)
        self._ep_def_down += d_down
        team = (c.rew_breach * self._breached.float() + c.rew_defender_down * d_down
                - c.rew_attacker_down * a_down - c.rew_out * out)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        team -= c.rew_timeout * (time_out & ~self._breached & self._attacker_alive.any(-1)).float()

        # potential shaping: Phi of a downed attacker, and of every attacker once the episode ends,
        # is 0. Then the shaping over an episode sums to -Phi(start), the same whatever the policy
        # does, so it guides without biasing (freezing Phi at death would pay for dying near the face)
        ended = self._breached | ~self._attacker_alive.any(-1) | time_out
        phi = self._potential(a_pos, self._local(self._defenders), self._defender_alive)
        phi = torch.where(self._attacker_alive & ~ended.unsqueeze(-1), phi, torch.zeros_like(phi))
        team += (phi - self._phi).sum(-1)
        self._phi = phi
        share = team / c.num_attackers
        return {agent: share for agent in self.cfg.possible_agents}

    def _get_dones(self):
        terminated, time_outs = super()._get_dones()
        finished = self.outcome_valid
        if finished.any():
            ids = finished.nonzero().flatten()
            self.extras["log"] = {
                "Episode/count": float(len(ids)),
                "Episode/breach_rate": float(self._breached[ids].float().mean()),
                "Episode/attackers_downed": float(self._episode_captures[ids].mean()),
                "Episode/defenders_downed": float(self._ep_def_down[ids].mean()),
                "Episode/out_of_play": float(self._ep_out[ids].mean()),
                "Episode/length_s": float(self.episode_length_buf[ids].float().mean() * self.step_dt),
            }
            # per-env records, for the episode cohort in sb3/play.py
            cols = torch.stack([self._breached[ids].float(), self._episode_captures[ids].float(),
                                self._ep_def_down[ids], self._ep_out[ids].float(),
                                self.episode_length_buf[ids].float() * self.step_dt, self._ep_att_vmax[ids]], -1).tolist()
            keys = ("breach_rate", "attackers_downed", "defenders_downed", "out_of_play", "length_s", "att_max_speed")
            self.extras["episode_records"] = [
                {"env_id": int(i), "metrics": {f"Episode/{k}": v for k, v in zip(keys, row)}}
                for i, row in zip(ids.tolist(), cols)]
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)
        if not hasattr(self, "pilot"):
            return  # during construction
        env_ids = torch.as_tensor(env_ids, device=self.device)
        self._att_cmd[env_ids] = 0.0
        self._ep_def_down[env_ids] = 0.0
        for key in self._defender_cmd:
            self._defender_cmd[key][env_ids] = 0.0
            self._actions[key][env_ids] = 0.0
        self.pilot.reset(env_ids)
        start = self._prev_a_pos[env_ids]  # the new starts (the sim buffers update on the next step)
        self.tracker.reset(env_ids, start, torch.ones_like(self._attacker_alive[env_ids]))
        self._phi[env_ids] = self._potential(start, self._prev_d_pos[env_ids], self._defender_alive[env_ids])
