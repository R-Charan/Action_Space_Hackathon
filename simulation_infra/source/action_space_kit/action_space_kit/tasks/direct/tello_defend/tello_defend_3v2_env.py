"""3-vs-2 defend task: evenly spaced defenders, attackers that may feint, weave, jink and dodge.

With ``mutual_kill`` (default) any two drones that come within ``collision_radius`` are both
downed and land: an interception trades a defender for an attacker, and defenders that hit
each other lose two. One wave; both attackers down without a breach is a win.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils.math import sample_uniform

from .tello_defend_3v2_env_cfg import TelloDefend3v2EnvCfg
from .tello_defend_env import TelloDefendEnv

DIVE, BAIT_IN, BAIT_OUT = 0, 1, 2


class TelloDefend3v2Env(TelloDefendEnv):
    cfg: TelloDefend3v2EnvCfg

    def __init__(self, cfg: TelloDefend3v2EnvCfg, render_mode: str | None = None, **kwargs):
        # per-attacker buffers must exist before the first reset, which the parent may run
        n, a, dev = cfg.scene.num_envs, cfg.num_attackers, cfg.sim.device
        self._bait_phase = torch.zeros(n, a, dtype=torch.long, device=dev)
        self._bait_depth = torch.zeros(n, a, device=dev)
        # aggressive flight: weave, jinks, speed bursts, altitude changes
        self._clock = torch.zeros(n, a, device=dev)
        self._weave_amp = torch.zeros(n, a, device=dev)
        self._weave_freq = torch.zeros(n, a, device=dev)
        self._weave_phase = torch.zeros(n, a, device=dev)
        self._jink = torch.zeros(n, a, device=dev)
        self._jink_timer = torch.zeros(n, a, device=dev)
        self._speed_timer = torch.zeros(n, a, device=dev)
        self._alt_target = torch.zeros(n, a, device=dev)
        self._alt_timer = torch.zeros(n, a, device=dev)
        # defenders can be downed too
        d = cfg.num_defenders
        self._defender_alive = torch.ones(n, d, dtype=torch.bool, device=dev)
        self._ep_defenders_lost = torch.zeros(n, device=dev)
        self._ep_teamkills = torch.zeros(n, device=dev)
        self.outcome_defenders_lost = torch.zeros(n, device=dev)
        self.outcome_teamkills = torch.zeros(n, device=dev)
        super().__init__(cfg, render_mode, **kwargs)

    # -- downed defenders land -------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        super()._pre_physics_step(actions)
        for i, drone in enumerate(self._defenders):
            z = drone.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
            land = torch.zeros_like(self._actions[f"defender_{i}"])
            land[:, 2] = -self.cfg.downed_descent_speed / self.cfg.command.max_lin_speed * (z > self.cfg.downed_floor_height)
            self._actions[f"defender_{i}"] = torch.where(self._defender_alive[:, i : i + 1], self._actions[f"defender_{i}"], land)

    # -- collisions down both drones -------------------------------------------------

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        if not self.cfg.mutual_kill:
            return super()._get_rewards()
        c = self.cfg
        att_pos, _ = self._attacker_states()
        def_pos = torch.stack([d.data.root_pos_w for d in self._defenders], 1)
        site = self.scene.env_origins.clone()
        site[:, 2] += c.site_height * 0.5
        d_alive, a_alive = self._defender_alive, self._attacker_alive
        eye = lambda k: torch.eye(k, dtype=torch.bool, device=self.device)  # noqa: E731

        # defender-attacker: an interception, both go down
        da = torch.linalg.norm(def_pos.unsqueeze(2) - att_pos.unsqueeze(1), dim=-1) < c.collision_radius
        da &= d_alive.unsqueeze(2) & a_alive.unsqueeze(1)
        # defender-defender and attacker-attacker collisions
        dd = torch.linalg.norm(def_pos.unsqueeze(2) - def_pos.unsqueeze(1), dim=-1) < c.collision_radius
        dd &= d_alive.unsqueeze(2) & d_alive.unsqueeze(1) & ~eye(c.num_defenders)
        aa = torch.linalg.norm(att_pos.unsqueeze(2) - att_pos.unsqueeze(1), dim=-1) < c.collision_radius
        aa &= a_alive.unsqueeze(2) & a_alive.unsqueeze(1) & ~eye(c.num_attackers)

        captured = da.any(1)  # (N, A)
        att_down = captured | aa.any(-1)
        intercepting = da.any(2)  # (N, D)
        teamkill = dd.any(-1)
        def_down = intercepting | teamkill
        self._attacker_alive = a_alive & ~att_down
        self._defender_alive = d_alive & ~def_down
        self._ep_defenders_lost += def_down.float().sum(-1)
        self._ep_teamkills += teamkill.float().sum(-1)

        radial = torch.linalg.norm(att_pos[:, :, :2] - site[:, :2].unsqueeze(1), dim=-1)
        self._breached = ((radial < c.site_radius) & self._attacker_alive).any(dim=-1)
        cleared = att_down.any(dim=-1) & ~self._attacker_alive.any(dim=-1)
        self._episode_captures += att_down.float().sum(dim=-1)

        rewards = {}
        any_alive = self._attacker_alive.any(dim=-1)
        for i in range(c.num_defenders):
            dist = torch.linalg.norm(att_pos - def_pos[:, i].unsqueeze(1), dim=-1)
            dist = torch.where(self._attacker_alive, dist, torch.full_like(dist, float("inf")))
            nearest = torch.where(any_alive, dist.min(dim=-1).values, self._prev_distance[:, i])
            closing = self._prev_distance[:, i] - nearest
            self._prev_distance[:, i] = nearest
            effort = torch.sum(torch.square(self._actions[f"defender_{i}"]), dim=-1)
            proximity = torch.exp(-nearest.clamp(min=0.0, max=c.spawn_radius * 2.0) / c.spawn_radius)
            shaping = (c.rew_scale_proximity * proximity + c.rew_scale_effort * effort + c.rew_scale_alive) * self.step_dt
            shaping = (shaping + c.rew_scale_closing * closing.clamp(-1.0, 1.0)) * d_alive[:, i].float()
            events = (
                c.rew_scale_capture * da[:, i].float().sum(-1)
                + c.rew_scale_defender_lost * def_down[:, i].float()
                + c.rew_scale_teamkill * teamkill[:, i].float()
                + c.rew_scale_breach * self._breached.float()
                + c.rew_scale_cleared * cleared.float()
            )
            rewards[f"defender_{i}"] = shaping + events
        return rewards

    def _get_dones(self):
        terminated, time_outs = super()._get_dones()
        finished = self.outcome_valid
        self.outcome_defenders_lost = torch.where(finished, self._ep_defenders_lost, self.outcome_defenders_lost)
        self.outcome_teamkills = torch.where(finished, self._ep_teamkills, self.outcome_teamkills)
        return terminated, time_outs

    def _sample_wave(self, env_ids: torch.Tensor) -> None:
        """Per-attacker behaviour for a new wave: bait or not, and its flight style."""
        c = self.cfg
        shape = (len(env_ids), c.num_attackers)
        u = lambda r: sample_uniform(r[0], r[1], shape, self.device)  # noqa: E731
        baiter = torch.rand(shape, device=self.device) < c.bait_prob
        self._bait_phase[env_ids] = torch.where(baiter, BAIT_IN, DIVE)
        self._bait_depth[env_ids] = u(c.bait_depth)
        self._clock[env_ids] = 0.0
        self._weave_amp[env_ids] = u(c.weave_amp)
        self._weave_freq[env_ids] = u(c.weave_freq)
        self._weave_phase[env_ids] = u((0.0, 2.0 * math.pi))
        self._jink[env_ids] = 0.0
        self._jink_timer[env_ids] = u(c.jink_interval)
        self._speed_timer[env_ids] = u(c.speed_interval)
        self._alt_target[env_ids] = u(c.altitude_range)
        self._alt_timer[env_ids] = u(c.altitude_interval)

    def _aggressive_actions(self, pos: torch.Tensor) -> torch.Tensor:
        """(N, A, 4) actions for attackers flying the aggressive style; pos is env-local (N, A, 3)."""
        c = self.cfg
        dt = self.physics_dt
        shape = self._clock.shape
        u = lambda r: sample_uniform(r[0], r[1], shape, self.device)  # noqa: E731
        self._clock += dt

        # piecewise-constant random swerves, speeds and heights, each on its own random timer
        self._jink_timer -= dt
        due = self._jink_timer <= 0.0
        self._jink = torch.where(due, u((-c.jink_amp, c.jink_amp)), self._jink)
        self._jink_timer = torch.where(due, u(c.jink_interval), self._jink_timer)
        self._speed_timer -= dt
        due = self._speed_timer <= 0.0
        self._attacker_speed = torch.where(due, u(c.attacker_speed), self._attacker_speed)
        self._speed_timer = torch.where(due, u(c.speed_interval), self._speed_timer)
        self._alt_timer -= dt
        due = self._alt_timer <= 0.0
        self._alt_target = torch.where(due, u(c.altitude_range), self._alt_target)
        self._alt_timer = torch.where(due, u(c.altitude_interval), self._alt_timer)

        # horizontal: towards the site, bent sideways by the weave and the current jink
        r = torch.linalg.norm(pos[..., :2], dim=-1, keepdim=True).clamp(min=1e-3)
        fwd = -pos[..., :2] / r
        side = torch.stack([-fwd[..., 1], fwd[..., 0]], -1)
        weave = self._weave_amp * torch.sin(2.0 * math.pi * self._weave_freq * self._clock + self._weave_phase)
        heading = fwd + (weave + self._jink).unsqueeze(-1) * side

        # dodge: swerve sideways (never backwards) away from defenders closer than dodge_radius
        if c.dodge_gain > 0.0:
            d_pos = torch.stack([d.data.root_pos_w for d in self._defenders], 1) - self.scene.env_origins.unsqueeze(1)
            away = pos[:, :, None, :2] - d_pos[:, None, :, :2]  # (N, A, D, 2)
            dist = torch.linalg.norm(away, dim=-1, keepdim=True).clamp(min=1e-3)
            push = ((c.dodge_radius - dist) / c.dodge_radius).clamp(min=0.0) * away / dist
            push = push.sum(2)
            lateral = (push * side).sum(-1, keepdim=True) * side  # keep only the sideways part
            heading = heading + c.dodge_gain * lateral

        heading = heading / torch.linalg.norm(heading, dim=-1, keepdim=True).clamp(min=1e-6)
        v_xy = heading * self._attacker_speed.unsqueeze(-1)
        v_z = (c.altitude_gain * (self._alt_target - pos[..., 2])).clamp(-c.max_climb_rate, c.max_climb_rate)
        act = torch.cat([v_xy, v_z.unsqueeze(-1), torch.zeros_like(v_z).unsqueeze(-1)], -1)
        return (act / self.cfg.command.max_lin_speed).clamp(-1.0, 1.0)

    def _scripted_attacker_actions(self) -> list[torch.Tensor]:
        actions = super()._scripted_attacker_actions()
        pos = torch.stack([d.data.root_pos_w for d in self._attackers], dim=1) - self.scene.env_origins.unsqueeze(1)
        r = torch.linalg.norm(pos[..., :2], dim=-1)
        phase = self._bait_phase
        phase = torch.where((phase == BAIT_IN) & (r < self._bait_depth), BAIT_OUT, phase)
        phase = torch.where((phase == BAIT_OUT) & (r > self.cfg.bait_retreat), DIVE, phase)
        self._bait_phase = phase
        aggressive = self._aggressive_actions(pos) if self.cfg.aggressive else None
        out = []
        for i, action in enumerate(actions):
            if aggressive is not None:
                action = torch.where(self._attacker_alive[:, i].unsqueeze(-1), aggressive[:, i], action)
            # retreating: the same speed, straight away from the site, holding height
            retreat = torch.cat([-actions[i][:, :2], torch.zeros_like(action[:, 2:])], dim=-1)
            action = torch.where((phase[:, i] == BAIT_OUT).unsqueeze(-1), retreat, action)
            # intercepted: it is sent `land` and descends at the SDK's 0.5 m/s, motors idle near the floor
            landing = ~self._attacker_alive[:, i] & (pos[:, i, 2] > self.cfg.downed_floor_height)
            land = torch.zeros_like(action)
            land[:, 2] = -self.cfg.downed_descent_speed / self.cfg.command.max_lin_speed
            out.append(torch.where(landing.unsqueeze(-1), land, action))
        return out

    def _respawn_attackers(self, env_ids: torch.Tensor) -> None:
        super()._respawn_attackers(env_ids)
        if len(env_ids):
            self._sample_wave(env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        self._defender_alive[env_ids] = True
        self._ep_defenders_lost[env_ids] = 0.0
        self._ep_teamkills[env_ids] = 0.0
        num = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        # evenly spaced around the site, as a team would set them out, with a random rotation
        base = sample_uniform(0.0, 6.2832, (num,), self.device)
        for i, drone in enumerate(self._defenders):
            angle = base + i * 6.2832 / self.cfg.num_defenders
            state = drone.data.default_root_state[env_ids].clone()
            state[:, 0] = origins[:, 0] + self.cfg.defender_radius * torch.cos(angle)
            state[:, 1] = origins[:, 1] + self.cfg.defender_radius * torch.sin(angle)
            state[:, 2] = origins[:, 2] + sample_uniform(1.0, 2.0, (num,), self.device)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self._sample_wave(env_ids)
