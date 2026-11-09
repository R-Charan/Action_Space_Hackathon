"""Defend a breach face: 3 defenders vs 2 identical attackers, one wave, both drones down on contact.

Attackers start across the arena and fly at a point of the face (``face_geometry.Face``); one
alive and inside the face rectangle at x < 0 is a breach. Scripted attackers dive, bait (fly
in, back out, then dive), and with ``aggressive`` weave, jink, change aim point and speed at
random and dodge defenders. Any two drones within ``collision_radius`` are both downed and
land; so is any drone that leaves the arena, or an attacker that passes the face plane
outside the face. Both attackers down without a breach is a win for the defence.

With ``room_map`` the arena is the real room from map.yaml (``room_layout.py``) and ``room_tags``
holds its tags, face-local; with ``defender_takeoff`` the defenders start on the floor and fly the
SDK takeoff (straight up, sticks ignored) before their pilot takes over. Every scripted attacker command is capped at ``attacker_max_speed``,
and each episode records the fastest TRUE speed of either side (``outcome_*_max_speed``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils.math import sample_uniform

from . import room_layout
from .face_geometry import Face
from .tello_defend_env import TelloDefendEnv
from .tello_defend_face_env_cfg import TelloDefendFaceEnvCfg

DIVE, BAIT_IN, BAIT_OUT = 0, 1, 2


class TelloDefendFaceEnv(TelloDefendEnv):
    cfg: TelloDefendFaceEnvCfg

    def __init__(self, cfg: TelloDefendFaceEnvCfg, render_mode: str | None = None, **kwargs):
        n, a, d, dev = cfg.scene.num_envs, cfg.num_attackers, cfg.num_defenders, cfg.sim.device
        if cfg.room_map:
            room_layout.apply_room(cfg)  # before the first reset: the starts and arena come from the room
        self.face = Face(cfg.face_width, cfg.face_height, cfg.face_bottom)
        z = lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype, device=dev)  # noqa: E731
        self._bait_phase = z(n, a, dtype=torch.long)
        self._bait_depth = z(n, a)
        self._aim = z(n, a, 2)  # (y, z) on the face
        self._aim_timer = z(n, a)
        self._clock = z(n, a)
        self._weave_amp, self._weave_freq, self._weave_phase = z(n, a), z(n, a), z(n, a)
        self._jink, self._jink_timer, self._speed_timer = z(n, a), z(n, a), z(n, a)
        self._defender_alive = torch.ones(n, d, dtype=torch.bool, device=dev)
        self._ep_defenders_lost, self._ep_teamkills, self._ep_out, self._ep_att_collisions = z(n), z(n), z(n), z(n)
        self.outcome_defenders_lost, self.outcome_teamkills, self.outcome_out = z(n), z(n), z(n)
        self.outcome_att_collisions = z(n)  # attackers downed by hitting each other
        self._prev_a_pos, self._prev_d_pos = z(n, a, 3), z(n, d, 3)  # last step's positions, for swept contact
        self._ep_att_vmax, self._ep_def_vmax = z(n), z(n)  # fastest true speed this episode, live drones
        self.taking_off = torch.zeros(n, d, dtype=torch.bool, device=dev)  # SDK takeoff in progress
        self._takeoff_alt = z(n, d)
        self.outcome_att_max_speed, self.outcome_def_max_speed = z(n), z(n)
        super().__init__(cfg, render_mode, **kwargs)
        self.room_tags = None  # None: the scripts fall back to bus_pilot.box_room_tags
        if cfg.room_map:
            pos, normal, size, ids = room_layout.room_tags(cfg, self.device)
            self.room_tags = (pos, normal, size)
            print(f"[face room] {len(ids)} tags {ids}; face at room X = {room_layout.face_x(cfg):.2f} m, "
                  f"arena x {cfg.arena_x} y {cfg.arena_y} (face-local); attackers start x {cfg.attacker_start_x} "
                  f"y {cfg.attacker_start_y}", flush=True)

    # -- helpers -----------------------------------------------------------------------

    def _local(self, objs) -> torch.Tensor:
        return torch.stack([o.data.root_pos_w for o in objs], 1) - self.scene.env_origins.unsqueeze(1)

    @staticmethod
    def _closest_approach(p0: torch.Tensor, q0: torch.Tensor, p1: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        """Pairwise closest distance over one step, both moving linearly: p (N, P, 3), q (N, Q, 3) -> (N, P, Q).

        Checking only the end of each step would let two drones closing at 3 m/s pass through each
        other unseen once a step is longer than ~0.1 s (a policy acting at 15 Hz).
        """
        r0 = p0.unsqueeze(2) - q0.unsqueeze(1)
        dr = (p1.unsqueeze(2) - q1.unsqueeze(1)) - r0
        t = (-(r0 * dr).sum(-1) / (dr * dr).sum(-1).clamp(min=1e-9)).clamp(0.0, 1.0)
        return torch.linalg.norm(r0 + t.unsqueeze(-1) * dr, dim=-1)

    def _sample_aim(self, env_ids: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        c = self.cfg
        shape = (len(env_ids), c.num_attackers)
        y = sample_uniform(-0.5 * c.face_width, 0.5 * c.face_width, shape, self.device)
        zz = sample_uniform(c.face_bottom, c.face_bottom + c.face_height, shape, self.device)
        aim = torch.stack([y, zz], -1)
        if mask is not None:
            aim = torch.where(mask[env_ids].unsqueeze(-1), aim, self._aim[env_ids])
        self._aim[env_ids] = aim

    def _sample_behaviour(self, env_ids: torch.Tensor) -> None:
        c = self.cfg
        shape = (len(env_ids), c.num_attackers)
        u = lambda r: sample_uniform(r[0], r[1], shape, self.device)  # noqa: E731
        self._bait_phase[env_ids] = torch.where(torch.rand(shape, device=self.device) < c.bait_prob, BAIT_IN, DIVE)
        self._bait_depth[env_ids] = u(c.bait_depth)
        self._sample_aim(env_ids)
        self._aim_timer[env_ids] = u(c.aim_interval)
        self._clock[env_ids] = 0.0
        self._weave_amp[env_ids] = u(c.weave_amp)
        self._weave_freq[env_ids] = u(c.weave_freq)
        self._weave_phase[env_ids] = u((0.0, 2.0 * math.pi))
        self._jink[env_ids] = 0.0
        self._jink_timer[env_ids] = u(c.jink_interval)
        self._speed_timer[env_ids] = u(c.speed_interval)
        self._attacker_speed[env_ids] = u(c.attacker_speed)

    # -- scripted attackers --------------------------------------------------------------

    def _scripted_attacker_actions(self) -> list[torch.Tensor]:
        c = self.cfg
        dt = self.physics_dt
        pos = self._local(self._attackers)
        n, a = self.num_envs, c.num_attackers
        all_envs = torch.arange(n, device=self.device)
        u = lambda r: sample_uniform(r[0], r[1], (n, a), self.device)  # noqa: E731

        # bait: in to bait_depth from the face, back out past bait_retreat_x, then dive at a new point
        dist = self.face.distance(pos)
        phase = self._bait_phase
        turn_out = (phase == BAIT_IN) & (dist < self._bait_depth)
        back = (phase == BAIT_OUT) & (pos[..., 0] > c.bait_retreat_x)
        phase = torch.where(turn_out, BAIT_OUT, torch.where(back, DIVE, phase))
        self._bait_phase = phase
        if back.any():
            self._sample_aim(all_envs, back)

        if c.aggressive:
            self._clock += dt
            self._aim_timer -= dt
            due = self._aim_timer <= 0.0
            if due.any():
                self._sample_aim(all_envs, due)
            self._aim_timer = torch.where(due, u(c.aim_interval), self._aim_timer)
            self._jink_timer -= dt
            due = self._jink_timer <= 0.0
            self._jink = torch.where(due, u((-c.jink_amp, c.jink_amp)), self._jink)
            self._jink_timer = torch.where(due, u(c.jink_interval), self._jink_timer)
            self._speed_timer -= dt
            due = self._speed_timer <= 0.0
            self._attacker_speed = torch.where(due, u(c.attacker_speed), self._attacker_speed)
            self._speed_timer = torch.where(due, u(c.speed_interval), self._speed_timer)

        aim = torch.cat([torch.zeros_like(self._aim[..., :1]), self._aim], -1)  # (N, A, 3)
        to_aim = aim - pos
        heading = to_aim / torch.linalg.norm(to_aim, dim=-1, keepdim=True).clamp(min=1e-6)
        if c.aggressive:
            side = torch.stack([-heading[..., 1], heading[..., 0], torch.zeros_like(heading[..., 0])], -1)
            side = side / torch.linalg.norm(side, dim=-1, keepdim=True).clamp(min=1e-6)
            weave = self._weave_amp * torch.sin(2.0 * math.pi * self._weave_freq * self._clock + self._weave_phase)
            heading = heading + (weave + self._jink).unsqueeze(-1) * side
            if c.dodge_gain > 0.0:
                d_pos = self._local(self._defenders)
                away = pos[:, :, None] - d_pos[:, None]  # (N, A, D, 3)
                gap = torch.linalg.norm(away, dim=-1, keepdim=True).clamp(min=1e-3)
                live = self._defender_alive[:, None, :, None].float()
                push = (((c.dodge_radius - gap) / c.dodge_radius).clamp(min=0.0) * away / gap * live).sum(2)
                heading = heading + c.dodge_gain * (push * side).sum(-1, keepdim=True) * side
            heading = heading / torch.linalg.norm(heading, dim=-1, keepdim=True).clamp(min=1e-6)
        retreat = torch.zeros_like(heading)
        retreat[..., 0] = 1.0
        heading = torch.where((phase == BAIT_OUT).unsqueeze(-1), retreat, heading)
        vel = heading * self._attacker_speed.unsqueeze(-1)

        # keep clear of each other: inside attacker_separation (judged on the gap a moment ahead),
        # drop the speed toward a live teammate (all of it by attacker_min_gap) and push apart.
        if c.attacker_separation > 0.0 and a > 1:
            away = pos[:, :, None] - pos[:, None]  # (N, A, A, 3)
            gap = torch.linalg.norm(away, dim=-1, keepdim=True).clamp(min=1e-3)
            away = away / gap
            other = (self._attacker_alive[:, None, :] & ~torch.eye(a, dtype=torch.bool, device=self.device)).unsqueeze(-1)
            # judge the gap attacker_react_s ahead at the actual closing speed: the drones lag their commands
            a_vel = torch.stack([o.data.root_lin_vel_w for o in self._attackers], 1)
            closing_now = (-((a_vel[:, :, None] - a_vel[:, None]) * away).sum(-1, keepdim=True)).clamp(min=0.0)
            ahead = gap - c.attacker_react_s * closing_now
            band = max(c.attacker_separation - c.attacker_min_gap, 1e-3)
            w = ((c.attacker_separation - ahead) / band).clamp(0.0, 1.0) * other.float()
            closing = (-(vel.unsqueeze(2) * away).sum(-1, keepdim=True)).clamp(min=0.0)
            vel = vel + (w * closing * away).sum(2) + c.attacker_separation_push * (w * away).sum(2)

        # the drones' limit: weave, dodge and separation push must not add speed past it
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
        vel = vel * (c.attacker_max_speed / speed.clamp(min=1e-6)).clamp(max=1.0)

        # downed: sent `land`
        landing = torch.zeros_like(vel)
        landing[..., 2] = -c.downed_descent_speed * (pos[..., 2] > c.downed_floor_height)
        vel = torch.where(self._attacker_alive.unsqueeze(-1), vel, landing)
        act = torch.cat([vel / c.command.max_lin_speed, torch.zeros_like(vel[..., :1])], -1).clamp(-1.0, 1.0)
        return [act[:, i] for i in range(a)]

    # -- downed defenders land -----------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        super()._pre_physics_step(actions)
        z = self._local(self._defenders)[..., 2]
        for i in range(self.cfg.num_defenders):
            key = f"defender_{i}"
            land = torch.zeros_like(self._actions[key])
            land[:, 2] = -self.cfg.downed_descent_speed / self.cfg.command.max_lin_speed * (z[:, i] > self.cfg.downed_floor_height)
            self._actions[key] = torch.where(self._defender_alive[:, i : i + 1], self._actions[key], land)
        if self.cfg.defender_takeoff:
            # the firmware's takeoff: straight up to its height, whatever the sticks say
            self.taking_off &= self._defender_alive & (z < self._takeoff_alt)
            climb = torch.zeros_like(self._actions["defender_0"])
            climb[:, 2] = self.cfg.takeoff_speed / self.cfg.command.max_lin_speed
            for i in range(self.cfg.num_defenders):
                key = f"defender_{i}"
                self._actions[key] = torch.where(self.taking_off[:, i : i + 1], climb, self._actions[key])

    # -- MDP -----------------------------------------------------------------------------

    def _out_of_arena(self, pos: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        out = torch.zeros(pos.shape[:-1], dtype=torch.bool, device=self.device)
        for k, (lo, hi) in enumerate((c.arena_x, c.arena_y, c.arena_z)):
            out |= (pos[..., k] < lo) | (pos[..., k] > hi)
        return out

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        c = self.cfg
        a_pos, d_pos = self._local(self._attackers), self._local(self._defenders)
        d_alive, a_alive = self._defender_alive, self._attacker_alive
        eye = lambda k: torch.eye(k, dtype=torch.bool, device=self.device)  # noqa: E731

        pa, pd = self._prev_a_pos, self._prev_d_pos
        da = self._closest_approach(pd, pa, d_pos, a_pos) < c.collision_radius
        da &= d_alive.unsqueeze(2) & a_alive.unsqueeze(1)
        dd = self._closest_approach(pd, pd, d_pos, d_pos) < c.collision_radius
        dd &= d_alive.unsqueeze(2) & d_alive.unsqueeze(1) & ~eye(c.num_defenders)
        aa = self._closest_approach(pa, pa, a_pos, a_pos) < c.collision_radius
        aa &= a_alive.unsqueeze(2) & a_alive.unsqueeze(1) & ~eye(c.num_attackers)
        self._prev_a_pos, self._prev_d_pos = a_pos.clone(), d_pos.clone()

        breach = a_alive & (a_pos[..., 0] < 0.0) & self.face.inside(a_pos)
        a_out = a_alive & (self._out_of_arena(a_pos) | ((a_pos[..., 0] < 0.0) & ~self.face.inside(a_pos)))
        d_out = d_alive & self._out_of_arena(d_pos) & ~self.taking_off  # on the floor before takeoff is not a crash

        captured = da.any(1)
        att_down = (captured | aa.any(-1) | a_out) & ~breach
        teamkill = dd.any(-1)
        def_down = da.any(2) | teamkill | d_out
        self._attacker_alive = a_alive & ~att_down
        self._defender_alive = d_alive & ~def_down
        self._breached = breach.any(-1)
        self._episode_captures += captured.float().sum(-1)
        self._ep_defenders_lost += def_down.float().sum(-1)
        self._ep_teamkills += teamkill.float().sum(-1)
        self._ep_out += a_out.float().sum(-1) + d_out.float().sum(-1)
        self._ep_att_collisions += (aa.any(-1) & ~captured & ~breach).float().sum(-1)
        speed = lambda objs, alive: (torch.linalg.norm(torch.stack([o.data.root_lin_vel_w for o in objs], 1), dim=-1)  # noqa: E731
                                     * alive.float()).max(-1).values
        self._ep_att_vmax = torch.maximum(self._ep_att_vmax, speed(self._attackers, a_alive))
        self._ep_def_vmax = torch.maximum(self._ep_def_vmax, speed(self._defenders, d_alive))

        rewards = {}
        for i in range(c.num_defenders):
            rewards[f"defender_{i}"] = (
                c.rew_scale_capture * da[:, i].float().sum(-1)
                + c.rew_scale_defender_lost * def_down[:, i].float()
                + c.rew_scale_teamkill * teamkill[:, i].float()
                + c.rew_scale_breach * self._breached.float()
            )
        return rewards

    def _get_dones(self):
        terminated, time_outs = super()._get_dones()
        finished = self.outcome_valid
        self.outcome_defenders_lost = torch.where(finished, self._ep_defenders_lost, self.outcome_defenders_lost)
        self.outcome_teamkills = torch.where(finished, self._ep_teamkills, self.outcome_teamkills)
        self.outcome_out = torch.where(finished, self._ep_out, self.outcome_out)
        self.outcome_att_collisions = torch.where(finished, self._ep_att_collisions, self.outcome_att_collisions)
        self.outcome_att_max_speed = torch.where(finished, self._ep_att_vmax, self.outcome_att_max_speed)
        self.outcome_def_max_speed = torch.where(finished, self._ep_def_vmax, self.outcome_def_max_speed)
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)  # controllers, alive flags, episode buffers (and ring starts, replaced below)
        c = self.cfg
        env_ids = torch.as_tensor(env_ids, device=self.device)
        num = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        jit = lambda: sample_uniform(-c.start_jitter, c.start_jitter, (num,), self.device)  # noqa: E731
        for i, drone in enumerate(self._defenders):
            state = drone.data.default_root_state[env_ids].clone()
            state[:, 0] = origins[:, 0] + c.defender_start_x + jit()
            state[:, 1] = origins[:, 1] + c.defender_start_y[i] + jit()
            state[:, 2] = origins[:, 2] + (c.floor_start_z if c.defender_takeoff else c.defender_start_z + jit())
            self._prev_d_pos[env_ids, i] = state[:, :3] - origins
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
        # attacker starts: uniform in the start box, redrawn (per env) until attacker_start_separation apart
        lo = torch.tensor([r[0] for r in (c.attacker_start_x, c.attacker_start_y, c.attacker_start_z)], device=self.device)
        hi = torch.tensor([r[1] for r in (c.attacker_start_x, c.attacker_start_y, c.attacker_start_z)], device=self.device)
        start = lo + (hi - lo) * torch.rand(num, c.num_attackers, 3, device=self.device)
        pair_i, pair_j = torch.triu_indices(c.num_attackers, c.num_attackers, 1, device=self.device)
        for _ in range(50):
            close = (torch.linalg.norm(start[:, pair_i] - start[:, pair_j], dim=-1) < c.attacker_start_separation).any(-1)
            if not close.any():
                break
            start[close] = lo + (hi - lo) * torch.rand(int(close.sum()), c.num_attackers, 3, device=self.device)
        for i, drone in enumerate(self._attackers):
            state = drone.data.default_root_state[env_ids].clone()
            state[:, :3] = origins + start[:, i]
            self._prev_a_pos[env_ids, i] = start[:, i]
            # face the defenders (yaw 180 deg) so the scene reads as two teams facing each other
            state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self._defender_alive[env_ids] = True
        self.taking_off[env_ids] = c.defender_takeoff
        self._takeoff_alt[env_ids] = sample_uniform(*c.takeoff_height, (num, c.num_defenders), self.device)
        self._ep_defenders_lost[env_ids] = 0.0
        self._ep_teamkills[env_ids] = 0.0
        self._ep_out[env_ids] = 0.0
        self._ep_att_collisions[env_ids] = 0.0
        self._ep_att_vmax[env_ids] = 0.0
        self._ep_def_vmax[env_ids] = 0.0
        self._sample_behaviour(env_ids)

