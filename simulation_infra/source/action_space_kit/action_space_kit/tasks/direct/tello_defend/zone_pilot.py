"""A classical 3-vs-2 defender: zones with hysteresis, assignment, tag-keeping heading, safety spheres.

No learning. Every tick, for every environment (batched in torch, no Isaac imports):

  zones       each attacker is OUTSIDE / ALERT / ENGAGE by its horizontal distance to the
              site centre, Schmitt-trigger style: it enters a zone at ``r_*_in`` and leaves
              it only past ``r_*_out``, and a change must persist ``debounce_s`` first
  assignment  attackers in ALERT or deeper are matched one-to-one to surviving defenders by
              minimum total time-to-intercept. Every partial assignment is enumerated (each
              attacker gets a distinct defender or none: 13 for 3 vs 2); leaving an active
              attacker uncovered costs ``unassigned_penalty_s``, a downed defender can't be
              picked, so with fewer defenders than attackers the best ones to cover win.
              The current assignment is kept unless a new one is ``switch_margin_s`` better
              and ``min_hold_s`` has passed, or the active attackers or the surviving
              defenders changed
  roles       unassigned defenders hold their posts (a single spare survivor, while any
              attacker is active and another defender is engaged, guards the site as goalie); an assigned defender
              shadows its attacker on the site-attacker line while it is in ALERT and flies
              at its intercept point once it is in ENGAGE. No target is ever further than
              ``leash_radius`` from the site, so an attacker that turns back cannot bait a
              defender out
  spheres     each defender carries a safety sphere: inside ``2 * sphere_radius +
              sphere_margin`` of another defender the closing velocity (relative, so a
              drone coming at a hovering one counts) is removed, and any overlap pushes them
              apart. Attackers get keep-out spheres too: small around a live one (capture
              still happens), larger around a downed one
  heading     yaw is independent of translation: each defender turns toward the heading
              that sees the most mapped tags (``tags/sensing.py`` camera model), preferring
              the closest acceptable heading, and turns slowly (blur-free) while it has a tag

Positions are env-local (the site is at the origin, z up from the floor). Outputs are
velocities in the room frame [m/s] and a counter-clockwise yaw rate [rad/s];
:func:`rc_command` turns them into the Tello's body-frame ``rc a b c d``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

from action_space_kit.tags.sensing import TagSensorCfg, camera_axes, decode_rate, detectable

OUTSIDE, ALERT, ENGAGE = 0, 1, 2


@dataclass
class ZonePilotCfg:
    # - zones: horizontal distance from the site centre [m]; enter at *_in, leave past *_out
    r_breach: float = 1.5
    r_engage_in: float = 3.0
    r_engage_out: float = 3.4
    r_alert_in: float = 4.5
    r_alert_out: float = 4.9
    debounce_s: float = 0.15
    """A zone change must hold this long before it counts (one noisy fix can't trigger it)."""

    # - where defenders wait and how far they may go
    post_radius: float = 2.2
    post_height: float = 1.5
    post_angles_deg: tuple[float, ...] = (0.0, 120.0, 240.0)
    leash_radius: float = 3.4
    """No defender target is ever further than this from the site (the anti-bait rule)."""
    shadow_radius: float = 2.7
    """While its attacker is in ALERT, an assigned defender waits on the site-attacker line here."""
    goalie_radius: float = 1.2
    z_min: float = 0.8
    z_max: float = 2.6

    # - assignment
    intercept_speed: float = 0.9
    """[m/s] defender speed assumed when solving for the intercept point (below the 1 m/s cap: lag)."""
    switch_margin_s: float = 0.5
    min_hold_s: float = 1.0
    leash_penalty_s: float = 5.0
    """Added to the cost of a pairing whose intercept point lies outside the leash."""
    unassigned_penalty_s: float = 100.0
    """Cost of leaving an active attacker without a defender (only when none can be spared)."""

    # - flight
    max_speed: float = 1.0
    gain: float = 1.5

    # - safety sphere around every defender
    sphere_radius: float = 0.5
    """[m] centres are kept at least twice this apart."""
    sphere_margin: float = 0.4
    """[m] the approaching velocity fades to zero over this band outside the spheres."""
    sphere_push: float = 2.0
    """[m/s per m] of overlap, pushing overlapping defenders apart."""
    downed_clearance: float = 0.6
    """[m] keep-out radius around a downed attacker (in sim it keeps hovering where it was hit)."""
    attacker_clearance: float = 0.2
    """[m] keep-out radius around a live attacker: inside the 0.35 m capture radius, so it never
    blocks a capture, but stops a defender flying straight through it."""
    attacker_margin: float = 0.8
    """[m] braking band outside an attacker's keep-out radius. Wider than ``sphere_margin``:
    head-on, defender and attacker close at ~1.9 m/s, and stopping takes ~0.3 m plus a tick."""

    # - heading: keep a tag in view
    yaw_candidates: int = 36
    yaw_keep_frac: float = 0.8
    """Any heading scoring at least this fraction of the best is acceptable; take the closest."""
    yaw_gain: float = 1.0
    yaw_rate_tracking: float = 0.2
    """[rad/s] turn cap while a tag is in view: faster turns blur frames (``TagSensorCfg.blur_free_rate``)."""
    yaw_rate_search: float = 0.8
    """[rad/s] turn cap when no tag is in view."""

    # - room: hardcoded until the competition map arrives
    room_half: float = 6.5
    """[m] square room, walls at +/- this from the site."""
    tag_height: float = 1.25
    tag_offsets: tuple[float, ...] = (-3.0, 0.0, 3.0)
    """[m] tags along every wall; the centre one on the +x wall is the big tag 10."""
    small_tag: float = 0.132
    big_tag: float = 0.5715


def room_tags(cfg: ZonePilotCfg, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The hardcoded tag layout: (pos (T, 3), inward normal (T, 3), size (T,)), env-local."""
    h = cfg.room_half
    pos, normal, size = [], [], []
    for wall, (n, along) in enumerate((((-1, 0), (0, 1)), ((1, 0), (0, -1)), ((0, -1), (-1, 0)), ((0, 1), (1, 0)))):
        # wall 0 is +x (normal -x), 1 is -x, 2 is +y, 3 is -y
        for off in cfg.tag_offsets:
            pos.append((-n[0] * h + along[0] * off, -n[1] * h + along[1] * off, cfg.tag_height))
            normal.append((float(n[0]), float(n[1]), 0.0))
            size.append(cfg.big_tag if wall == 0 and off == 0.0 else cfg.small_tag)
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
    return t(pos), t(normal), t(size)


def yaw_rotation(yaw: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) body-to-world rotation for a level drone with this yaw."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def tag_score(sensor: TagSensorCfg, pos: torch.Tensor, rot: torch.Tensor, tags) -> torch.Tensor:
    """Expected number of tags decoded per frame, for cameras at pos (M, 3) with body rotation (M, 3, 3)."""
    tag_pos, tag_normal, tag_size = tags
    cam_pos = pos + rot @ torch.tensor(sensor.camera_offset, device=pos.device)
    mask, rng = detectable(sensor, cam_pos, camera_axes(rot, sensor.camera_pitch), tag_pos, tag_normal, tag_size)
    d = tag_pos.unsqueeze(0) - cam_pos.unsqueeze(1)
    cos_view = -(d * tag_normal.unsqueeze(0)).sum(-1) / rng
    return (mask.float() * decode_rate(sensor, cos_view)).sum(-1)


def intercept(p_def: torch.Tensor, p_att: torch.Tensor, v_att: torch.Tensor, speed: float):
    """Earliest meeting of a defender flying at ``speed`` with a constant-velocity attacker.

    Shapes broadcast: (N, D, 1, 3) with (N, 1, A, 3). Returns (time (N, D, A), point (N, D, A, 3)).
    Where no meeting exists (attacker faster and moving away) the time is the straight-line
    distance over speed and the point is the attacker's current position: pure pursuit.
    """
    r = p_att - p_def
    a = (v_att * v_att).sum(-1) - speed**2
    b = 2.0 * (r * v_att).sum(-1)
    c = (r * r).sum(-1)
    disc = b * b - 4.0 * a * c
    sq = disc.clamp(min=0.0).sqrt()
    safe_a = torch.where(a.abs() < 1e-6, torch.full_like(a, -1e-6), a)
    t1, t2 = (-b - sq) / (2.0 * safe_a), (-b + sq) / (2.0 * safe_a)
    big = torch.full_like(c, float("inf"))
    t = torch.minimum(torch.where(t1 > 0, t1, big), torch.where(t2 > 0, t2, big))
    t = torch.where(disc >= 0, t, big)
    ok = torch.isfinite(t)
    pursuit = c.sqrt() / speed
    point = torch.where(ok.unsqueeze(-1), p_att + v_att * t.nan_to_num(0.0, posinf=0.0).unsqueeze(-1), p_att)
    return torch.where(ok, t, pursuit), point


def rc_command(vel_room: torch.Tensor, yaw_rate_ccw: torch.Tensor, yaw: torch.Tensor, max_speed: float = 1.0,
               max_yaw_rate: float = 1.745) -> torch.Tensor:
    """Room-frame velocity + CCW yaw rate -> the real Tello's ``rc a b c d`` in [-100, 100].

    The Tello's sticks are body-frame (a left, b forward, c up, d clockwise), so the velocity
    is rotated into the drone's heading. (The sim controller takes room-frame velocity and a
    CCW rate directly; this is the deploy-side conversion.)
    """
    c, s = torch.cos(yaw), torch.sin(yaw)
    fwd = c * vel_room[..., 0] + s * vel_room[..., 1]
    left = -s * vel_room[..., 0] + c * vel_room[..., 1]
    rc = torch.stack([left / max_speed, fwd / max_speed, vel_room[..., 2] / max_speed, -yaw_rate_ccw / max_yaw_rate], -1)
    return (rc * 100.0).round().clamp(-100.0, 100.0)


class ZonePilot:
    """Stateful, batched: call :meth:`step` once per command tick, :meth:`reset` on episode resets."""

    def __init__(self, cfg: ZonePilotCfg, num_envs: int, num_defenders: int, num_attackers: int, tick_s: float,
                 device: str, sensor: TagSensorCfg | None = None):
        if len(cfg.post_angles_deg) != num_defenders:
            raise ValueError("post_angles_deg needs one angle per defender")
        self.cfg, self.n, self.nd, self.na, self.tick_s, self.device = (
            cfg, num_envs, num_defenders, num_attackers, tick_s, device)
        self.sensor = sensor or TagSensorCfg()
        self.tags = room_tags(cfg, device)
        ang = torch.tensor(cfg.post_angles_deg, device=device) * math.pi / 180.0
        self.posts = torch.stack(
            [cfg.post_radius * torch.cos(ang), cfg.post_radius * torch.sin(ang), torch.full_like(ang, cfg.post_height)], -1)
        # every partial one-to-one attacker -> defender map, (P, A); num_defenders means "none"
        maps = [m for m in itertools.product(range(num_defenders + 1), repeat=num_attackers)
                if len({d for d in m if d < num_defenders}) == sum(d < num_defenders for d in m)]
        self.perms = torch.tensor(maps, device=device)
        self.debounce_ticks = max(1, round(cfg.debounce_s / tick_s))
        self.cand_yaw = torch.arange(cfg.yaw_candidates, device=device) * (2.0 * math.pi / cfg.yaw_candidates)

        self.att_state = torch.zeros(num_envs, num_attackers, dtype=torch.long, device=device)
        self._pending = torch.zeros_like(self.att_state)
        self._pending_ticks = torch.zeros_like(self.att_state)
        self.perm_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._perm_age = torch.zeros(num_envs, device=device)
        self._prev_active = torch.zeros(num_envs, num_attackers, dtype=torch.bool, device=device)
        self._prev_def_alive = torch.ones(num_envs, num_defenders, dtype=torch.bool, device=device)
        self.role = torch.zeros(num_envs, num_defenders, dtype=torch.long, device=device)
        """Per defender: -1 hold, -2 goalie, -3 downed, else the attacker index it is assigned to."""

    def reset(self, env_ids: torch.Tensor) -> None:
        self.att_state[env_ids] = OUTSIDE
        self._pending[env_ids] = OUTSIDE
        self._pending_ticks[env_ids] = 0
        self.perm_idx[env_ids] = -1
        self._perm_age[env_ids] = 0.0
        self._prev_active[env_ids] = False
        self._prev_def_alive[env_ids] = True

    # -- zones -------------------------------------------------------------------

    def _update_zones(self, att_pos: torch.Tensor, alive: torch.Tensor) -> None:
        c = self.cfg
        r = torch.linalg.norm(att_pos[..., :2], dim=-1)
        s = self.att_state
        from_out = torch.where(r < c.r_engage_in, ENGAGE, torch.where(r < c.r_alert_in, ALERT, OUTSIDE))
        from_alert = torch.where(r < c.r_engage_in, ENGAGE, torch.where(r > c.r_alert_out, OUTSIDE, ALERT))
        from_engage = torch.where(r > c.r_alert_out, OUTSIDE, torch.where(r > c.r_engage_out, ALERT, ENGAGE))
        want = torch.where(s == OUTSIDE, from_out, torch.where(s == ALERT, from_alert, from_engage))
        changing = want != s
        same_pending = changing & (want == self._pending)
        self._pending_ticks = torch.where(same_pending, self._pending_ticks + 1, changing.long())
        self._pending = torch.where(changing, want, s)
        commit = changing & (self._pending_ticks >= self.debounce_ticks)
        self.att_state = torch.where(commit, want, s)
        self._pending_ticks = torch.where(commit, 0, self._pending_ticks)
        # a downed attacker is out of the game at once
        self.att_state = torch.where(alive, self.att_state, OUTSIDE)
        self._pending = torch.where(alive, self._pending, OUTSIDE)
        self._pending_ticks = torch.where(alive, self._pending_ticks, 0)

    # -- assignment --------------------------------------------------------------

    def _assign(self, cost: torch.Tensor, active: torch.Tensor, def_alive: torch.Tensor) -> None:
        """cost (N, D, A) seconds; picks self.perm_idx (N,) with hysteresis."""
        c = self.cfg
        a_idx = torch.arange(self.na, device=self.device)
        # a downed defender is never picked (dearer than leaving the attacker uncovered)
        cost = torch.where(def_alive.unsqueeze(-1), cost, torch.full_like(cost, 1e4))
        none = torch.full_like(cost[:, :1], c.unassigned_penalty_s)
        per = torch.cat([cost, none], 1)[:, self.perms, a_idx]  # (N, P, A)
        total = (per * active.unsqueeze(1).float()).sum(-1)  # (N, P)
        best_cost, best = total.min(dim=-1)
        cur = total.gather(1, self.perm_idx.clamp(min=0).unsqueeze(1)).squeeze(1)
        changed = (active != self._prev_active).any(dim=-1) | (def_alive != self._prev_def_alive).any(dim=-1)
        better = (best_cost < cur - c.switch_margin_s) & (self._perm_age >= c.min_hold_s)
        switch = (self.perm_idx < 0) | changed | better
        self.perm_idx = torch.where(switch, best, self.perm_idx)
        self._perm_age = torch.where(switch, torch.zeros_like(self._perm_age), self._perm_age + self.tick_s)
        self._prev_active = active.clone()
        self._prev_def_alive = def_alive.clone()

    # -- tick --------------------------------------------------------------------

    def step(self, def_pos, def_vel, def_quat, att_pos, att_vel, alive, has_tag=None, def_alive=None):
        """One command tick. Positions env-local.

        Args:
            def_pos, def_vel: (N, D, 3).  def_quat: (N, D, 4) w, x, y, z.
            att_pos, att_vel: (N, A, 3).  alive: (N, A) bool.
            has_tag: (N, D) bool, whether each defender has a tag fix now (None: assume yes).
            def_alive: (N, D) bool, defenders still in the game (None: all).
        Returns:
            vel (N, D, 3) room frame [m/s], yaw_rate (N, D) CCW [rad/s].
        """
        c = self.cfg
        n, nd = self.n, self.nd
        self._update_zones(att_pos, alive)
        active = alive & (self.att_state >= ALERT)
        if def_alive is None:
            def_alive = torch.ones(n, nd, dtype=torch.bool, device=self.device)

        t_hit, p_hit = intercept(def_pos.unsqueeze(2), att_pos.unsqueeze(1), att_vel.unsqueeze(1), c.intercept_speed)
        outside_leash = torch.linalg.norm(p_hit[..., :2], dim=-1) > c.leash_radius
        self._assign(t_hit + c.leash_penalty_s * outside_leash.float(), active, def_alive)

        # roles
        perm = self.perms[self.perm_idx]  # (N, A) defender per attacker
        role = torch.full((n, nd), -1, dtype=torch.long, device=self.device)
        for a in range(self.na):
            hit = (torch.arange(nd, device=self.device).unsqueeze(0) == perm[:, a : a + 1]) & active[:, a : a + 1]
            role = torch.where(hit, a, role)
        role = torch.where(def_alive, role, -3)
        spare = role == -1
        goalie = spare & (spare.sum(-1, keepdim=True) == 1) & (role >= 0).any(-1, keepdim=True)
        role = torch.where(goalie, -2, role)
        self.role = role

        # targets
        target = self.posts.unsqueeze(0).expand(n, -1, -1).clone()
        a_of = role.clamp(min=0)
        tgt_att = att_pos.gather(1, a_of.unsqueeze(-1).expand(-1, -1, 3))  # (N, D, 3)
        tgt_state = self.att_state.gather(1, a_of)
        tgt_hit = p_hit.gather(2, a_of.view(n, nd, 1, 1).expand(-1, -1, 1, 3)).squeeze(2)
        assigned = role >= 0

        r_att = torch.linalg.norm(tgt_att[..., :2], dim=-1, keepdim=True).clamp(min=1e-3)
        dir_att = tgt_att[..., :2] / r_att
        shadow = torch.cat([dir_att * r_att.clamp(max=c.shadow_radius), tgt_att[..., 2:]], -1)
        engage = tgt_hit
        point = torch.where((tgt_state == ENGAGE).unsqueeze(-1), engage, shadow)
        target = torch.where(assigned.unsqueeze(-1), point, target)

        # goalie: between the site and the nearest active attacker
        r_all = torch.linalg.norm(att_pos[..., :2], dim=-1).masked_fill(~active, float("inf"))
        near = att_pos.gather(1, r_all.argmin(-1).view(n, 1, 1).expand(-1, 1, 3)).squeeze(1)  # (N, 3)
        near_dir = near[:, :2] / torch.linalg.norm(near[:, :2], dim=-1, keepdim=True).clamp(min=1e-3)
        goal_pt = torch.cat([near_dir * c.goalie_radius, torch.full((n, 1), c.post_height, device=self.device)], -1)
        target = torch.where(goalie.unsqueeze(-1), goal_pt.unsqueeze(1), target)

        # the leash and the height band
        r_t = torch.linalg.norm(target[..., :2], dim=-1, keepdim=True).clamp(min=1e-6)
        target = torch.cat([target[..., :2] * (c.leash_radius / r_t).clamp(max=1.0),
                            target[..., 2:].clamp(c.z_min, c.z_max)], -1)

        self.target = target  # (N, D, 3), for visualisation

        # velocity: full speed at an engaged intercept point, proportional otherwise
        err = target - def_pos
        vel = c.gain * err
        dist = torch.linalg.norm(err, dim=-1, keepdim=True).clamp(min=1e-6)
        sprint = (assigned & (tgt_state == ENGAGE)).unsqueeze(-1) & (dist > 0.05)
        vel = torch.where(sprint, err / dist * c.max_speed, vel)
        vel = vel / (torch.linalg.norm(vel, dim=-1, keepdim=True) / c.max_speed).clamp(min=1.0)

        vel = self._spheres(def_pos, vel, def_vel, att_pos, att_vel, alive)

        yaw_rate = self._heading(def_pos, def_quat, has_tag)
        return vel, yaw_rate

    def _spheres(self, pos, vel, def_vel, att_pos, att_vel, alive) -> torch.Tensor:
        """Remove closing speed toward other defenders and attackers inside their braking bands."""
        c = self.cfg
        n, nd = self.n, self.nd
        # obstacles per defender: the other defenders, then every attacker
        obs = torch.cat([pos, att_pos], 1).unsqueeze(1)  # (N, 1, D + A, 3)
        obs_vel = torch.cat([def_vel, att_vel], 1).unsqueeze(1)
        clear = torch.cat([torch.full((n, nd), 2.0 * c.sphere_radius, device=self.device),
                           torch.where(alive, c.attacker_clearance, c.downed_clearance)], 1).view(n, 1, -1, 1)
        margin = torch.cat([torch.full((n, nd), c.sphere_margin, device=self.device),
                            torch.full((n, self.na), c.attacker_margin, device=self.device)], 1).view(n, 1, -1, 1)
        diff = pos.unsqueeze(2) - obs  # (N, D, D + A, 3): from the obstacle to the defender
        dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=1e-3)
        away = diff / dist
        self_mask = torch.zeros(nd, nd + self.na, dtype=torch.bool, device=self.device)
        self_mask[:, :nd] = torch.eye(nd, dtype=torch.bool, device=self.device)
        self_mask = self_mask.view(1, nd, -1, 1)
        w = ((clear + margin - dist) / margin).clamp(0.0, 1.0).masked_fill(self_mask, 0.0)
        closing = (-((vel.unsqueeze(2) - obs_vel) * away).sum(-1, keepdim=True)).clamp(min=0.0)
        vel = vel + (w * closing * away).sum(2)
        overlap = (clear - dist).clamp(min=0.0).masked_fill(self_mask, 0.0)
        vel = vel + (c.sphere_push * overlap * away).sum(2)
        return vel / (torch.linalg.norm(vel, dim=-1, keepdim=True) / c.max_speed).clamp(min=1.0)

    def _heading(self, pos: torch.Tensor, quat: torch.Tensor, has_tag) -> torch.Tensor:
        c = self.cfg
        n, nd, k = self.n, self.nd, c.yaw_candidates
        yaw = yaw_from_quat(quat)  # (N, D)
        p = pos.unsqueeze(2).expand(-1, -1, k, -1).reshape(-1, 3)
        rot = yaw_rotation(self.cand_yaw).unsqueeze(0).expand(n * nd, -1, -1, -1).reshape(-1, 3, 3)
        score = tag_score(self.sensor, p, rot, self.tags).view(n, nd, k)
        best = score.max(-1, keepdim=True).values
        turn = wrap(self.cand_yaw.view(1, 1, k) - yaw.unsqueeze(-1))
        ok = (score >= c.yaw_keep_frac * best) & (score > 0)
        pick = turn.abs().masked_fill(~ok, float("inf")).argmin(-1, keepdim=True)
        err = turn.gather(-1, pick).squeeze(-1)
        # nothing visible from anywhere here: keep turning to search
        err = torch.where(best.squeeze(-1) > 0, err, torch.full_like(err, math.pi))
        if has_tag is None:
            has_tag = torch.ones_like(yaw, dtype=torch.bool)
        cap = torch.where(has_tag, c.yaw_rate_tracking, c.yaw_rate_search)
        return torch.maximum(torch.minimum(c.yaw_gain * err, cap), -cap)
