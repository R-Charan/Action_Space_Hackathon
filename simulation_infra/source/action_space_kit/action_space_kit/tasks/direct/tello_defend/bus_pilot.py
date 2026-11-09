"""'Park the bus': a classical defender for the breach face that does not predict the attackers.

Attackers and defenders are identical drones, so this is Isaacs' target-guarding game with
equal speeds: a defender guarantees that an attacker cannot reach the face while every point
of the face is closer to the defender than to the attacker (``Face.margin`` > 0). The pilot
keeps that margin up instead of chasing predicted intercept points, so weaving, jinks and
feints buy the attacker nothing; to get through it has to come within capture range.

Every command tick (batched in torch, no Isaac imports):

  telemetry   :class:`AttackerTracker` turns delayed, noisy attacker poses + alive flags (what
              the hardware round provides) into position and velocity estimates (constant-
              velocity Kalman filter per axis, extrapolated over the delay)
  zones       each attacker is OUTSIDE / ALERT / ENGAGE by its distance to the face rectangle,
              Schmitt-trigger style (enter at ``r_*_in``, leave past ``r_*_out``, debounced)
  assignment  attackers in ALERT or deeper get one surviving defender each, choosing among all
              partial assignments the one whose worst guarding margin is best (bottleneck);
              kept unless another is ``switch_margin`` better after ``min_hold_s``, or the
              active attackers / surviving defenders changed
  guarding    an assigned defender moves to the reachable point (``lookahead_s`` ahead, 26
              directions) that maximises its margin against its attacker, capped at
              ``margin_cap`` above which it drifts back to its post; spare survivors guard
              every active attacker at once (maximise the smallest margin); with nothing
              active everyone holds its post in a line in front of the face. Defenders plan
              one after another and avoid the spots already planned, which stacks two guards
              on one bearing at different depths or heights instead of colliding
  strike      in ENGAGE, an attacker within ``strike_radius`` of its guard is attacked
              directly (the trade the rules allow: both are downed)
  spheres     the safety-sphere avoidance and tag-keeping heading of ``zone_pilot``

Outputs: room-frame velocity [m/s] and CCW yaw rate [rad/s]; ``rc_command`` converts for the Tello.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

from action_space_kit.tags.sensing import TagSensorCfg

from .face_geometry import Face
from .zone_pilot import ALERT, ENGAGE, OUTSIDE, intercept, rc_command, tag_score, wrap, yaw_from_quat, yaw_rotation  # noqa: F401


# -- telemetry ------------------------------------------------------------------------------


@dataclass
class TelemetryCfg:
    rate_hz: float = 20.0
    """Attacker pose messages per second."""
    delay_s: float = 0.2
    """[s] age of a pose when it reaches the defender's laptop."""
    noise: float = 0.03
    """[m] std-dev of each published position, per axis."""
    process_accel: float = 4.0
    """[m/s^2] Kalman process noise: how hard an attacker may accelerate between messages."""
    max_speed: float = 1.5
    """[m/s] estimated speeds are capped here (the drones' limit)."""


class AttackerTracker:
    """Delayed, noisy attacker poses in; filtered position and velocity estimates out. Batched."""

    def __init__(self, cfg: TelemetryCfg, num_envs: int, num_attackers: int, step_dt: float, device: str):
        self.cfg, self.dt, self.device = cfg, step_dt, device
        self.delay_steps = max(0, round(cfg.delay_s / step_dt))
        self.pub_every = max(1, round(1.0 / (cfg.rate_hz * step_dt)))
        n, a = num_envs, num_attackers
        self._ring_pos = torch.zeros(self.delay_steps + 1, n, a, 3, device=device)
        self._ring_alive = torch.ones(self.delay_steps + 1, n, a, dtype=torch.bool, device=device)
        self._head = 0
        self._tick = 0
        self.p = torch.zeros(n, a, 3, device=device)
        self.v = torch.zeros(n, a, 3, device=device)
        self.P = torch.zeros(n, a, 3, 2, 2, device=device)
        self.age = torch.zeros(n, a, device=device)  # since the last message
        self.alive = torch.ones(n, a, dtype=torch.bool, device=device)
        self._init = torch.zeros(n, a, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor, pos: torch.Tensor, alive: torch.Tensor) -> None:
        """Start these envs over from their current true state (pos (len, A, 3), alive (len, A))."""
        self._ring_pos[:, env_ids] = pos.unsqueeze(0)
        self._ring_alive[:, env_ids] = alive.unsqueeze(0)
        self._init[env_ids] = False
        self.v[env_ids] = 0.0
        self.p[env_ids] = pos
        self.age[env_ids] = 0.0
        self.alive[env_ids] = alive

    def step(self, pos: torch.Tensor, alive: torch.Tensor) -> None:
        """Advance one env step with the true attacker state (N, A, 3) / (N, A)."""
        c = self.cfg
        L = self.delay_steps + 1
        self._head = (self._head + 1) % L
        self._ring_pos[self._head] = pos
        self._ring_alive[self._head] = alive
        self._tick += 1
        self.age += self.dt
        if self._tick % self.pub_every:
            return
        oldest = (self._head + 1) % L  # delay_steps ago
        z = self._ring_pos[oldest] + c.noise * torch.randn_like(pos)
        self.alive = self._ring_alive[oldest].clone()
        r = c.noise**2
        # predict over the time since the last message, then correct
        dt = self.age.unsqueeze(-1)  # (N, A, 1)
        p = self.p + self.v * dt
        q = c.process_accel**2
        P = self.P.clone()
        P00 = P[..., 0, 0] + dt * (P[..., 1, 0] + P[..., 0, 1]) + dt * dt * P[..., 1, 1] + q * dt**4 / 4
        P01 = P[..., 0, 1] + dt * P[..., 1, 1] + q * dt**3 / 2
        P11 = P[..., 1, 1] + q * dt**2
        S = P00 + r
        k0, k1 = P00 / S, P01 / S
        y = z - p
        p = p + k0 * y
        v = self.v + k1 * y
        P[..., 0, 0] = (1 - k0) * P00
        P[..., 0, 1] = (1 - k0) * P01
        P[..., 1, 0] = P[..., 0, 1]
        P[..., 1, 1] = P11 - k1 * P01
        # first message (or after a reset): start at the measurement, at rest, unsure of speed
        fresh = (~self._init).unsqueeze(-1)
        p = torch.where(fresh, z, p)
        v = torch.where(fresh, torch.zeros_like(v), v)
        P0 = torch.zeros_like(P)
        P0[..., 0, 0], P0[..., 1, 1] = r, 1.0
        P = torch.where(fresh[..., None, None], P0, P)
        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        v = v * (c.max_speed / speed.clamp(min=1e-6)).clamp(max=1.0)
        self.p, self.v, self.P = p, v, P
        self._init[:] = True
        self.age.zero_()

    def estimate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(position now (N, A, 3), velocity (N, A, 3), alive (N, A)): the filter state carried over the delay."""
        ahead = (self.age + self.delay_steps * self.dt).unsqueeze(-1)
        return self.p + self.v * ahead, self.v, self.alive


# -- the pilot ------------------------------------------------------------------------------


@dataclass
class BusPilotCfg:
    # - zones: distance from the face rectangle [m]; enter at *_in, leave past *_out
    r_engage_in: float = 1.5
    r_engage_out: float = 1.9
    r_alert_in: float = 4.0
    r_alert_out: float = 4.5
    debounce_s: float = 0.15

    # - the bus: posts in a line in front of the face, and the box defenders stay in
    post_depth: float = 0.8
    post_y: tuple[float, ...] = (-1.0, 0.0, 1.0)
    post_height: float = 1.5
    x_min: float = 0.3
    """[m] never closer to the face plane than this (the box is behind it)."""
    x_max: float = 3.0
    """[m] the leash: never further out than this, whatever an attacker does."""
    y_max: float = 3.0
    y_min: float | None = None
    """[m] None: -y_max. The real room is not symmetric about the face (room_layout.pilot_y_range)."""
    z_min: float = 0.6
    z_max: float = 2.8

    # - guarding
    face_grid: tuple[int, int] = (7, 5)
    lookahead_s: float = 0.3
    """Candidate positions are this far ahead at full speed."""
    margin_cap: float = 1.5
    """[m] margin beyond this is worth nothing more: the guard drifts back to its post instead."""
    post_weight: float = 0.05
    strike_radius: float = 1.3
    strike_lead_s: float = 0.4
    """[s] longest the strike leads the attacker's estimated velocity (head-on intercept)."""
    max_speed: float = 1.5
    gain: float = 2.0

    # - assignment
    switch_margin: float = 0.3
    """[m] of bottleneck margin a new assignment must gain to replace the current one."""
    min_hold_s: float = 0.5
    unassigned_penalty: float = 100.0

    # - safety spheres (as zone_pilot)
    sphere_radius: float = 0.35
    sphere_margin: float = 0.3
    sphere_push: float = 2.0
    downed_clearance: float = 0.6
    attacker_clearance: float = 0.2
    attacker_margin: float = 0.8
    avoid_attackers: bool = False
    """Brake for attackers too. Off: under mutual kill a collision is a 1-for-1 trade, which
    the side with the spare defender wins, and braking froze strikes 0.4 m short of contact."""
    conflict_weight: float = 10.0
    """Planning penalty per metre a candidate intrudes on a teammate's planned sphere."""

    # - heading (as zone_pilot)
    yaw_candidates: int = 36
    yaw_keep_frac: float = 0.8
    yaw_gain: float = 1.0
    yaw_rate_tracking: float = 0.2
    yaw_rate_search: float = 0.8

    # - stand-in room for the open-arena tasks (the room tasks pass map.yaml's tags instead):
    #   walls at these bounds, tags along every wall
    room_x: tuple[float, float] = (-1.5, 8.0)
    room_y: tuple[float, float] = (-5.0, 5.0)
    tag_height: float = 1.25
    tag_spacing: float = 2.5
    small_tag: float = 0.132
    big_tag: float = 0.5715


def box_room_tags(cfg: BusPilotCfg, device: str):
    """Tags every ``tag_spacing`` along the four walls, facing in; the big tag 10 mid far wall (+x)."""
    (x0, x1), (y0, y1) = cfg.room_x, cfg.room_y
    pos, normal, size = [], [], []

    def along(a, b):
        k = max(1, int((b - a) / cfg.tag_spacing))
        return [a + (b - a) * (i + 0.5) / k for i in range(k)]

    for x, nx in ((x1, -1.0), (x0, 1.0)):
        for y in along(y0, y1):
            pos.append((x, y, cfg.tag_height)); normal.append((nx, 0.0, 0.0)); size.append(cfg.small_tag)
    for y, ny in ((y1, -1.0), (y0, 1.0)):
        for x in along(x0, x1):
            pos.append((x, y, cfg.tag_height)); normal.append((0.0, ny, 0.0)); size.append(cfg.small_tag)
    far = [i for i, (p, nrm) in enumerate(zip(pos, normal)) if nrm[0] < 0]
    mid = min(far, key=lambda i: abs(pos[i][1]))
    size[mid] = cfg.big_tag
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
    return t(pos), t(normal), t(size)


def _directions(device) -> torch.Tensor:
    """(27, 3): zero and the 26 unit directions of a cube's faces, edges and corners."""
    d = torch.tensor([v for v in itertools.product((-1.0, 0.0, 1.0), repeat=3)], device=device)
    n = torch.linalg.norm(d, dim=-1, keepdim=True)
    return torch.where(n > 0, d / n.clamp(min=1e-6), d)


class BusPilot:
    """Stateful, batched: :meth:`step` once per command tick, :meth:`reset` on episode resets."""

    def __init__(self, cfg: BusPilotCfg, face: Face, num_envs: int, num_defenders: int, num_attackers: int,
                 tick_s: float, device: str, sensor: TagSensorCfg | None = None, tags=None):
        """``tags``: (pos, normal, size) face-local, e.g. the env's ``room_tags``; None = ``box_room_tags``."""
        if len(cfg.post_y) != num_defenders:
            raise ValueError("post_y needs one entry per defender")
        self.cfg, self.face, self.n, self.nd, self.na, self.tick_s, self.device = (
            cfg, face, num_envs, num_defenders, num_attackers, tick_s, device)
        self.sensor = sensor or TagSensorCfg()
        self.tags = tags if tags is not None else box_room_tags(cfg, device)
        self.pts = face.grid(*cfg.face_grid, device)
        self.posts = torch.tensor([[cfg.post_depth, y, cfg.post_height] for y in cfg.post_y], device=device)
        self.dirs = _directions(device)
        maps = [m for m in itertools.product(range(num_defenders + 1), repeat=num_attackers)
                if len({d for d in m if d < num_defenders}) == sum(d < num_defenders for d in m)]
        self.perms = torch.tensor(maps, device=device)  # num_defenders = "none"
        self.debounce_ticks = max(1, round(cfg.debounce_s / tick_s))
        self.cand_yaw = torch.arange(cfg.yaw_candidates, device=device) * (2.0 * math.pi / cfg.yaw_candidates)
        y_min = -cfg.y_max if cfg.y_min is None else cfg.y_min
        self.lo = torch.tensor([cfg.x_min, y_min, cfg.z_min], device=device)
        self.hi = torch.tensor([cfg.x_max, cfg.y_max, cfg.z_max], device=device)

        n, a, d = num_envs, num_attackers, num_defenders
        self.att_state = torch.zeros(n, a, dtype=torch.long, device=device)
        self._pending = torch.zeros_like(self.att_state)
        self._pending_ticks = torch.zeros_like(self.att_state)
        self.perm_idx = torch.full((n,), -1, dtype=torch.long, device=device)
        self._perm_age = torch.zeros(n, device=device)
        self._prev_active = torch.zeros(n, a, dtype=torch.bool, device=device)
        self._prev_def_alive = torch.ones(n, d, dtype=torch.bool, device=device)
        self.role = torch.full((n, d), -1, dtype=torch.long, device=device)
        """Per defender: -1 hold, -2 spare guard, -3 downed, else the attacker it guards."""
        self.margin = torch.zeros(n, d, a, device=device)
        """Guarding margin of every defender against every attacker, at the last tick (estimates)."""
        self.target = self.posts.unsqueeze(0).expand(n, -1, -1).clone()
        self.strike = torch.zeros(n, d, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.att_state[env_ids] = OUTSIDE
        self._pending[env_ids] = OUTSIDE
        self._pending_ticks[env_ids] = 0
        self.perm_idx[env_ids] = -1
        self._perm_age[env_ids] = 0.0
        self._prev_active[env_ids] = False
        self._prev_def_alive[env_ids] = True

    # -- zones ---------------------------------------------------------------------------

    def _update_zones(self, att_pos: torch.Tensor, alive: torch.Tensor) -> None:
        c = self.cfg
        r = self.face.distance(att_pos)
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
        self.att_state = torch.where(alive, self.att_state, OUTSIDE)
        self._pending = torch.where(alive, self._pending, OUTSIDE)
        self._pending_ticks = torch.where(alive, self._pending_ticks, 0)

    # -- assignment ----------------------------------------------------------------------

    def _assign(self, margin: torch.Tensor, active: torch.Tensor, def_alive: torch.Tensor) -> None:
        """Bottleneck assignment on margin (N, D, A): maximise the worst guarded margin."""
        c = self.cfg
        a_idx = torch.arange(self.na, device=self.device)
        m = torch.where(def_alive.unsqueeze(-1), margin, torch.full_like(margin, -1e4))
        none = torch.full_like(m[:, :1], -c.unassigned_penalty)
        per = torch.cat([m, none], 1)[:, self.perms, a_idx]  # (N, P, A)
        act = active.unsqueeze(1)
        worst = torch.where(act, per, torch.full_like(per, float("inf"))).min(-1).values
        worst = torch.where(active.any(-1, keepdim=True), worst, torch.zeros_like(worst))
        score = worst + 1e-3 * (per * act.float()).sum(-1)  # ties: the better total
        best_score, best = score.max(dim=-1)
        cur = score.gather(1, self.perm_idx.clamp(min=0).unsqueeze(1)).squeeze(1)
        changed = (active != self._prev_active).any(-1) | (def_alive != self._prev_def_alive).any(-1)
        better = (best_score > cur + c.switch_margin) & (self._perm_age >= c.min_hold_s)
        switch = (self.perm_idx < 0) | changed | better
        self.perm_idx = torch.where(switch, best, self.perm_idx)
        self._perm_age = torch.where(switch, torch.zeros_like(self._perm_age), self._perm_age + self.tick_s)
        self._prev_active = active.clone()
        self._prev_def_alive = def_alive.clone()

    # -- tick ----------------------------------------------------------------------------

    def step(self, def_pos, def_vel, def_quat, def_alive, att_pos, att_vel, att_alive, has_tag=None):
        """One command tick. Positions env-local; attacker state is the tracker's estimate.

        Args:
            def_pos, def_vel: (N, D, 3).  def_quat: (N, D, 4) w, x, y, z.  def_alive: (N, D) bool.
            att_pos, att_vel: (N, A, 3) estimates.  att_alive: (N, A) bool (from telemetry).
            has_tag: (N, D) bool, whether each defender has a tag fix (None: assume yes).
        Returns:
            vel (N, D, 3) room frame [m/s], yaw_rate (N, D) CCW [rad/s].
        """
        c = self.cfg
        n, nd, dev = self.n, self.nd, self.device
        self._update_zones(att_pos, att_alive)
        active = att_alive & (self.att_state >= ALERT)

        self.margin = self.face.margin(def_pos.unsqueeze(2), att_pos.unsqueeze(1), self.pts)  # (N, D, A)
        self._assign(self.margin, active, def_alive)

        perm = self.perms[self.perm_idx]  # (N, A)
        role = torch.full((n, nd), -1, dtype=torch.long, device=dev)
        for a in range(self.na):
            hit = (torch.arange(nd, device=dev).unsqueeze(0) == perm[:, a : a + 1]) & active[:, a : a + 1]
            role = torch.where(hit, a, role)
        role = torch.where(def_alive, role, -3)
        role = torch.where((role == -1) & active.any(-1, keepdim=True), -2, role)
        self.role = role

        # candidates: where each defender can be lookahead_s from now, inside the allowed box
        step = c.max_speed * c.lookahead_s
        cands = (def_pos.unsqueeze(2) + step * self.dirs).clamp(self.lo, self.hi)  # (N, D, K, 3)
        mc = self.face.margin(cands.unsqueeze(3), att_pos[:, None, None], self.pts)  # (N, D, K, A)
        a_of = role.clamp(min=0)
        own = mc.gather(3, a_of.view(n, nd, 1, 1).expand(-1, -1, mc.shape[2], 1)).squeeze(3)
        worst = torch.where(active[:, None, None], mc, torch.full_like(mc, float("inf"))).min(-1).values
        guard = torch.where((role >= 0).unsqueeze(-1), own, worst).clamp(max=c.margin_cap)
        to_post = torch.linalg.norm(cands - self.posts[None, :, None], dim=-1)
        score = torch.where((role == -1).unsqueeze(-1), -to_post, guard - c.post_weight * to_post)

        # plan one defender after another, keeping clear of the spots already taken
        clear = 2.0 * c.sphere_radius + c.sphere_margin
        planned = def_pos.clone()
        for d in range(nd):
            others = torch.cat([planned[:, :d], planned[:, d + 1 :]], 1)
            o_alive = torch.cat([def_alive[:, :d], def_alive[:, d + 1 :]], 1)
            gap = torch.linalg.norm(cands[:, d, :, None] - others[:, None], dim=-1)  # (N, K, D-1)
            intrude = ((clear - gap).clamp(min=0.0) * o_alive[:, None].float()).sum(-1)
            pick = (score[:, d] - c.conflict_weight * intrude).argmax(-1)
            planned[:, d] = cands[:, d].gather(1, pick.view(n, 1, 1).expand(-1, 1, 3)).squeeze(1)
        target = planned

        # strike: an engaged attacker inside strike_radius of its guard is taken on directly
        tgt_att = att_pos.gather(1, a_of.unsqueeze(-1).expand(-1, -1, 3))
        tgt_state = self.att_state.gather(1, a_of)
        near = torch.linalg.norm(tgt_att - def_pos, dim=-1) < c.strike_radius
        self.strike = (role >= 0) & (tgt_state == ENGAGE) & near
        tgt_vel = att_vel.gather(1, a_of.unsqueeze(-1).expand(-1, -1, 3))
        t_hit, _ = intercept(def_pos, tgt_att, tgt_vel, c.max_speed)
        aim = tgt_att + tgt_vel * t_hit.clamp(max=c.strike_lead_s).unsqueeze(-1)
        aim = torch.maximum(aim, self.lo)  # not behind the face
        target = torch.where(self.strike.unsqueeze(-1), aim, target)
        self.target = target

        err = target - def_pos
        dist = torch.linalg.norm(err, dim=-1, keepdim=True).clamp(min=1e-6)
        # guards fly to their chosen candidate in lookahead_s: full speed for a full step, so
        # they move as fast as an attacker can; holding defenders settle on their posts
        guarding = (role >= 0) | (role == -2)
        vel = torch.where(guarding.unsqueeze(-1), err / c.lookahead_s, c.gain * err)
        vel = torch.where(self.strike.unsqueeze(-1), err / dist * c.max_speed, vel)
        vel = vel / (torch.linalg.norm(vel, dim=-1, keepdim=True) / c.max_speed).clamp(min=1.0)
        vel = self._spheres(def_pos, vel, def_vel, def_alive, att_pos, att_vel, att_alive)
        vel = vel * def_alive.unsqueeze(-1).float()
        return vel, self._heading(def_pos, def_quat, has_tag)

    def _spheres(self, pos, vel, def_vel, def_alive, att_pos, att_vel, att_alive) -> torch.Tensor:
        """Remove closing speed toward teammates and attackers inside their braking bands."""
        c = self.cfg
        n, nd = self.n, self.nd
        if not c.avoid_attackers:
            att_alive = torch.zeros_like(att_alive)  # live attackers are not obstacles
            att_pos = torch.full_like(att_pos, 1e3)
        obs = torch.cat([pos, att_pos], 1).unsqueeze(1)
        obs_vel = torch.cat([def_vel, att_vel], 1).unsqueeze(1)
        clear = torch.cat([torch.where(def_alive, 2.0 * c.sphere_radius, c.downed_clearance),
                           torch.where(att_alive, c.attacker_clearance, c.downed_clearance)], 1).view(n, 1, -1, 1)
        margin = torch.cat([torch.full((n, nd), c.sphere_margin, device=self.device),
                            torch.full((n, self.na), c.attacker_margin, device=self.device)], 1).view(n, 1, -1, 1)
        diff = pos.unsqueeze(2) - obs
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
        yaw = yaw_from_quat(quat)
        p = pos.unsqueeze(2).expand(-1, -1, k, -1).reshape(-1, 3)
        rot = yaw_rotation(self.cand_yaw).unsqueeze(0).expand(n * nd, -1, -1, -1).reshape(-1, 3, 3)
        score = tag_score(self.sensor, p, rot, self.tags).view(n, nd, k)
        best = score.max(-1, keepdim=True).values
        turn = wrap(self.cand_yaw.view(1, 1, k) - yaw.unsqueeze(-1))
        ok = (score >= c.yaw_keep_frac * best) & (score > 0)
        pick = turn.abs().masked_fill(~ok, float("inf")).argmin(-1, keepdim=True)
        err = turn.gather(-1, pick).squeeze(-1)
        err = torch.where(best.squeeze(-1) > 0, err, torch.full_like(err, math.pi))
        if has_tag is None:
            has_tag = torch.ones_like(yaw, dtype=torch.bool)
        cap = torch.where(has_tag, c.yaw_rate_tracking, c.yaw_rate_search)
        return torch.maximum(torch.minimum(c.yaw_gain * err, cap), -cap)
