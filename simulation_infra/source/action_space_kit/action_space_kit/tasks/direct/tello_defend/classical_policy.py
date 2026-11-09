"""The classical bus defender behind the shared defender policy's interface (59 inputs -> 4 outputs).

The deploy runner (``drone_deployment_infra/defender/run_defender.py``) builds, 13.5 times a second,
one 59-number observation per defender (``policy_runtime.build_obs``) and asks a policy for one
4-number command per defender in [-1, 1]: world vx, vy, vz in action units (1.0 = the runner's
``--max-speed``) and a yaw that the runner ignores (the nose follows the flight direction). Its
safety layer, stick conversion, +-40 cap and +-15 slew come after. ``ClassicalPolicy.act`` is a
drop-in for ``SharedPolicy.act`` with the same contract:

    policy = ClassicalPolicy.for_room(site=(4.2, 0.0, 1.5))   # where the runner's --site is
    act = policy.act(obs)                                      # (D, 59) -> (D, 4)

Inside, the layers match the network's stack: the bus's own assignment (dominance margins over
the face, debounced) decides who guards whom, and its guard / strike / hold motion decides how to
fly there. The planner fields of the observation (target, base command, slot) come from the
network's cylinder planner and are not used: the bus defends the breach FACE, not a cylinder.

From the observation alone each defender's view gives the whole picture: its own fix (relative to
the site), every attacker and teammate relative to it, their velocities and alive flags. A
defender that is out shows (clipped) 12 m away in every axis. The views of one tick agree (the
runner builds them from one set of estimates), so the planner runs once per tick on them.

Frames: world = the runner's (X out of the tag-10 wall, Y along it, Z up); the bus works face-local
(``room_layout``: x = face_x - X, y = face_center_y - Y, z = Z), the face plane at x = 0.
Only torch is needed (no Isaac): ``drone_deployment_infra/core/classical_defender.py`` loads it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from .bus_pilot import BusPilot, BusPilotCfg
from .face_geometry import Face

OBS_CLIP_POS = 12.0  # policy_runtime.OBS_CLIP_POS: a teammate that is out reads as this far away
OUT_READING = 11.5  # [m] |relative position| at or past this in every axis = out


def obs_layout(num_defenders: int, num_attackers: int) -> dict[str, slice]:
    """Slices of one defender's observation (policy_runtime.build_obs)."""
    sizes = [("own_pos", 3), ("own_vel", 3), ("target", 3), ("base", 3), ("slot", 2 * num_attackers + 2),
             ("attackers", 12 * num_attackers), ("mates", 6 * (num_defenders - 1)), ("applied", 4), ("age", 1)]
    out, k = {}, 0
    for name, n in sizes:
        out[name] = slice(k, k + n)
        k += n
    out["dim"] = slice(0, k)
    return out


@dataclass
class FaceFrame:
    """The breach face in the world frame: the plane X = face_x, attackers at X < face_x."""

    face_x: float = 4.2  # room_depth 5.2 - box_depth 1.0 (TelloDefendFaceRoomEnvCfg)
    center_y: float = 0.0  # directly opposite tag 10
    width: float = 3.0
    height: float = 2.0
    bottom: float = 0.5

    def to_local(self, p: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.face_x - p[..., 0], self.center_y - p[..., 1], p[..., 2]], -1)

    @staticmethod
    def vel_to_local(v: torch.Tensor) -> torch.Tensor:
        return torch.stack([-v[..., 0], -v[..., 1], v[..., 2]], -1)

    vel_to_world = vel_to_local  # the half turn is its own inverse


class ClassicalPolicy:
    """Bus defender with the shared policy's ``act`` contract. Stateful: one call per tick, ``reset`` between rounds."""

    def __init__(self, site, frame: FaceFrame, num_defenders: int = 3, num_attackers: int = 2, hz: float = 13.5,
                 action_speed: float = 1.0, pilot_cfg: BusPilotCfg | None = None, device: str = "cpu"):
        """``site``: the world point the observation's own position is relative to (the runner's --site).
        ``action_speed``: m/s per action unit (the runner's --max-speed). ``pilot_cfg.max_speed`` is the
        top speed the bus plans with: what the drones really fly at under the runner's stick limits."""
        self.site = torch.as_tensor(site, dtype=torch.float32, device=device).reshape(3)
        self.frame, self.nd, self.na, self.device = frame, num_defenders, num_attackers, device
        self.action_speed = action_speed
        self.layout = obs_layout(num_defenders, num_attackers)
        self.cfg = pilot_cfg or BusPilotCfg()
        face = Face(frame.width, frame.height, frame.bottom)
        self.pilot = BusPilot(self.cfg, face, 1, num_defenders, num_attackers, 1.0 / hz, device)
        self.identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(1, num_defenders, 4)

    @classmethod
    def for_room(cls, site, room_y=(-2.0, 3.2), wall_margin: float = 0.5, frame: FaceFrame | None = None, **kwargs):
        """The final arena: side walls ``room_y`` (world Y) bound the guards, less ``wall_margin``."""
        frame = frame or FaceFrame()
        cfg = kwargs.pop("pilot_cfg", None) or BusPilotCfg()
        cfg = dataclasses.replace(cfg, y_min=frame.center_y - room_y[1] + wall_margin,
                                  y_max=frame.center_y - room_y[0] - wall_margin)
        return cls(site, frame, pilot_cfg=cfg, **kwargs)

    def reset(self) -> None:
        self.pilot.reset(torch.zeros(1, dtype=torch.long, device=self.device))

    def decode(self, obs: torch.Tensor):
        """(D, 59) -> world state: def_pos/vel (D, 3), def_alive (D,), att_pos/vel (A, 3), att_alive (A,)."""
        L, D, A = self.layout, self.nd, self.na
        obs = obs.to(self.device, torch.float32)
        own = obs[:, L["own_pos"]]
        def_alive = ~(own.abs() >= OUT_READING).all(-1)
        def_pos = self.site + own
        def_vel = obs[:, L["own_vel"]]
        # attackers from the first live defender's view (all views of a tick share the estimates)
        i = int(def_alive.nonzero()[0]) if def_alive.any() else 0
        att = obs[i, L["attackers"]].view(A, 12)
        att_alive = att[:, 10] > 0.5
        att_pos = def_pos[i] + att[:, 0:3]
        att_vel = att[:, 3:6]
        return def_pos, def_vel, def_alive, att_pos, att_vel, att_alive

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        def_pos, def_vel, def_alive, att_pos, att_vel, att_alive = self.decode(obs)
        f = self.frame
        vel, _ = self.pilot.step(f.to_local(def_pos)[None], f.vel_to_local(def_vel)[None], self.identity,
                                 def_alive[None], f.to_local(att_pos)[None], f.vel_to_local(att_vel)[None],
                                 att_alive[None])
        v = f.vel_to_world(vel[0]) / self.action_speed
        out = torch.cat([v, torch.zeros_like(v[:, :1])], -1).clamp(-1.0, 1.0)
        return torch.where(def_alive[:, None], out, torch.zeros_like(out))


def build_obs(def_pos, def_vel, att_pos, att_vel, att_alive, site, applied, own_age_s, def_alive=None,
              target=None, base=None, slot=None, tau=None, zone=None, role=None):
    """The runner's observation (policy_runtime.build_obs), (D, 59), for driving either policy in the sim.

    The planner fields (target, base, slot, tau, zone, role) default to zeros / "post": the classical
    policy does not read them. Shapes as policy_runtime: def (D, 3), att (A, 3), site (3,).
    """
    D, A = def_pos.shape[0], att_pos.shape[0]
    cp, cv = OBS_CLIP_POS, 3.0
    pos, vel = def_pos.clone().float(), def_vel.clone().float()
    if def_alive is not None:
        pos = pos + (~def_alive).float()[:, None] * 1.0e3
        vel = torch.where(def_alive[:, None], vel, torch.zeros_like(vel))
    z3 = torch.zeros(D, 3, device=pos.device)
    target = pos if target is None else target
    base = z3 if base is None else base
    slot = torch.full((D,), -1, dtype=torch.long, device=pos.device) if slot is None else slot
    role = torch.full((D,), -1, dtype=torch.long, device=pos.device) if role is None else role
    alive = att_alive.float()
    tau = torch.zeros(A, device=pos.device) if tau is None else tau
    zone = torch.zeros(A, dtype=torch.long, device=pos.device) if zone is None else zone
    zone_1h = torch.nn.functional.one_hot(zone.clamp(min=0), 3).float() * alive[:, None]
    slot_1h = torch.nn.functional.one_hot(slot + 1, 2 * A + 2).float()
    ids = torch.arange(A, device=pos.device)
    rows = []
    for i in range(D):
        p = pos[i]
        att_rel = (att_pos - p).clamp(-cp, cp) * alive[:, None]
        a_vel = att_vel.clamp(-cv, cv) * alive[:, None]
        mine = (role[i] == ids).float()
        per_att = torch.cat([att_rel, a_vel, (tau * alive)[:, None], zone_1h, alive[:, None], mine[:, None]], -1).reshape(-1)
        mate = torch.cat([torch.cat([(pos[k] - p).clamp(-cp, cp), vel[k].clamp(-cv, cv)]) for k in range(D) if k != i])
        rows.append(torch.cat([(p - site).clamp(-cp, cp), vel[i].clamp(-cv, cv), (target[i] - p).clamp(-cp, cp),
                               base[i, :3], slot_1h[i], per_att, mate, applied[i].float(), own_age_s[i : i + 1].float()]))
    return torch.nan_to_num(torch.stack(rows), nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)


def safety(act, def_pos, def_alive=None, floor_z=0.0, action_speed=1.0):
    """The runner's safety layer (policy_runtime.safety): yaw off, teammate repulsion within 0.5 m,
    no descending below 0.45 m, clamp. (D, 4) action units."""
    cmd = act.clone().float()
    cmd[:, 3] = 0.0
    diff = def_pos[:, None] - def_pos[None]
    dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=1e-3)
    push = (0.5 - dist).clamp(min=0.0) * 2.0 * diff / dist
    if def_alive is not None:
        push = push * def_alive[None, :, None].float()
    cmd[:, :3] = cmd[:, :3] + push.sum(1) / action_speed
    low = (def_pos[:, 2] - floor_z) < 0.45
    cmd[:, 2] = torch.where(low, cmd[:, 2].clamp(min=0.3), cmd[:, 2])
    cmd = cmd.clamp(-1.0, 1.0)
    if def_alive is not None:
        cmd = torch.where(def_alive[:, None], cmd, torch.zeros_like(cmd))
    return cmd
