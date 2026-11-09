"""The real room (``drone_deployment_infra/config/map.yaml``, the ground truth) in the face task's frame.

The face tasks keep their env-local frame: breach face on the plane x = 0, the box behind it at
x < 0, attackers coming in from +x. The room is placed in that frame turned half a turn about z:

  room (sim world, ``tags.room_map``)   X out of the tag-10 wall, Y along it, Z up from the floor
  face-local                           x = face_x - X,  y = face_center_y - Y,  z = Z

with face_x = room_depth - box_depth: the box stands against the far wall, opposite tag 10, and
its face looks back at tag 10. Defenders start in front of the face facing +x, i.e. at tag 10,
which they localize on; attackers start near the tag-10 wall and fly away from it.
"""

from __future__ import annotations

import torch

from action_space_kit.tags.room_map import load_room_tags


def face_x(cfg) -> float:
    """Room X of the breach face plane."""
    return cfg.room_depth - cfg.box_depth


def to_local(cfg, p: torch.Tensor) -> torch.Tensor:
    """Room (sim world) points (..., 3) -> face-local."""
    return torch.stack([face_x(cfg) - p[..., 0], cfg.face_center_y - p[..., 1], p[..., 2]], -1)


def room_tags(cfg, device: str):
    """(pos, normal, size) of every mapped tag with a known size, face-local."""
    tags = load_room_tags(tag10_height=cfg.tag10_height)
    ids = sorted(tags)
    pos = torch.tensor([tags[i].pos for i in ids], dtype=torch.float32, device=device)
    normal = torch.tensor([tags[i].normal for i in ids], dtype=torch.float32, device=device)
    size = torch.tensor([tags[i].size_m for i in ids], dtype=torch.float32, device=device)
    normal = torch.stack([-normal[:, 0], -normal[:, 1], normal[:, 2]], -1)  # the half turn, no shift
    return to_local(cfg, pos), normal, size, ids


def apply_room(cfg) -> None:
    """Derive the arena (the room's walls) and the start boxes, face-local, from the room fields."""
    fx, cy = face_x(cfg), cfg.face_center_y
    (y0, y1), m = cfg.room_y, cfg.room_start_margin
    cfg.arena_x = [fx - cfg.room_depth, fx]
    cfg.arena_y = [cy - y1, cy - y0]
    sx0, sx1 = cfg.attacker_start_room_x
    cfg.attacker_start_x = [fx - sx1, fx - sx0]
    sy0, sy1 = cfg.attacker_start_room_y
    cfg.attacker_start_y = [max(cy - sy1, cfg.arena_y[0] + m), min(cy - sy0, cfg.arena_y[1] - m)]
    # baiting attackers turn back before the tag-10 wall
    cfg.bait_retreat_x = min(cfg.bait_retreat_x, cfg.arena_x[1] - 1.0)


def pilot_y_range(cfg, wall_margin: float = 0.5) -> tuple[float, float]:
    """(y_min, y_max) face-local for the bus pilot: the side walls less ``wall_margin``."""
    return cfg.arena_y[0] + wall_margin, cfg.arena_y[1] - wall_margin
