#!/usr/bin/env python3
"""Our defenders, the whole pipeline in one file: localize, target, fly.

    python swarm_defend.py --only 6,7,9                              # dry-run: connect, localize, plan, print
    python swarm_defend.py --only 6,7,9 --fake-feed                  # dry-run against scripted attackers
    python swarm_defend.py --only 6,7,9 --fly --headless             # fly
    python swarm_defend.py --bench 200                               # no drones: closed loop on a Tello rc model
    python swarm_defend.py --selftest

Pipeline, 13.5 times a second (one rc packet per drone per tick, as the sim):

  localization  swarm_formation.py's link, video and AprilTag fixes against config/map.yaml (any
                mapped tag; tag 10 preferred), height from the downward range sensor. Takeoff,
                CLIMB to --start-height, SEARCH (look, turn 20 deg, ...) until a fix, then DEFEND.
  attackers     the organizers' UDP feed (pose + alive), through a constant-velocity Kalman filter
                per axis (the sim's telemetry model) carried forward to now.
  targeting     the sim's classical "bus" (tello_defend/bus_pilot.py, baked in below by
                build_defend.py, so this is the code the sim scored): Schmitt-trigger zones by
                distance to the breach face, bottleneck (Hungarian-style) assignment of one
                guard per active attacker on the equal-speed guarding margin, guard points that
                keep that margin, strike inside the strike radius, safety spheres. It picks every
                drone's waypoint each tick.
  motion        holding drones fly to their post with swarm_formation's waypoint controller
                (spacing, rc kick, nose on tag 10); guards and strikers fly the bus's velocity
                (full speed), converted to sticks with the fitted rc gains; the nose stays on
                tag 10 so the fixes stay fresh. No fix for 0.4 s: no translation, turn back to
                the tags; none for --lost-land s: that drone lands and is out (the assignment
                then covers both attackers with the others). A defender that ends up on the
                floor (a collision) is out too.

Frames: world = X out of the tag-10 wall, Y along it (right facing the wall), Z up from the floor;
the bus works face-local (x = face_x - X, y = face_y - Y, z = Z: the face at x = 0, attackers at x > 0).

Attacker feed (UDP JSON, one per datagram), p in --feed-frame (map: tag-10 frame x right, y up,
z out of the wall; world: as above), age (s, optional), alive (optional):
    {"attackers": {"<name>": {"p": [x, y, z], "age": 0.05, "alive": true}, ...}}
  or "attackers": [{"name": ..., "p": ...}, ...]. Names map to A0, A1 in order first seen (or --attackers).

The umbrella (swarm_policies.py) can launch it as an external group like formation: it prints a
"] PASS" line per drone, asks "Type FLY:", and lands all on SIGINT. DEFAULT IS DRY-RUN; --fly to fly.
Libs: numpy opencv-python pupil-apriltags av pyyaml torch.
"""
import argparse
import json
import math
import os
import socket
import sys
import threading
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ======================= SETTINGS =======================
PERIOD_S = 1.0 / 13.5         # one rc packet per drone per tick
TAG10_HEIGHT_M = 1.22         # tag 10 centre above the floor (the map has no floor)
START_HEIGHT_CM = 130
CLIMB_TIMEOUT_S = 6.0
SEARCH_YAW_RC = 25
SEARCH_STEP_DEG = 20
SEARCH_LOOK_S = 0.8
SEARCH_MAX_DEG = 380
LOST_TRANSLATE_S = 0.4        # fix older than this: no translation (there is no odometry)
RECOVER_YAW_RC = 15
MAX_YAW_RC = 20               # faster turns blur frames and cost fixes
YAW_DEADBAND_DEG = 4.0
HOLD_RC = 30                  # waypoint-controller cap while holding a post
MIN_RC = 8                    # a Tello ignores smaller sticks (deadband ~5) ...
ZERO_RC = 3                   # ... and below this we send 0 (no jitter at +-MIN_RC)
GAIN_RIGHT, GAIN_FWD, GAIN_UP = 0.008, 0.00731, 0.006   # m/s per rc unit, tello_motion_fit.yaml
VEL_WINDOW_S = 0.3
FEED_STALE_S = 1.0            # an attacker missing from the feed this long: held, not extrapolated
# engagement: CHASE. Every attacker anywhere in the room is assigned a defender, which goes
# straight for it (head-on intercept) from the first tick; no waiting for it to come close.
# (The sim's bus defaults were alert 4.0 / engage 1.5 / strike 1.3 m with a 3 m leash: guards
# held back until the attackers reached the face, which looked like a stall.)
ALERT_RADIUS_M = 50.0
ENGAGE_RADIUS_M = 50.0
STRIKE_RADIUS_M = 50.0
WALL_GAP_M = 0.2              # the leash: chase out to 0.2 m from the tag-10 wall
# telemetry Kalman filter (bus_pilot.TelemetryCfg)
FEED_NOISE_M = 0.03
FEED_ACCEL = 4.0
MAX_SPEED = 0.89              # our top speed, m/s (the highest the drones reached on the rc model)
ATT_MAX_SPEED = 1.5           # the rules' ceiling: attacker velocity estimates are capped here


# ======================= THE SIM'S DEFENDER (baked) =======================
# >>> BAKED SIM: generated by build_defend.py from simulation_infra; edit the sim files, not this block
BAKED = {}
BAKED['_baked_sensing'] = r'''"""What the drone's camera can localize from: which mapped tags are detectable, and the fix.

On the real Tello the only position source is the room's AprilTags (``core/positional.py``):
a frame with any mapped tag in it gives the camera position in the tag-10 frame, averaged
over every visible tag. Between sightings the flight engine dead-reckons on the state
packet. This module models the sighting half, batched in torch:

  detectable  tag centre in front of the camera, the whole tag inside the image, the
              printed face turned toward the camera (``max_view_angle``), its apparent
              edge at least ``min_side_px`` pixels; then it decodes at ``decode_rate``
              (``dropout``, falling with the view angle past ``view_full_angle``) unless
              motion blur loses the whole frame (mostly turning, ``blur_*``)
  fix         the true camera position plus PnP-style noise per tag (depth error grows
              as range^2 / (fx * size), lateral error as range / fx), averaged over the
              detectable tags like ``Positioner.locate``

Camera frame is OpenCV's: x right, y down, z forward. Only torch is imported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TagSensorCfg:
    # intrinsics: defaults are config/camera/tello_3.yaml (960x720, fx ~ fy ~ 914.5 px)
    fx: float = 914.5
    fy: float = 914.4
    width: int = 960
    height: int = 720
    camera_offset: tuple[float, float, float] = (0.05, 0.0, 0.0)
    """[m] camera centre in the body frame (x forward, y left, z up): the Tello's nose."""
    camera_pitch: float = 0.0
    """[rad] camera tilt below the horizon. The Tello's camera is fixed, looking ahead."""

    min_side_px: float = 15.0
    """Smallest apparent tag edge that still decodes (tags.yaml gate is 12 px; blur costs more)."""
    max_view_angle: float = 62.0 * math.pi / 180.0
    """[rad] between the tag normal and the line to the camera: past this a tag never decodes."""
    view_full_angle: float = 10.0 * math.pi / 180.0
    """[rad] up to this view angle a tag decodes at the full rate (1 - dropout); from here to
    max_view_angle the rate falls linearly in cos(view angle) to zero. Fitted to 3,700 in-frame
    tag sightings in flights 030-034 and 040-043: detected 89% at 0-20 deg, 77% at 20-35,
    41% at 35-50, 16% at 50-60, 6% at 60-70 (the old flat 90% up to 70 deg overcounted)."""
    max_range: float = 8.0
    """[m] beyond this no tag is used (also caps how far the small tags are trusted)."""
    dropout: float = 0.1
    """Per tag, per frame probability that a detectable tag is missed anyway."""
    # Motion blur loses whole frames. Flights 030-034 (IMU yaw rate vs fresh fixes, tags in
    # view): turning dominates (fix on 95% of frames below 0.2 rad/s, 84% at 0.2-0.5, 53% at
    # 0.5-1, 56% above 1 rad/s); straight flight barely matters (89-100% up to > 2 m/s).
    blur_free_speed: float = 0.5
    """[m/s] below this linear speed frames are sharp."""
    blur_loss_per_mps: float = 0.05
    """Whole-frame loss probability per m/s of linear speed above blur_free_speed."""
    blur_free_rate: float = 0.2
    """[rad/s] below this yaw rate frames are sharp."""
    blur_loss_per_radps: float = 1.0
    """Whole-frame loss probability per rad/s of yaw rate above blur_free_rate."""
    blur_rot_max_loss: float = 0.45
    """Cap on the turning loss (the real loss plateaus at ~45% from 0.5 rad/s up)."""
    blur_max_loss: float = 0.5
    """Cap on the combined loss 1 - (1 - linear)(1 - turning)."""
    pixel_noise: float = 1.0
    """[px] corner noise that drives the PnP position error."""
    video_delay_steps: int = 4
    """Policy steps between a frame being captured and its fix reaching the controller
    (Tello H.264 stream + decode + detect: ~0.2 s at 20 Hz)."""


def camera_axes(body_rot: torch.Tensor, pitch: float) -> torch.Tensor:
    """(N, 3, 3) OpenCV camera axes in the world (columns x right, y down, z forward).

    ``body_rot`` is (N, 3, 3) body-to-world with body x forward, y left, z up.
    """
    c, s = math.cos(pitch), math.sin(pitch)
    # in the body frame: forward tilted down by pitch, right = -y, down completes the triad
    fwd = torch.tensor([c, 0.0, -s], device=body_rot.device)
    right = torch.tensor([0.0, -1.0, 0.0], device=body_rot.device)
    down = torch.linalg.cross(fwd, right)
    cam_in_body = torch.stack([right, down, fwd], dim=-1)  # (3, 3) columns
    return body_rot @ cam_in_body


def detectable(cfg: TagSensorCfg, cam_pos: torch.Tensor, cam_rot: torch.Tensor, tag_pos: torch.Tensor,
               tag_normal: torch.Tensor, tag_size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Which tags the camera can decode, ignoring dropout.

    Args:
        cam_pos: (N, 3) world.  cam_rot: (N, 3, 3) from :func:`camera_axes`.
        tag_pos, tag_normal: (T, 3) world.  tag_size: (T,) black-square edge.
    Returns:
        mask (N, T) bool, range (N, T) metres.
    """
    d = tag_pos.unsqueeze(0) - cam_pos.unsqueeze(1)  # (N, T, 3) camera -> tag, world
    rng = torch.linalg.norm(d, dim=-1).clamp(min=1e-6)
    p = torch.einsum("nij,nti->ntj", cam_rot, d)  # in the camera frame
    z = p[..., 2]
    in_front = z > 0.1
    zc = z.clamp(min=0.1)
    # all four black-square corners must be inside the image (the decoder needs them)
    half = 0.5 * tag_size.unsqueeze(0)
    u = (p[..., 0].abs() + half) / zc * cfg.fx
    v = (p[..., 1].abs() + half) / zc * cfg.fy
    in_image = (u < 0.5 * cfg.width) & (v < 0.5 * cfg.height)
    # face toward the camera, within the decode angle
    cos_view = -(d * tag_normal.unsqueeze(0)).sum(-1) / rng
    facing = cos_view > math.cos(cfg.max_view_angle)
    # foreshortened edge in pixels: the short side shrinks with cos(view)
    side_px = cfg.fx * tag_size.unsqueeze(0) * cos_view.clamp(min=0.0).sqrt() / rng
    big_enough = side_px >= cfg.min_side_px
    mask = in_front & in_image & facing & big_enough & (rng <= cfg.max_range)
    return mask, rng


def blur_loss(cfg: TagSensorCfg, speed: torch.Tensor, yaw_rate: torch.Tensor) -> torch.Tensor:
    """Probability that motion blur loses a whole frame, from linear speed [m/s] and |yaw rate| [rad/s]."""
    lin = (cfg.blur_loss_per_mps * (speed - cfg.blur_free_speed)).clamp(0.0, 1.0)
    rot = (cfg.blur_loss_per_radps * (yaw_rate.abs() - cfg.blur_free_rate)).clamp(0.0, cfg.blur_rot_max_loss)
    return (1.0 - (1.0 - lin) * (1.0 - rot)).clamp(max=cfg.blur_max_loss)


def decode_rate(cfg: TagSensorCfg, cos_view: torch.Tensor) -> torch.Tensor:
    """Per-frame probability that a detectable tag decodes: 1 - dropout up to view_full_angle,
    then linear in cos(view angle) down to zero at max_view_angle."""
    c0, c1 = math.cos(cfg.view_full_angle), math.cos(cfg.max_view_angle)
    return (1.0 - cfg.dropout) * ((cos_view - c1) / (c0 - c1)).clamp(0.0, 1.0)


def sighting(cfg: TagSensorCfg, cam_pos: torch.Tensor, cam_rot: torch.Tensor, tag_pos: torch.Tensor,
             tag_normal: torch.Tensor, tag_size: torch.Tensor, noise: bool = True,
             speed: torch.Tensor | None = None, yaw_rate: torch.Tensor | None = None,
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One camera frame: (fix (N, 3) world, any_tag (N,) bool, mask (N, T) bool).

    ``fix`` is only meaningful where ``any_tag``; elsewhere it is the true position.
    With ``noise`` each detectable tag decodes at :func:`decode_rate` (dropout and view angle),
    and ``speed`` / ``yaw_rate`` (N,) blur away whole frames (:func:`blur_loss`).
    """
    mask, rng = detectable(cfg, cam_pos, cam_rot, tag_pos, tag_normal, tag_size)
    if noise:
        d = tag_pos.unsqueeze(0) - cam_pos.unsqueeze(1)
        cos_view = -(d * tag_normal.unsqueeze(0)).sum(-1) / rng
        mask = mask & (torch.rand_like(rng) < decode_rate(cfg, cos_view))
    if noise and speed is not None and yaw_rate is not None:
        p_blur = blur_loss(cfg, speed, yaw_rate)
        mask = mask & (torch.rand_like(p_blur) >= p_blur).unsqueeze(-1)
    per_tag = cam_pos.unsqueeze(1).expand(-1, tag_pos.shape[0], -1)
    if noise:
        los = (cam_pos.unsqueeze(1) - tag_pos.unsqueeze(0)) / rng.unsqueeze(-1)
        sigma_depth = cfg.pixel_noise * rng**2 / (cfg.fx * tag_size.unsqueeze(0))
        sigma_lat = cfg.pixel_noise * rng / cfg.fx
        per_tag = (
            per_tag
            + los * (torch.randn_like(rng) * sigma_depth).unsqueeze(-1)
            + torch.randn_like(per_tag) * sigma_lat.unsqueeze(-1)
        )
    w = mask.float().unsqueeze(-1)
    count = w.sum(1)
    fix = torch.where(count > 0, (per_tag * w).sum(1) / count.clamp(min=1.0), cam_pos)
    return fix, mask.any(dim=1), mask
'''
BAKED['_baked_zone_pilot'] = r'''"""A classical 3-vs-2 defender: zones with hysteresis, assignment, tag-keeping heading, safety spheres.

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

from _baked_sensing import TagSensorCfg, camera_axes, decode_rate, detectable

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
'''
BAKED['_baked_face_geometry'] = r'''"""The breach face: one face of a cuboid that attackers must fly through. Pure torch.

Env-local frame: the face is the plane x = 0, spanning y in [-width/2, width/2] and z in
[bottom, bottom + height]; the box is behind it (x < 0) and attackers come from +x.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Face:
    width: float = 3.0
    height: float = 2.0
    bottom: float = 0.5

    def distance(self, pos: torch.Tensor) -> torch.Tensor:
        """Distance from points (..., 3) to the face rectangle, (...)."""
        dx = pos[..., 0].clamp(min=0.0)
        dy = (pos[..., 1].abs() - 0.5 * self.width).clamp(min=0.0)
        dz = torch.maximum(self.bottom - pos[..., 2], pos[..., 2] - (self.bottom + self.height)).clamp(min=0.0)
        return torch.sqrt(dx * dx + dy * dy + dz * dz)

    def inside(self, pos: torch.Tensor) -> torch.Tensor:
        """Whether points (..., 3) project inside the face rectangle (y, z), (...) bool."""
        return (pos[..., 1].abs() <= 0.5 * self.width) & (pos[..., 2] >= self.bottom) & (
            pos[..., 2] <= self.bottom + self.height)

    def grid(self, ny: int, nz: int, device) -> torch.Tensor:
        """(ny * nz, 3) points covering the face, edges and corners included."""
        y = torch.linspace(-0.5 * self.width, 0.5 * self.width, ny, device=device)
        z = torch.linspace(self.bottom, self.bottom + self.height, nz, device=device)
        yy, zz = torch.meshgrid(y, z, indexing="ij")
        return torch.stack([torch.zeros_like(yy), yy, zz], -1).reshape(-1, 3)

    def margin(self, d_pos: torch.Tensor, a_pos: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """Equal-speed guarding margin: min over face points T of |A - T| - |D - T|.

        Positive: the defender reaches every point of the face before the attacker (Isaacs'
        target guarding with equal speeds), so that attacker cannot breach past it. Shapes
        broadcast: d_pos (..., 3) and a_pos (..., 3) -> (...).
        """
        da = torch.linalg.norm(a_pos.unsqueeze(-2) - pts, dim=-1)
        dd = torch.linalg.norm(d_pos.unsqueeze(-2) - pts, dim=-1)
        return (da - dd).min(dim=-1).values
'''
BAKED['_baked_bus_pilot'] = r'''"""'Park the bus': a classical defender for the breach face that does not predict the attackers.

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

from _baked_sensing import TagSensorCfg

from _baked_face_geometry import Face
from _baked_zone_pilot import ALERT, ENGAGE, OUTSIDE, intercept, rc_command, tag_score, wrap, yaw_from_quat, yaw_rotation  # noqa: F401


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
'''
# <<< BAKED SIM


_OUT_LOCK = threading.Lock()


def log_line(text):
    """One whole line per write: the umbrella counts "] PASS" per line, so parallel preflight
    threads must not merge their lines."""
    with _OUT_LOCK:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()


def load_bus():
    """The baked sim modules as real modules; returns bus_pilot."""
    for name, src in BAKED.items():
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        sys.modules[name] = mod                    # dataclasses look their module up here
        exec(compile(src, name, "exec"), mod.__dict__)
    return sys.modules["_baked_bus_pilot"]


# ======================= FRAMES =======================
def world_from_internal(p):
    """swarm_formation's tag-10 frame (x right, y DOWN, z INTO the wall) -> world."""
    return np.array([-p[2], p[0], -p[1] + TAG10_HEIGHT_M])


def internal_from_world(w):
    return np.array([w[1], TAG10_HEIGHT_M - w[2], -w[0]])


def world_from_map(p):
    """map.yaml / tag-10 frame (x right, y up, z out of the wall) -> world."""
    return np.array([p[2], p[0], p[1] + TAG10_HEIGHT_M])


def body_axes(psi_deg):
    """World forward and right of a drone with swarm_formation yaw psi (0 = facing tag 10, + = clockwise)."""
    s, c = math.sin(math.radians(psi_deg)), math.cos(math.radians(psi_deg))
    return np.array([-c, s, 0.0]), np.array([s, c, 0.0])


def face_tag_yaw(p_int):
    return math.degrees(math.atan2(-p_int[0], -p_int[2]))


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def kick(v):
    v = int(round(v))
    return 0 if abs(v) < ZERO_RC else (max(v, MIN_RC) if v > 0 else min(v, -MIN_RC))


def yaw_to_tag10(p_int, psi):
    err = wrap_deg(face_tag_yaw(p_int) - psi)
    return 0 if abs(err) < YAW_DEADBAND_DEG else kick(np.clip(err, -MAX_YAW_RC, MAX_YAW_RC))


def velocity_rc(v_world, psi, cap):
    """World velocity (m/s) -> rc a (right), b (forward), c (up) with the fitted gains, capped."""
    fwd, right = body_axes(psi)
    a, b, c = (v_world @ right) / GAIN_RIGHT, (v_world @ fwd) / GAIN_FWD, v_world[2] / GAIN_UP
    n = math.hypot(a, b)
    if n > cap:
        a, b = a * cap / n, b * cap / n
    return kick(a), kick(b), kick(np.clip(c, -cap, cap))


# ======================= ATTACKER FEED =======================
class FeedTracker:
    """Constant-velocity Kalman filter per axis on timestamped positions (the sim's telemetry filter)."""

    def __init__(self, n):
        self.n = n
        self.p, self.v = np.zeros((n, 3)), np.zeros((n, 3))
        self.P = np.zeros((n, 3, 2, 2))
        self.t = np.full(n, -1.0)              # time of the last measurement (-1: none yet)
        self.seen = np.zeros(n)                # wall time it arrived
        self.alive = np.ones(n, bool)

    def update(self, j, t_meas, z, now):
        r, q = FEED_NOISE_M ** 2, FEED_ACCEL ** 2
        if self.t[j] < 0:
            self.p[j], self.v[j], self.t[j], self.seen[j] = z, 0.0, t_meas, now
            self.P[j] = np.array([[r, 0.0], [0.0, 1.0]])
            return
        dt = t_meas - self.t[j]
        if dt <= 1e-3:
            return
        P = self.P[j]                          # (3, 2, 2)
        p = self.p[j] + self.v[j] * dt
        P00 = P[:, 0, 0] + dt * (P[:, 1, 0] + P[:, 0, 1]) + dt * dt * P[:, 1, 1] + q * dt ** 4 / 4
        P01 = P[:, 0, 1] + dt * P[:, 1, 1] + q * dt ** 3 / 2
        P11 = P[:, 1, 1] + q * dt ** 2
        S = P00 + r
        k0, k1 = P00 / S, P01 / S
        y = z - p
        self.p[j], v = p + k0 * y, self.v[j] + k1 * y
        speed = np.linalg.norm(v)
        self.v[j] = v * min(1.0, ATT_MAX_SPEED / max(speed, 1e-6))
        P[:, 0, 0], P[:, 0, 1] = (1 - k0) * P00, (1 - k0) * P01
        P[:, 1, 0], P[:, 1, 1] = P[:, 0, 1], P11 - k1 * P01
        self.t[j], self.seen[j] = t_meas, now

    def state(self, now):
        """(pos (A,3), vel (A,3), alive (A,), known (A,)) carried forward to now."""
        known = self.t >= 0
        fresh = (now - self.seen) < FEED_STALE_S
        vel = np.where((known & fresh)[:, None], self.v, 0.0)
        ahead = np.clip(now - self.t, 0.0, FEED_STALE_S)[:, None]
        return self.p + vel * ahead, vel, self.alive & known, known


class AttackerFeed:
    """The organizers' UDP feed, drained every tick into the tracker."""

    def __init__(self, port, frame, names, n):
        self.frame, self.n, self.names = frame, n, list(names)
        self.tracker = FeedTracker(n)
        self.datagrams = 0
        self.bad = 0                                 # packets we couldn't use (printed: the first 5)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)

    def slot(self, name):
        if name not in self.names:
            if len(self.names) >= self.n:
                if name not in getattr(self, "ignored", set()):
                    self.ignored = getattr(self, "ignored", set()) | {name}
                    log_line(f"[feed] attacker {name!r} IGNORED: already tracking {self.names} (--num-attackers {self.n})")
                return None
            self.names.append(name)
            log_line(f"[feed] attacker {name!r} -> A{len(self.names) - 1}")
        return self.names.index(name)

    def drain(self, now):
        while True:
            try:
                data, addr = self.sock.recvfrom(65536)
            except (BlockingIOError, OSError):
                return
            if self.datagrams == 0 and self.bad == 0:
                log_line(f"[feed] FIRST PACKET from {addr[0]}:{addr[1]} ({len(data)} B): {data[:300]!r}")
            try:
                msg = json.loads(data)
            except ValueError:
                self.complain(f"not JSON from {addr[0]}: {data[:120]!r}")
                continue
            self.datagrams += 1
            items = msg.get("attackers", {}) if isinstance(msg, dict) else {}
            if not items:
                self.complain(f"no 'attackers' in the packet: {data[:160]!r}")
            if isinstance(items, list):
                items = {str(it.get("name") or it.get("id")): it for it in items if isinstance(it, dict)}
            for name, it in items.items():
                j = self.slot(str(name))
                if j is None:
                    continue
                if not isinstance(it, dict) or "p" not in it:
                    self.complain(f"attacker {name!r} has no 'p' [x, y, z]: {str(it)[:120]}")
                    continue
                p = np.asarray(it["p"], float)
                self.tracker.update(j, now - float(it.get("age", 0.0)),
                                    world_from_map(p) if self.frame == "map" else p, now)
                if it.get("alive") is False and self.tracker.alive[j]:
                    self.tracker.alive[j] = False
                    log_line(f"[feed] {name} (A{j}) is DOWN (alive=false in the feed)")

    def complain(self, msg):
        self.bad += 1
        if self.bad <= 5:
            log_line(f"[feed] BAD PACKET #{self.bad}: {msg}" + ("  (no more of these printed)" if self.bad == 5 else ""))

    def state(self, now):
        return self.tracker.state(now)


def fake_feed(port, frame, face, speed, stop, seed=0):
    """Scripted attackers on the feed (for dry runs): launch beside tag 10, fly at the face, repeat."""
    rng = np.random.default_rng(seed)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while not stop.is_set():
        start = np.array([[rng.uniform(0.3, 0.6), rng.uniform(-1.5, 1.5), rng.uniform(1.0, 2.0)] for _ in range(2)])
        goal = np.array([[face["x"], face["y"] + rng.uniform(-1.2, 1.2), rng.uniform(0.8, 2.2)] for _ in range(2)])
        t0 = time.time()
        dur = np.linalg.norm(goal - start, axis=1) / speed
        while not stop.is_set() and time.time() - t0 < dur.max() + 2.0:
            s = np.clip((time.time() - t0) / dur, 0.0, 1.0)[:, None]
            pos = start + s * (goal - start)
            out = {}
            for j in range(2):
                p = pos[j] if frame == "world" else np.array([pos[j][1], pos[j][2] - TAG10_HEIGHT_M, pos[j][0]])
                out[f"FAKE-{j}"] = {"p": [round(float(x), 3) for x in p], "age": 0.0}
            tx.sendto(json.dumps({"attackers": out}).encode(), ("127.0.0.1", port))
            time.sleep(0.05)


# ======================= TARGETING + MOTION =======================
class Defense:
    """The bus over our defenders: per tick, world states in, rc per drone out."""

    def __init__(self, args, nd, na):
        import torch
        self.torch, self.nd, self.na = torch, nd, na
        bus = load_bus()
        self.face = dict(x=args.face_x, y=args.face_y)
        cfg = bus.BusPilotCfg(max_speed=args.max_speed)
        # engagement: zones by distance to the face (enter / leave + 0.5 m), strike radius, and the
        # leash (how far out from the face a defender may go: up to --wall-gap off the tag-10 wall)
        cfg.r_alert_in, cfg.r_alert_out = args.alert_radius, args.alert_radius + 0.5
        cfg.r_engage_in, cfg.r_engage_out = args.engage_radius, args.engage_radius + 0.5
        cfg.strike_radius = args.strike_radius
        cfg.x_max = max(cfg.x_min + 0.5, args.face_x - args.wall_gap)
        cfg.y_min = args.face_y - args.room_y[1] + args.wall_margin   # side walls, face-local
        cfg.y_max = args.face_y - args.room_y[0] - args.wall_margin
        if nd != len(cfg.post_y):
            cfg.post_y = tuple(np.linspace(-1.0, 1.0, nd).tolist()) if nd > 1 else (0.0,)
        self.cfg = cfg
        geom = sys.modules["_baked_face_geometry"].Face(args.face_width, args.face_height, args.face_bottom)
        self.pilot = bus.BusPilot(cfg, geom, 1, nd, na, PERIOD_S, "cpu")
        self.quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(1, nd, 4)
        self.rc_cap = args.rc_cap
        self.hist = [[] for _ in range(nd)]          # (t, world pos) per defender
        self.last = {}

    # face-local <-> world (a half turn about Z through the face centre)
    def to_face(self, w):
        w = np.asarray(w, float)
        return np.stack([self.face["x"] - w[..., 0], self.face["y"] - w[..., 1], w[..., 2]], -1)

    def from_face(self, f):
        return self.to_face(f)                       # the half turn is its own inverse

    @staticmethod
    def vel_world(v_face):
        return np.array([-v_face[0], -v_face[1], v_face[2]])

    def observe(self, i, t_fix, p_world):
        h = self.hist[i]
        if not h or t_fix > h[-1][0] + 1e-3:
            h.append((t_fix, np.asarray(p_world, float)))
            self.hist[i] = [x for x in h if t_fix - x[0] <= 1.0]

    def velocity(self, i):
        h = self.hist[i]
        if len(h) < 2:
            return np.zeros(3)
        t1, p1 = h[-1]
        older = [x for x in h if t1 - x[0] >= VEL_WINDOW_S] or h[:1]
        t0, p0 = older[-1]
        return (p1 - p0) / max(t1 - t0, 1e-3)

    def tick(self, defenders, att):
        """defenders: per drone dict(p=world pos or None, psi, age (s since its fix), alive).
        att: (pos (A,3), vel (A,3), alive (A,)) world. Returns ([rc or None per drone], info)."""
        t = self.torch
        nd = self.nd
        d_pos = np.array([d["p"] if d["p"] is not None else [0.0, 0.0, 0.0] for d in defenders], float)
        d_alive = np.array([d["alive"] and d["p"] is not None for d in defenders])
        d_vel = np.array([self.velocity(i) for i in range(nd)])
        a_pos, a_vel, a_alive = att
        f = lambda x: t.tensor(np.asarray(x), dtype=t.float32)[None]  # noqa: E731
        vel, _ = self.pilot.step(f(self.to_face(d_pos)), f(d_vel * [-1, -1, 1]), self.quat, t.tensor(d_alive)[None],
                                 f(self.to_face(a_pos)), f(np.asarray(a_vel) * [-1, -1, 1]), t.tensor(a_alive)[None],
                                 t.tensor(np.array([d["age"] < LOST_TRANSLATE_S for d in defenders]))[None])
        vel, role = vel[0].numpy(), self.pilot.role[0].tolist()
        target, strike = self.from_face(self.pilot.target[0].numpy()), self.pilot.strike[0].tolist()
        rcs = []
        for i, d in enumerate(defenders):
            if not d_alive[i]:
                rcs.append(None)
                continue
            p_int = internal_from_world(d_pos[i])
            if d["age"] > LOST_TRANSLATE_S:            # no fix: hold, turn back to tag 10
                err = wrap_deg(face_tag_yaw(p_int) - d["psi"])
                rcs.append((0, 0, 0, 0 if abs(err) < YAW_DEADBAND_DEG else int(math.copysign(RECOVER_YAW_RC, err))))
                continue
            yaw = yaw_to_tag10(p_int, d["psi"])
            if role[i] == -1 and not strike[i]:       # hold: waypoint to the post
                rcs.append(self.to_waypoint(i, p_int, d["psi"], target[i], d_pos, d_alive) + (yaw,))
            else:                                      # guard / spare / strike: the bus's velocity
                rcs.append(velocity_rc(self.vel_world(vel[i]), d["psi"], self.rc_cap) + (yaw,))
        names = {-1: "hold", -2: "spare", -3: "out"}
        info = {"role": [("STRIKE A%d" % r if s else names.get(r, "guard A%d" % r)) for r, s in zip(role, strike)],
                "target": target, "zone": self.pilot.att_state[0].tolist(), "vel": vel}
        return rcs, info

    def to_waypoint(self, i, p_int, psi, target_w, d_pos, d_alive):
        """swarm_formation's waypoint law (P on position, spacing push, rc kick), capped at HOLD_RC."""
        e = internal_from_world(target_w) - p_int
        v = SF_KP_POS * e if np.linalg.norm(e) > 0.05 else np.zeros(3)
        for k in range(self.nd):
            if k == i or not d_alive[k]:
                continue
            q = internal_from_world(d_pos[k])
            dv = np.array([p_int[0] - q[0], 0.0, p_int[2] - q[2]])
            dist = float(np.linalg.norm(dv))
            if dist < SF_D_SAFE_M:
                v = v + SF_K_REP * (SF_D_SAFE_M - dist) * (dv / dist if dist > 1e-6 else np.array([1.0, 0, 0]))
        nxz = math.hypot(v[0], v[2])
        if nxz > HOLD_RC:
            v[0], v[2] = v[0] * HOLD_RC / nxz, v[2] * HOLD_RC / nxz
        s, c = math.sin(math.radians(psi)), math.cos(math.radians(psi))
        fwd, right, up = v[0] * s + v[2] * c, v[0] * c - v[2] * s, -float(np.clip(v[1], -20, 20))
        mag = math.hypot(fwd, right)
        if 0.5 < mag < SF_MIN_RC_XY:                  # too small to move a Tello: scale up
            fwd, right = fwd * SF_MIN_RC_XY / mag, right * SF_MIN_RC_XY / mag
        return kick(right), kick(fwd), kick(up)


# swarm_formation's waypoint constants (its control()), kept here so --bench and --selftest need no cv2
SF_KP_POS, SF_D_SAFE_M, SF_K_REP, SF_MIN_RC_XY = 40.0, 0.6, 60.0, 14


def add_geometry_args(ap):
    ap.add_argument("--face-x", type=float, default=4.2, help="breach face plane, world X (m out from the tag-10 wall)")
    ap.add_argument("--face-y", type=float, default=0.0, help="face centre, world Y")
    ap.add_argument("--face-width", type=float, default=3.0)
    ap.add_argument("--face-height", type=float, default=2.0)
    ap.add_argument("--face-bottom", type=float, default=0.5)
    ap.add_argument("--room-y", type=float, nargs=2, default=[-2.0, 3.2], help="side walls, world Y")
    ap.add_argument("--wall-margin", type=float, default=0.5, help="guards stay this far off the side walls")
    ap.add_argument("--max-speed", type=float, default=MAX_SPEED, help="planner top speed, m/s")
    ap.add_argument("--alert-radius", type=float, default=ALERT_RADIUS_M,
                    help="an attacker this close to the face (m) gets a guard")
    ap.add_argument("--engage-radius", type=float, default=ENGAGE_RADIUS_M,
                    help="an attacker this close to the face (m) may be struck")
    ap.add_argument("--strike-radius", type=float, default=STRIKE_RADIUS_M,
                    help="an engaged attacker this close to its guard (m) is attacked head-on")
    ap.add_argument("--wall-gap", type=float, default=WALL_GAP_M,
                    help="the leash: defenders go out to this far from the tag-10 wall (m)")
    ap.add_argument("--rc-cap", type=int, default=100, help="|rc| cap on the sticks a, b, c")


# ======================= MOCK DRONES (--mock: the whole program without hardware) =======================
class MockDrone:
    """swarm_formation.Drone's interface on the Tello rc model: takeoff, rc, land, tag fixes facing tag 10."""

    def __init__(self, num, start_world, fly):
        self.num, self.name, self.fly = num, f"drone{num}", fly
        self.pos, self.vel, self.psi = np.array(start_world, float), np.zeros(3), 0.0
        self.rc, self.flying = (0, 0, 0, 0), False
        self.airborne = self.landed = self.landing = False
        self.land_thread, self.overlay, self.dry_noted = None, None, False
        self.battery, self.st, self.yaw_acc = 90, {"h": "0"}, 0.0
        self.fix, self.seen_t, self.tags_now = None, 0.0, []
        self.lock, self.stop = threading.Lock(), threading.Event()
        threading.Thread(target=self._physics, daemon=True).start()

    def log(self, msg):
        log_line(f"[{self.name}] {msg}")

    def _physics(self):
        dt = 0.02
        while not self.stop.wait(dt):
            with self.lock:
                if self.flying:
                    a, b, c, yaw = [0 if abs(x) < 5 else x for x in self.rc]
                    fwd, right = body_axes(self.psi)
                    want = a * GAIN_RIGHT * right + b * GAIN_FWD * fwd + np.array([0, 0, c * GAIN_UP])
                    self.vel += (want - self.vel) * min(1.0, dt / 0.3)
                    self.psi = wrap_deg(self.psi + yaw * dt)
                    self.yaw_acc += abs(yaw * dt)
                    self.pos = self.pos + self.vel * dt
                    self.pos[2] = max(self.pos[2], 0.0)
                now = time.time()
                p_int = internal_from_world(self.pos)
                if self.flying and abs(wrap_deg(face_tag_yaw(p_int) - self.psi)) < 60:
                    self.fix = (p_int + np.random.normal(0, 0.03, 3), self.psi, now - 0.1, None)
                    self.seen_t, self.tags_now = now - 0.1, [10]
                self.st = {"h": str(int(self.pos[2] * 100)), "tof": str(int(self.pos[2] * 100))}

    def send(self, cmd, timeout=10, quiet=False):
        if cmd == "takeoff" and self.fly:
            with self.lock:
                self.flying = True
            for _ in range(40):                       # the firmware climbs to ~0.8 m
                with self.lock:
                    self.pos[2] = min(0.8, self.pos[2] + 0.04)
                time.sleep(0.05)
        if not quiet:
            self.log(f"{cmd} -> ok")
        return "ok"

    def send_noreply(self, cmd):
        if cmd.startswith("rc "):
            self.rc = tuple(int(x) for x in cmd.split()[1:])
        elif cmd in ("land", "emergency"):
            with self.lock:
                self.flying, self.vel, self.pos[2] = False, np.zeros(3), 0.0

    def _stv(self, key):
        try:
            return float(self.st[key])
        except (KeyError, ValueError):
            return None

    def height_cm(self):
        return self._stv("tof") if self.flying else None

    def on_ground(self):
        return (self._stv("h") or 0) <= 5

    def has_fix(self, now):
        return self.fix is not None and now - self.seen_t < 0.5

    def est(self, now):
        if self.fix is None:
            return None
        p, psi, t, _ = self.fix
        return p, psi, now - t

    def land(self):
        if not self.airborne or self.landed:
            return
        self.send_noreply("land")
        self.landed = True
        self.log("landed")

    def close(self):
        self.stop.set()


def mock_preflight(d, fly):
    d.log(f"PASS (battery {d.battery}%, MOCK drone)")
    return True


# ======================= FLYING =======================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", required=True, help="our defender drone numbers, e.g. 6,7,9")
    ap.add_argument("--fly", action="store_true", help="actually fly (default dry-run)")
    ap.add_argument("--headless", action="store_true", help="no windows; q+Enter = land all, e+Enter = emergency")
    ap.add_argument("--duration", type=float, default=0.0, help="land all after this many s of defending (0 = never)")
    ap.add_argument("--start-height", type=int, default=START_HEIGHT_CM, help="climb to this after takeoff, cm")
    ap.add_argument("--lost-land", type=float, default=6.0, help="a drone with no tag this long lands (out), s")
    ap.add_argument("--rc-step", type=int, default=30, help="max change of a, b, c per tick")
    ap.add_argument("--num-attackers", type=int, default=2)
    ap.add_argument("--attackers", default="", help="attacker names in the feed, in slot order (default: as first seen)")
    ap.add_argument("--feed-port", type=int, default=9200)
    ap.add_argument("--feed-frame", choices=["map", "world"], default="map")
    ap.add_argument("--fake-feed", action="store_true", help="publish scripted attackers on the feed port (testing)")
    ap.add_argument("--fake-speed", type=float, default=0.5, help="--fake-feed attacker speed, m/s")
    ap.add_argument("--config", default=None, help="dongles JSON (default config/dongles.json)")
    ap.add_argument("--camera-dir", default=None)
    ap.add_argument("--map", default=None)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--mock", action="store_true",
                    help="no hardware: simulated drones on the rc model + scripted attackers (implies --fake-feed)")
    ap.add_argument("--ignore-tags", default="1,4,7,22,24",
                    help="mapped tags never used (default: the floor tags mapped face-down in the 2026-10-24 map)")
    add_geometry_args(ap)
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "..", "core"))
    import swarm_formation as SF                       # link, video, tags, localize, preflight
    SF.Drone.log = lambda self, msg: log_line(f"[{self.name}] {msg}")   # atomic lines (umbrella PASS count)
    from paths import CAMERA_DIR, DONGLES_PATH, MAP_PATH, TAGS_PATH
    fly = args.fly
    only = {int(x) for x in args.only.split(",") if x.strip()}
    print(f"MODE: {'FLY' if fly else 'DRY-RUN (no takeoff, no rc)'}", flush=True)
    origin, tag_map, sizes = SF.load_tag_files(args.map or str(MAP_PATH), args.tags or str(TAGS_PATH))
    for k in {int(x) for x in args.ignore_tags.split(",") if x.strip()} - {SF.TAG_ID}:
        tag_map.pop(k, None)
    print(f"Map: tags {sorted(tag_map)}", flush=True)
    if args.mock:
        args.fake_feed = True
        n_only = sorted(only)
        ys = np.linspace(-1.0, 1.0, len(n_only)) if len(n_only) > 1 else [0.0]
        drones = [MockDrone(n, [args.face_x - 0.8, args.face_y + y, 0.0], fly) for n, y in zip(n_only, ys)]
        preflight = mock_preflight
        print("MOCK: simulated drones on the Tello rc model (no hardware)", flush=True)
    else:
        try:
            drones = SF.load_drones(args.config or str(DONGLES_PATH), only, args.camera_dir or str(CAMERA_DIR), fly,
                                    tag_map, sizes, SF.MapCheck(tag_map))
        except OSError as ex:
            sys.exit(f"Cannot open the drones' adapters ({ex.strerror}): USB Wi-Fi sticks plugged in? (connect.py)")
        preflight = SF.preflight
    nd, na = len(drones), args.num_attackers
    defense = Defense(args, nd, na)
    feed = AttackerFeed(args.feed_port, args.feed_frame, [n for n in args.attackers.split(",") if n], na)
    print(f"[defend] {[d.name for d in drones]} guard the face at X={args.face_x} m ({args.face_width}x"
          f"{args.face_height} m from {args.face_bottom} m); feed UDP :{args.feed_port} ({args.feed_frame} frame)",
          flush=True)
    stop_fake = threading.Event()
    if args.fake_feed:
        threading.Thread(target=fake_feed, args=(args.feed_port, args.feed_frame, defense.face, args.fake_speed,
                                                 stop_fake), daemon=True).start()
        print(f"[defend] FAKE FEED: scripted attackers at {args.fake_speed} m/s", flush=True)

    if not all(SF.parallel(drones, lambda d: preflight(d, fly)).values()):
        for d in drones:
            d.close()
        sys.exit("Preflight FAILED -> nothing takes off.")

    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log_path = os.path.join(HERE, time.strftime("logs/defend_%Y%m%d_%H%M%S.jsonl"))
    logf = open(log_path, "w")
    for d in drones:
        d.state, d.sub, d.sub_t, d.searched, d.turn_start = "CLIMB", "LOOK", 0.0, 0.0, 0.0
        d.last_rc, d.to_state, d.out_why = (0, 0, 0, 0), None, None
    defend_t = None
    t0 = time.time()

    def ev(msg):                                     # one event line, stamped with the run time
        log_line(f"[event t={time.time() - t0:5.1f}] {msg}")

    prev_role = ["-"] * nd
    prev_zone = [None] * na
    prev_fix = [None] * nd
    lost_since = [None] * nd
    prev_fresh = [None] * na
    prev_alive = [True] * na
    n_role_changes = 0
    top_speed = [0.0] * nd
    last_p = [None] * nd
    no_feed_warned = False
    to_abort, to_wait_noted = threading.Event(), False
    try:
        if fly:
            if sys.stdin.isatty():
                SF.termios.tcflush(sys.stdin, SF.termios.TCIFLUSH)
            answer = input("All PASS. Space clear, drones 1 m+ apart? Type FLY: ").strip()
            if answer != "FLY":
                sys.exit(f"Aborted, no takeoff (got {answer!r}).")
            t0 = time.time()
            ev("FLY confirmed: run clock started, staggered takeoff")

            # swarm_formation.py's takeoff: 2 tries, `height?` when the reply is lost or stale
            # (the 10-24 run got "error" and a late battery answer "82"); the main loop also calls
            # a drone airborne from the status stream (h >= 60 cm for 0.5 s). to_busy stays True
            # while this thread may still talk to the drone, so "land all" waits for it.
            def to(d):
                try:
                    time.sleep(SF.TAKEOFF_STAGGER_S * drones.index(d))
                    ok = False
                    for k in range(2):
                        if d.to_state is True:             # status stream already saw it airborne
                            ok = True
                            break
                        if to_abort.is_set():
                            break
                        d.airborne = True
                        ev(f"{d.name}: sending takeoff" + (" (retry)" if k else ""))
                        reply = d.send("takeoff", 20)
                        if reply == "ok":
                            ev(f"{d.name}: takeoff ok (height {d.height_cm()} cm)")
                            ok = True
                            break
                        time.sleep(1.0)
                        hdm = "".join(c for c in str(d.send("height?", 5)) if c.isdigit())
                        ev(f"{d.name}: takeoff reply {reply!r}; height? {hdm or '-'} dm, status h {d._stv('h')} cm")
                        if hdm and int(hdm) >= 3:
                            ev(f"{d.name}: takeoff reply lost but airborne -> ok")
                            ok = True
                            break
                    if d.to_state is not True:
                        d.to_state = ok
                        if not ok:
                            ev(f"{d.name}: takeoff FAILED" + (" (aborted)" if to_abort.is_set() else " after 2 tries"))
                finally:
                    d.to_busy = False
            for d in drones:
                d.to_state, d.to_busy, d.h_ok_since = "pending", True, None
                threading.Thread(target=to, args=(d,), daemon=True).start()
        else:
            for d in drones:
                d.to_state = True
        if not fly:
            t0 = time.time()
            ev("DRY-RUN: run clock started (no takeoff, no rc)")
        for d in drones:
            d.sub_t = t0
        term_key = SF.start_terminal_keys() if args.headless else None
        print(">>> RUNNING (Ctrl+C = land all" + (", q+Enter = land all" if args.headless else "") + ")", flush=True)
        last_print = 0.0
        while True:
            t_loop = now = time.time()
            for d in drones:                               # airborne per the status stream?
                if d.to_state == "pending":
                    hh = d._stv("h")
                    if hh is not None and hh >= 60:
                        d.h_ok_since = d.h_ok_since or now
                        if now - d.h_ok_since > 0.5:
                            d.to_state = True
                            ev(f"{d.name}: airborne (status h {hh:.0f} cm)")
                    else:
                        d.h_ok_since = None
            if any(d.to_state is False for d in drones):
                to_abort.set()
                busy = [d.name for d in drones if getattr(d, "to_busy", False)]
                if busy:                                   # never send land into a pending takeoff
                    if not to_wait_noted:
                        to_wait_noted = True
                        print(f"A takeoff failed -> waiting for {', '.join(busy)}'s takeoff reply, then landing all",
                              flush=True)
                    for d in drones:                       # airborne ones hover meanwhile
                        if fly and d.to_state is True and not d.landed and not d.landing:
                            d.send_noreply("rc 0 0 0 0")
                            d.last_rc = (0, 0, 0, 0)
                    time.sleep(0.1)
                    continue
                print("A takeoff failed -> landing all", flush=True)
                return
            if all(d.landed or d.landing for d in drones):
                print("No defenders left.", flush=True)
                return
            feed.drain(now)
            cmds, states = {}, []

            for i, d in enumerate(drones):
                up = not d.landed and not d.landing
                h = d.height_cm()
                e = d.est(now)
                if up and d.to_state is True:
                    if d.battery is not None and d.battery < SF.LAND_BATTERY:
                        SF.land_async(d, f"battery {d.battery}%", fly)
                    if fly and d.state == "DEFEND" and d.on_ground():
                        d.out_why = "on the floor (collision?)"
                        d.landed = True
                        d.log("on the floor -> OUT")
                    fx = d.has_fix(now)
                    if d.state == "CLIMB":
                        c = 0 if h is None else int(np.clip(0.5 * (args.start_height - h), -20, 20))
                        if fx or h is None or abs(args.start_height - h) < 10 or now - d.sub_t > CLIMB_TIMEOUT_S:
                            why = ("already sees a tag" if fx else "no height reading" if h is None
                                   else f"at {h} cm" if abs(args.start_height - h) < 10 else f"climb timeout at {h} cm")
                            ev(f"{d.name}: CLIMB -> SEARCH ({why})")
                            d.state, d.sub, d.sub_t = "SEARCH", "LOOK", now
                        cmds[d.num] = (0, 0, c, 0)
                    elif d.state == "SEARCH":
                        if fx:
                            d.state = "DEFEND"
                            d.log(f"LOCALIZED from tags {d.tags_now}: world "
                                  f"{np.round(world_from_internal(e[0]), 2).tolist()}")
                        else:
                            yaw = 0
                            if d.sub == "LOOK":
                                if now - d.sub_t > SEARCH_LOOK_S:
                                    if d.searched == 0 or d.searched // 90 != (d.searched + SEARCH_STEP_DEG) // 90:
                                        ev(f"{d.name}: no trusted tag in view -> turning {SEARCH_STEP_DEG:.0f} deg "
                                           f"(turned {d.searched:.0f} of {SEARCH_MAX_DEG:.0f} deg so far)")
                                    d.sub, d.sub_t, d.turn_start = "TURN", now, d.yaw_acc
                            else:
                                turned = (d.yaw_acc - d.turn_start) if d.st else (now - d.sub_t) * 40.0
                                if turned >= SEARCH_STEP_DEG:
                                    d.searched += turned
                                    d.sub, d.sub_t = "LOOK", now
                                else:
                                    yaw = SEARCH_YAW_RC
                            if d.searched > SEARCH_MAX_DEG:
                                SF.land_async(d, "full turn without a trusted tag", fly)
                            cmds[d.num] = (0, 0, 0, yaw)
                    elif now - d.seen_t > args.lost_land:
                        if not d.landing and not getattr(d, "lost_noted", False):
                            d.lost_noted = True
                            ev(f"{d.name}: LOST for {now - d.seen_t:.1f} s (> --lost-land {args.lost_land}) -> LANDING, OUT")
                        d.out_why = f"no tag for {now - d.seen_t:.1f} s"
                        SF.land_async(d, d.out_why, fly)
                up = not d.landed and not d.landing
                p_w = None
                if e is not None:
                    p_w = world_from_internal(e[0])
                    if h is not None and 10 <= h <= 400:
                        p_w[2] = h / 100.0                     # height sensor, not the tag pose
                    if d.has_fix(now):
                        defense.observe(i, d.fix[2], p_w)
                    if up and d.has_fix(now):                  # speed over >= 1 s (a single fix is too noisy)
                        if last_p[i] is None:
                            last_p[i] = (now, p_w.copy())
                        elif now - last_p[i][0] >= 1.0:
                            v = float(np.linalg.norm(p_w[:2] - last_p[i][1][:2])) / (now - last_p[i][0])
                            top_speed[i] = max(top_speed[i], v)
                            last_p[i] = (now, p_w.copy())
                fx_now = d.has_fix(now)
                if d.state == "DEFEND" and up and prev_fix[i] is not None and fx_now != prev_fix[i]:
                    if not fx_now:
                        lost_since[i] = now
                        ev(f"{d.name}: LOST TAGS (fix {now - d.seen_t:.1f} s old) -> no translation, turning back "
                           f"to the tags; lands after {args.lost_land:.0f} s")
                    else:
                        ev(f"{d.name}: tags back after {now - (lost_since[i] or now):.1f} s (tags {d.tags_now})")
                prev_fix[i] = fx_now
                states.append(dict(p=p_w, psi=e[1] if e else 0.0, age=now - d.seen_t,
                                   alive=up and d.to_state is True and d.state == "DEFEND"))

            a_pos, a_vel, a_alive, a_known = feed.state(now)
            info = None
            if any(s["alive"] for s in states):
                if defend_t is None:
                    defend_t = now
                    print(">>> DEFENDING", flush=True)
                    ev("planner ON: defenders " + ", ".join(d.name for d, s in zip(drones, states) if s["alive"])
                       + f"; attackers known {int(a_known.sum())}/{na}, feed packets {feed.datagrams}")
                rcs, info = defense.tick(states, (a_pos, a_vel, a_alive))
                for d, rc in zip(drones, rcs):
                    if rc is not None:
                        cmds[d.num] = rc
                for i, d in enumerate(drones):
                    r = info["role"][i]
                    if r != prev_role[i]:
                        n_role_changes += 1
                        extra = ""
                        if r.startswith("STRIKE") or r.startswith("guard"):
                            j = int(r.split("A")[-1])
                            if states[i]["p"] is not None:
                                extra = (f": A{j} at ({a_pos[j][0]:.1f} {a_pos[j][1]:.1f} {a_pos[j][2]:.1f}), "
                                         f"{np.linalg.norm(a_pos[j] - states[i]['p']):.1f} m away")
                        ev(f"{d.name}: role {prev_role[i]} -> {r}{extra}")
                        prev_role[i] = r
                for j in range(na):
                    z = info["zone"][j]
                    if z != prev_zone[j]:
                        if prev_zone[j] is not None or z:
                            ev(f"A{j}: zone {prev_zone[j]} -> {z} (0 far, 1 alert, 2 engage), "
                               f"{np.linalg.norm(a_pos[j][:1] - args.face_x):.1f} m from the face")
                        prev_zone[j] = z

            for d in drones:
                rc = cmds.get(d.num)
                if rc is None or d.landed or d.landing:
                    continue
                lr = d.last_rc
                rc = tuple(int(np.clip(rc[k], lr[k] - args.rc_step, lr[k] + args.rc_step)) for k in range(3)) + (rc[3],)
                if fly:
                    d.send_noreply("rc %d %d %d %d" % rc)
                d.last_rc = rc

            fresh = a_known & ((now - feed.tracker.seen) < FEED_STALE_S)
            for j in range(na):
                if fresh[j] != prev_fresh[j] and a_known[j]:
                    ev(f"A{j}: feed " + ("LIVE" if prev_fresh[j] is None else "RESUMED" if fresh[j]
                                         else f"STALE (no update for {FEED_STALE_S:.0f} s): position held"))
                if a_known[j]:                             # unknown -> stays None, so first sight reads LIVE
                    prev_fresh[j] = bool(fresh[j])
                if prev_alive[j] and a_known[j] and not a_alive[j]:
                    near = [(np.linalg.norm(a_pos[j] - s["p"]), d.name) for d, s in zip(drones, states) if s["p"] is not None]
                    by = min(near) if near else None
                    ev(f"A{j}: DOWN at ({a_pos[j][0]:.1f} {a_pos[j][1]:.1f} {a_pos[j][2]:.1f})"
                       + (f", nearest defender {by[1]} {by[0]:.2f} m" if by else "")
                       + f" | attackers left {int(a_alive.sum())}")
                prev_alive[j] = bool(a_alive[j]) or not a_known[j]
            if defend_t is not None and not no_feed_warned and feed.datagrams == 0 and now - defend_t > 3.0:
                no_feed_warned = True
                ev(f"WARNING: no attacker packets on UDP :{args.feed_port} after 3 s of defending "
                   f"(bad packets: {feed.bad}) -> defenders only hold their posts. Check the port / the organizers' feed.")
            roles = info["role"] if info else ["-"] * nd
            logf.write(json.dumps({
                "t": round(now - t0, 3),
                "def": [{"name": d.name, "state": d.state, "out": d.out_why or (d.landed or d.landing),
                         "pos": None if s["p"] is None else np.round(s["p"], 3).tolist(), "psi": round(s["psi"], 1),
                         "age": round(s["age"], 2), "role": roles[i], "rc": list(d.last_rc),
                         "target": None if info is None else np.round(info["target"][i], 3).tolist()}
                        for i, (d, s) in enumerate(zip(drones, states))],
                "att": {"pos": np.round(a_pos, 3).tolist(), "alive": a_alive.tolist(), "known": a_known.tolist(),
                        "zone": None if info is None else info["zone"]},
            }) + "\n")
            if now - last_print >= 1.0:
                last_print = now
                parts = []
                for i, (d, s) in enumerate(zip(drones, states)):
                    st = "OUT" if d.landed or d.landing else d.state
                    pos = "?" if s["p"] is None else "(%.1f %.1f %.1f)" % tuple(s["p"])
                    parts.append(f"{d.name} {st} {roles[i]} {pos} fix {min(s['age'], 99):.1f}s rc{list(d.last_rc)}")
                zones = info["zone"] if info else ["-"] * na
                att = " ".join(f"A{j}:" + ("(%.1f %.1f %.1f)" % tuple(a_pos[j]) if a_known[j] else "?")
                               + ("" if a_alive[j] or not a_known[j] else " DOWN") + f" z{zones[j]}" for j in range(na))
                print(f"[defend] t={now - t0:5.1f} | " + " | ".join(parts) + f" | {att} | feed {feed.datagrams}"
                      + ("" if fly else " | DRY"), flush=True)
            if args.duration and defend_t is not None and now - defend_t > args.duration:
                print(">>> duration over -> landing all", flush=True)
                return
            if not args.headless:
                for d in drones:
                    if d.overlay is not None:
                        SF.cv2.imshow(d.name, SF.cv2.resize(d.overlay, (480, 360)))
            k = term_key() if args.headless else SF.cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                print("q -> landing all", flush=True)
                return
            if k == ord(" ") and fly:
                print("!!! EMERGENCY: motors off all", flush=True)
                for d in drones:
                    d.send_noreply("emergency")
                    d.landed = True
                return
            time.sleep(max(0.0, PERIOD_S - (time.time() - t_loop)))
    except KeyboardInterrupt:
        print("\nCtrl+C -> landing all", flush=True)
    finally:
        stop_fake.set()
        to_abort.set()                                     # no takeoff retries once we are landing
        if fly:
            for d in drones:
                if d.land_thread is not None:
                    d.land_thread.join(20)
            SF.parallel([d for d in drones if d.airborne and not d.landed], lambda d: d.land())
        for d in drones:
            d.close()
        logf.close()
        if not args.headless:
            SF.cv2.destroyAllWindows()
        a_pos, a_vel, a_alive, a_known = feed.state(time.time())
        log_line("=" * 70)
        log_line(f"SUMMARY: ran {time.time() - t0:.1f} s, defending {0 if defend_t is None else time.time() - defend_t:.1f} s, "
                 f"feed packets {feed.datagrams} (bad {feed.bad}), role changes {n_role_changes}")
        for i, d in enumerate(drones):
            log_line(f"  {d.name}: last state {d.state}, last role {prev_role[i]}, "
                     f"{'OUT: ' + str(d.out_why) if d.out_why else 'landed at the end'}, top speed {top_speed[i]:.2f} m/s (horizontal, 1 s average)")
        for j in range(na):
            name = feed.names[j] if j < len(feed.names) else "?"
            log_line(f"  A{j} {name}: " + ("never seen in the feed" if not a_known[j] else "DOWN" if not a_alive[j] else "alive at the end"))
        log_line("=" * 70)
        print(f"log: {log_path}\nDone.", flush=True)


# ======================= BENCH: the closed loop without drones =======================
PATTERN_NAMES = ("straight", "swerve", "arc", "feint", "weave", "jink")


class PatternAttackers:
    """Evasive scripted attackers for --bench --patterns: a numpy port of the attacker scripts in
    the teammate PPO-defender task (RL-based-drone-interceptors, defend_task/tello_defend_real_env.py,
    same parameter ranges). Per attacker, per episode: one sideways pattern relative to its dive
    direction (straight, swerve, arc, feint, weave, jink), plus independent modifiers: an altitude
    sinusoid, reactive dodging of the nearest defender, a mid-flight speed burst, a launch delay."""

    PROBS = (0.15, 0.2, 0.15, 0.15, 0.2, 0.15)
    Z_BAND = (0.6, 2.2)

    def __init__(self, rng, na, speed, only=None):
        u = rng.uniform
        self.na = na
        self.pat = (np.full(na, only) if only is not None else rng.choice(6, na, p=self.PROBS)).astype(int)
        side = np.where(rng.random(na) < 0.5, -1.0, 1.0)
        amp = np.zeros(na)
        for k, (lo, hi, signed) in {1: (0.3, 1.0, False), 2: (0.2, 0.7, True), 3: (0.6, 1.2, True),
                                    4: (0.4, 1.0, True), 5: (0.6, 1.2, False)}.items():
            amp = np.where(self.pat == k, u(lo, hi, na) * (side if signed else 1.0), amp)
        self.amp, self.freq, self.phase = amp, u(0.8, 2.5, na), u(0, 2 * np.pi, na)
        self.ratio, self.phase2 = u(1.7, 3.1, na), u(0, 2 * np.pi, na)
        self.jink_T, self.jink_seed = u(0.4, 1.2, na), u(0, 100, na)
        self.feint_t = u(1.0, 4.0, na)
        self.alt = np.stack([(rng.random(na) < 0.5) * u(0.2, 0.6, na), u(0.6, 1.8, na), u(0, 2 * np.pi, na)], 1)
        self.dodge = rng.random(na) < 0.4
        bursting = rng.random(na) < 0.3
        self.burst_t = np.where(bursting, u(1.5, 5.0, na), 1e9)
        self.burst_m = np.where(bursting, u(1.3, 1.8, na), 1.0)
        self.launch = u(0.0, 4.0, na)
        self.speed = speed

    def velocity(self, t, ap_, goal, dp, d_up):
        """World velocity (na, 3) at time t; zero before launch."""
        v = np.zeros((self.na, 3))
        for j in range(self.na):
            tf = t - self.launch[j]
            if tf < 0:
                continue
            d = goal[j] - ap_[j]
            d = d / max(np.linalg.norm(d), 1e-6)
            lat = np.array([-d[1], d[0], 0.0])
            a, f, ph, pat = self.amp[j], self.freq[j], self.phase[j], self.pat[j]
            side = {0: 0.0, 1: a * np.sin(f * tf + ph), 2: a, 3: a * (1.0 if tf < self.feint_t[j] else -1.0),
                    4: a * (np.sin(f * tf + ph) + 0.6 * np.sin(self.ratio[j] * f * tf + self.phase2[j])) / 1.6,
                    5: a * np.sign(np.sin(12.9898 * np.floor(tf / max(self.jink_T[j], 0.1)) + 78.233 * self.jink_seed[j]) + 1e-3)}[pat]
            d = d + side * lat
            vz = self.alt[j, 0] * np.sin(self.alt[j, 1] * tf + self.alt[j, 2])
            lo, hi = self.Z_BAND
            vz = abs(vz) + 0.3 if ap_[j, 2] < lo + 0.15 else -abs(vz) - 0.3 if ap_[j, 2] > hi - 0.15 else vz
            d = d + np.array([0.0, 0.0, vz])
            up = np.flatnonzero(d_up)
            if self.dodge[j] and up.size:                # sidestep the nearest live defender
                rel = ap_[j] - dp[up]
                dist = np.linalg.norm(rel, axis=1)
                k = int(dist.argmin())
                away = np.array([-rel[k, 1], rel[k, 0], 0.0])
                away = away / max(np.linalg.norm(away), 1e-6)
                away = away * np.sign(away @ lat + 1e-3)
                d = d + max(0.0, 1.2 * (1.0 - dist[k] / 1.2)) * away
            d = d / max(np.linalg.norm(d), 1e-6)
            spd = self.speed[j] * (self.burst_m[j] if tf >= self.burst_t[j] else 1.0)
            v[j] = d * min(spd, ATT_MAX_SPEED)
        return v


def bench(argv):
    """Episodes of the real tick() code against straight-line attackers (or, with --patterns, the
    teammate task's evasive ones: see PatternAttackers), on a Tello rc model:
    sticks -> body velocity with the fitted gains and deadband, first-order lag, tag fixes with
    noise and delay, the feed with delay and noise. Capture: centres < 0.35 m (both down)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--att-speed", type=float, nargs=2, default=[1.0, 1.5])
    ap.add_argument("--tau", type=float, default=0.3, help="velocity response time constant, s")
    ap.add_argument("--loiter", type=float, default=0.0, help="attackers hover up to this long (s) before rushing")
    ap.add_argument("--patterns", action="store_true",
                    help="evasive attackers (swerve, arc, feint, weave, jink + altitude, dodge, burst, launch delay)")
    ap.add_argument("--pattern", choices=PATTERN_NAMES, default=None, help="with --patterns: only this pattern")
    add_geometry_args(ap)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    nd, na, dt = 3, 2, PERIOD_S
    fx, fy, fw, fh, fb = args.face_x, args.face_y, args.face_width, args.face_height, args.face_bottom
    breaches = caps = lost = 0
    by_pat = {k: [0, 0] for k in PATTERN_NAMES}     # pattern -> [breaching attackers, attackers flown]
    t_end = []
    vmax_seen = 0.0
    for ep in range(args.bench):
        defense = Defense(args, nd, na)
        trk = FeedTracker(na)
        posts = defense.from_face(np.array([[defense.cfg.post_depth, y, defense.cfg.post_height]
                                            for y in defense.cfg.post_y]))
        dp, dvel, psi = posts.copy(), np.zeros((nd, 3)), np.zeros(nd)
        d_up = np.ones(nd, bool)
        a0 = np.stack([rng.uniform(0.3, 0.6, na), rng.uniform(-1.5, 1.5, na), rng.uniform(1.0, 2.0, na)], 1)
        goal = np.stack([np.full(na, fx + 0.3), fy + rng.uniform(-0.45, 0.45, na) * fw, fb + rng.uniform(0.1, 0.9, na) * fh], 1)
        a_spd = rng.uniform(*args.att_speed, na)
        wait = rng.uniform(0.0, args.loiter, na)
        av = (goal - a0) / np.linalg.norm(goal - a0, axis=1, keepdims=True) * a_spd[:, None]
        pats = (PatternAttackers(rng, na, a_spd, None if args.pattern is None else PATTERN_NAMES.index(args.pattern))
                if args.patterns else None)
        ap_, a_up = a0.copy(), np.ones(na, bool)
        hist = []                                        # (t, true attacker pos) for the feed delay
        outcome = None
        for k in range(int(30.0 / dt)):
            t = k * dt
            hist.append((t, ap_.copy(), a_up.copy()))
            old = [h for h in hist if t - h[0] >= 0.2]
            if old and k % 1 == 0:
                tm, pm, am = old[-1]
                for j in range(na):
                    trk.update(j, tm, pm[j] + rng.normal(0, FEED_NOISE_M, 3), t)
                trk.alive = am.copy()
            for i in range(nd):
                if d_up[i]:
                    defense.observe(i, t - 0.1, dp[i] + rng.normal(0, 0.03, 3))
            states = [dict(p=dp[i] + rng.normal(0, 0.03, 3), psi=psi[i] + rng.normal(0, 1.0), age=0.1, alive=bool(d_up[i]))
                      for i in range(nd)]
            pos, vel, alive, _ = trk.state(t)
            rcs, _ = defense.tick(states, (pos, vel, alive))
            for i, rc in enumerate(rcs):                 # the Tello: sticks -> velocity
                if rc is None or not d_up[i]:
                    dvel[i] = 0.0
                    continue
                a, b, c, yaw = [0 if abs(x) < 5 else x for x in rc]
                fwd, right = body_axes(psi[i])
                want = a * GAIN_RIGHT * right + b * GAIN_FWD * fwd + np.array([0, 0, c * GAIN_UP])
                dvel[i] += (want - dvel[i]) * min(1.0, dt / args.tau)
                psi[i] = wrap_deg(psi[i] + yaw * 1.0 * dt)
                vmax_seen = max(vmax_seen, float(np.linalg.norm(dvel[i])))
            dp += dvel * dt
            if pats is not None:
                ap_ = np.where(a_up[:, None], ap_ + pats.velocity(t, ap_, goal, dp, d_up) * dt, ap_)
                ap_[:, 1] = np.clip(ap_[:, 1], args.room_y[0] + 0.2, args.room_y[1] - 0.2)   # the side walls
                ap_[:, 0] = np.maximum(ap_[:, 0], 0.2)                                       # the tag-10 wall
            else:
                moving = a_up & (t >= wait)
                ap_ = np.where(moving[:, None], ap_ + av * dt, ap_)
            for i in range(nd):                          # mutual kill on contact
                for j in range(na):
                    if d_up[i] and a_up[j] and np.linalg.norm(dp[i] - ap_[j]) < 0.35:
                        d_up[i] = a_up[j] = False
                        caps += 1
            for j in range(na):
                if a_up[j] and ap_[j, 0] >= fx:
                    a_up[j] = False
                    if abs(ap_[j, 1] - fy) <= fw / 2 and fb <= ap_[j, 2] <= fb + fh:
                        outcome = "breach"
                        if pats is not None:
                            by_pat[PATTERN_NAMES[pats.pat[j]]][0] += 1
            if outcome or not a_up.any():
                break
        if pats is not None:
            for j in range(na):
                by_pat[PATTERN_NAMES[pats.pat[j]]][1] += 1
        breaches += outcome == "breach"
        lost += int((~d_up).sum())
        t_end.append(t)
    n = max(args.bench, 1)
    kind = "straight at the face" if pats is None else \
        "evasive" + (f" ({args.pattern})" if args.pattern else "") + ", launch 0-4 s, burst capped at " + f"{ATT_MAX_SPEED} m/s"
    print(f"[bench] {n} episodes, attackers {args.att_speed[0]}-{args.att_speed[1]} m/s {kind}: "
          f"breach {100.0 * breaches / n:.1f}%, captures {caps / n:.2f}/ep, defenders lost {lost / n:.2f}/ep, "
          f"resolved in {np.mean(t_end):.1f} s (median {np.median(t_end):.1f}), "
          f"top defender speed {vmax_seen:.2f} m/s (rc gains cap {100 * GAIN_FWD:.2f}-{100 * GAIN_RIGHT:.2f} m/s at rc 100)")
    if args.patterns:
        print("[bench] breaches by attacker pattern: " + ", ".join(
            f"{k} {b}/{c}" for k, (b, c) in by_pat.items() if c), flush=True)
    return breaches / n


# ======================= SELF-TEST =======================
def selftest():
    import importlib.util
    # 1. the baked planner is the sim's current code
    sim_dir = os.path.join(HERE, "..", "..", "simulation_infra")
    if os.path.isdir(sim_dir):
        spec = importlib.util.spec_from_file_location("build_defend", os.path.join(HERE, "build_defend.py"))
        bd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bd)
        assert bd.baked_sources() == BAKED, "baked sim is stale: run build_defend.py"
        print("[selftest] baked planner == simulation_infra sources")
    # 2. frames and sticks
    p_int = np.array([0.5, 0.2, -3.0])
    assert np.allclose(world_from_internal(p_int), [3.0, 0.5, 1.02])
    assert np.allclose(internal_from_world(world_from_internal(p_int)), p_int)
    assert np.allclose(world_from_map(np.array([0.5, -0.2, 3.0])), [3.0, 0.5, 1.02])
    a, b, c = velocity_rc(np.array([-0.5, 0.0, 0.0]), 0.0, 100)
    assert a == 0 and b > 60 and c == 0, (a, b, c)          # toward tag 10 = forward stick
    a, b, c = velocity_rc(np.array([0.0, 0.4, 0.0]), 90.0, 100)
    assert b > 40 and a == 0, (a, b, c)                     # turned 90 deg clockwise: +Y ahead
    assert velocity_rc(np.zeros(3), 0.0, 100) == (0, 0, 0)
    # 3. the feed over loopback UDP: map frame, names, a down flag, the filter's velocity
    port = 9299
    feed = AttackerFeed(port, "map", [], 2)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for k in range(8):
        msg = {"attackers": {"TELLO-3": {"p": [-0.5, 0.3, 0.5 + 0.06 * k]},
                             "TELLO-4": {"p": [0.8, 0.0, 0.5 + 0.06 * k], "alive": k < 7}}}
        tx.sendto(json.dumps(msg).encode(), ("127.0.0.1", port))
        time.sleep(0.05)
        feed.drain(time.time())
    pos, vel, alive, known = feed.state(time.time())
    assert known.all() and alive.tolist() == [True, False], alive
    assert 0.6 < vel[0, 0] < 1.5, vel                       # 1.2 m/s out of the wall (+X)
    # 4. the bus picks waypoints: idle -> posts in front of the face; a rush -> a guard goes out
    args = argparse.Namespace(face_x=4.2, face_y=0.0, face_width=3.0, face_height=2.0, face_bottom=0.5,
                              room_y=[-2.0, 3.2], wall_margin=0.5, max_speed=1.5, rc_cap=100,
                              alert_radius=ALERT_RADIUS_M, engage_radius=ENGAGE_RADIUS_M,
                              strike_radius=STRIKE_RADIUS_M, wall_gap=WALL_GAP_M)
    defense = Defense(args, 3, 2)
    posts = defense.from_face(np.array([[0.8, y, 1.5] for y in (-1.0, 0.0, 1.0)]))
    st = [dict(p=posts[i] + [-0.3, 0.0, 0.0], psi=0.0, age=0.1, alive=True) for i in range(3)]
    rcs, info = defense.tick(st, (np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2, bool)))
    assert info["role"] == ["hold"] * 3 and all(r[1] < 0 for r in rcs), (info["role"], rcs)   # back to the posts
    att = (np.array([[2.0, 0.3, 1.5], [0.5, -1.0, 1.2]]), np.array([[1.4, 0, 0], [1.4, 0, 0]]), np.ones(2, bool))
    for _ in range(5):
        rcs, info = defense.tick(st, att)
    assert any(r.startswith(("guard", "STRIKE", "spare")) for r in info["role"]), info["role"]
    print(f"[selftest] rush: roles {info['role']}, zones {info['zone']}, rc {rcs}")
    # 5. a lost fix: no translation
    st[1]["age"] = 1.0
    rcs, _ = defense.tick(st, att)
    assert rcs[1][:3] == (0, 0, 0), rcs[1]
    # 6. the closed loop on the rc model
    rate = bench(["--bench", "40", "--att-speed", "0.6", "1.0"])
    assert rate < 0.5, f"bench breach rate {rate:.0%}"
    print("SELFTEST PASSED")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        selftest()
    elif sys.argv[1:2] == ["--bench"]:
        bench(sys.argv[1:])
    else:
        main()
