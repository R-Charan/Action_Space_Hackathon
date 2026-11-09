"""Record the bus defender on AS-Defend-Face-v0, with the face, zones, roles and tag sightings drawn in.

Same capture path and dotted-marker style as ``record_zone.py`` (the RTX viewport here renders
neither marker opacity nor glass). The pilot sees the attackers only through ``AttackerTracker``
(0.2 s telemetry delay, noise, Kalman velocity), exactly as ``evaluate_face.py`` scores it.

  face          red dots: the breach rectangle on the plane x = 0; grey dots: the box behind it
  floor lines   orange = engage, yellow = alert, as distance to the face; big dots at the enter
                distance, small dots at the leave distance (the hysteresis band)
  defenders     the Tello inside a dotted safety shell, colored by role: blue hold, cyan spare
                guard, magenta guarding (attacker in alert), green striking; a dot on top, white
                dots along the camera heading, a big dot at its target and a dotted line to its attacker
  attackers     the Tello painted red with a red dot on top while alive, grey once downed (it then lands)
  shadows       a floor disc under every drone and a dotted stem up to it
  tags          wall tags green while some defender decodes them, grey otherwise

    python -u scripts/record_face.py --episodes 2 --out outputs/videos/face_defense.mp4 \
        --frame_dir outputs/frames_face > outputs/record_face.log 2>&1
With --checkpoint (an AS-Attack-Face-v0 PPO model_*.zip, its vecnormalize pkl beside it) the task is
the attack env: the env's own bus pilot flies the defenders and the policy flies the attackers.
"""

import argparse
import ast
import dataclasses
import os
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(description="Record the bus defender on the breach face.")
parser.add_argument("--task", type=str, default="AS-Defend-Face-v0", help="Or AS-Defend-Face-Aggressive-v0.")
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--max_seconds", type=float, default=30.0)
parser.add_argument("--capture_every", type=int, default=None, help="Env steps per frame (default: ~30 fps).")
parser.add_argument("--ticks_per_capture", type=int, default=3)
parser.add_argument("--pilot_hz", type=float, default=13.5)
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="BusPilotCfg overrides.")
parser.add_argument("--env_set", nargs="*", default=[], metavar="KEY=VALUE", help="Env cfg overrides.")
parser.add_argument("--checkpoint", type=str, default=None, help="PPO attackers (AS-Attack-Face-v0).")
parser.add_argument("--keep", choices=("all", "breach"), default="all",
                    help="breach: keep only breach episodes, playing until --episodes of them (or --max_attempts).")
parser.add_argument("--max_attempts", type=int, default=40)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--frame_dir", type=str, required=True)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "enable_cameras": True, "livestream": 1})
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import action_space_kit.tasks  # noqa: F401, E402
import isaaclab.sim as sim_utils  # noqa: E402
from action_space_kit.tags.sensing import TagSensorCfg, camera_axes, sighting  # noqa: E402
from action_space_kit.tasks.direct.tello_defend.bus_pilot import (  # noqa: E402
    AttackerTracker, BusPilot, BusPilotCfg, TelemetryCfg, box_room_tags)
from action_space_kit.tasks.direct.tello_defend import room_layout  # noqa: E402
from action_space_kit.tasks.direct.tello_defend.zone_pilot import ENGAGE, yaw_from_quat  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from action_space_kit.rl.policy_export import export_policy  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

COLORS = {
    "breach": (0.9, 0.0, 0.0), "engage": (1.0, 0.12, 0.0), "alert": (0.9, 0.85, 0.0),
    "hold": (0.0, 0.1, 1.0), "spare": (0.0, 0.9, 0.9), "guard": (0.9, 0.0, 0.9), "strike": (0.1, 0.9, 0.2),
    "attacker": (0.95, 0.05, 0.05), "downed": (0.45, 0.45, 0.45), "white": (1.0, 1.0, 1.0),
}
NAMES = list(COLORS)
CI = {k: i for i, k in enumerate(NAMES)}
ROLE_NAME = {-1: "hold", -2: "spare", -3: "downed"}
STATE_NAME = {0: "outside", 1: "ALERT", 2: "ENGAGE"}
BAIT_NAME = {0: "dive", 1: "bait in", 2: "bait out"}


def apply(cfg, items, label):
    names = {f.name for f in dataclasses.fields(cfg)} if dataclasses.is_dataclass(cfg) else None
    for item in items:
        key, value = item.split("=", 1)
        if names is not None and key not in names or names is None and not hasattr(cfg, key):
            raise SystemExit(f"unknown {label} field: {key}")
        setattr(cfg, key, ast.literal_eval(value))
    return cfg


def markers(path, make):
    mat = lambda c: sim_utils.PreviewSurfaceCfg(diffuse_color=c, emissive_color=tuple(0.25 * x for x in c))  # noqa: E731
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path=path, markers={k: make(mat(c)) for k, c in COLORS.items()}))


def shell(radius: float, n: int, device) -> torch.Tensor:
    """(n, 3) points evenly spread over a sphere (Fibonacci lattice)."""
    k = torch.arange(n, device=device, dtype=torch.float32) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    rho = torch.sqrt(1.0 - z * z)
    return radius * torch.stack([rho * torch.cos(phi), rho * torch.sin(phi), z], -1)


def offset_line(half_width: float, r: float, spacing: float, device) -> torch.Tensor:
    """Floor points at distance r in front of the segment x = 0, |y| <= half_width: a line and two quarter arcs."""
    k = max(2, int(2 * half_width / spacing))
    y = torch.linspace(-half_width, half_width, k, device=device)
    pts = [torch.stack([torch.full_like(y, r), y], -1)]
    m = max(3, int(0.5 * math.pi * r / spacing))
    a = torch.linspace(0.0, 0.5 * math.pi, m, device=device)[1:]
    for s in (1.0, -1.0):
        pts.append(torch.stack([r * torch.cos(a), s * (half_width + r * torch.sin(a))], -1))
    p = torch.cat(pts)
    return torch.cat([p, torch.full_like(p[:, :1], 0.03)], -1)


def rect_outline(x: float, y0: float, y1: float, z0: float, z1: float, spacing: float, device) -> torch.Tensor:
    ys = torch.arange(y0, y1 + 1e-6, spacing, device=device)
    zs = torch.arange(z0, z1 + 1e-6, spacing, device=device)
    f = lambda v: torch.full_like(v, 0.0)  # noqa: E731
    return torch.cat([torch.stack([f(ys) + x, ys, f(ys) + z0], -1), torch.stack([f(ys) + x, ys, f(ys) + z1], -1),
                      torch.stack([f(zs) + x, f(zs) + y0, zs], -1), torch.stack([f(zs) + x, f(zs) + y1, zs], -1)])


def main() -> int:
    os.makedirs(args.frame_dir, exist_ok=True)
    for stale in os.listdir(args.frame_dir):
        os.remove(os.path.join(args.frame_dir, stale))

    torch.manual_seed(args.seed)
    learned = args.checkpoint is not None
    if learned and args.task == parser.get_default("task"):
        args.task = "AS-Attack-Face-v0"
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env_cfg.seed = args.seed
    apply(env_cfg, args.env_set, "env cfg")
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    dev, nd, na = u.device, u.cfg.num_defenders, u.cfg.num_attackers
    agents = u.cfg.possible_agents
    hold = max(1, round(1.0 / (args.pilot_hz * u.step_dt)))
    pcfg = BusPilotCfg()
    if u.room_tags is not None:  # the real room: its side walls bound the pilot (--set still overrides)
        pcfg.y_min, pcfg.y_max = room_layout.pilot_y_range(u.cfg)
    pcfg = apply(pcfg, args.set, "BusPilotCfg")
    tags = u.room_tags if u.room_tags is not None else box_room_tags(pcfg, dev)
    pilot = BusPilot(pcfg, u.face, 1, nd, na, hold * u.step_dt, dev, tags=tags)
    if learned:  # the attack env runs its own bus pilot; show its roles, fly the attackers by the policy
        pilot, n_stack = u.pilot, 2  # sb3_ppo_attack_cfg.yaml frame_stack
        policy = torch.jit.load(str(export_policy(args.checkpoint, n_stack, os.path.join(args.frame_dir, "policy.pt"))),
                                map_location=dev)
    if args.capture_every is None:
        args.capture_every = max(1, round(1.0 / (30.0 * u.step_dt)))
    tracker = AttackerTracker(TelemetryCfg(), 1, na, u.step_dt, dev)
    sensor = TagSensorCfg()
    origin = u.scene.env_origins[0]
    face = u.face
    hw, zb, zt = 0.5 * face.width, face.bottom, face.bottom + face.height

    big_dots = markers("/Visuals/face_big", lambda m: sim_utils.SphereCfg(radius=0.05, visual_material=m))
    small_dots = markers("/Visuals/face_small", lambda m: sim_utils.SphereCfg(radius=0.025, visual_material=m))
    caps = markers("/Visuals/face_caps", lambda m: sim_utils.SphereCfg(radius=0.06, visual_material=m))
    shell_dots = markers("/Visuals/face_shells", lambda m: sim_utils.SphereCfg(radius=0.012, visual_material=m))
    sphere_pts = shell(pcfg.sphere_radius, 120, dev)
    discs = markers("/Visuals/face_shadows", lambda m: sim_utils.CylinderCfg(radius=0.16, height=0.006, visual_material=m))
    tag_x = markers("/Visuals/face_tags_x", lambda m: sim_utils.CuboidCfg(size=(0.03, 0.4, 0.4), visual_material=m))
    tag_y = markers("/Visuals/face_tags_y", lambda m: sim_utils.CuboidCfg(size=(0.4, 0.03, 0.4), visual_material=m))
    tag_z = markers("/Visuals/face_tags_z", lambda m: sim_utils.CuboidCfg(size=(0.2, 0.2, 0.01), visual_material=m))

    # static scenery: (points, color, big?)
    static = []
    static.append((rect_outline(0.0, -hw, hw, zb, zt, 0.12, dev), CI["breach"], True))
    gy, gz = torch.meshgrid(torch.arange(-hw + 0.25, hw - 0.1, 0.25, device=dev),
                            torch.arange(zb + 0.25, zt - 0.1, 0.25, device=dev), indexing="ij")
    static.append((torch.stack([torch.zeros_like(gy), gy, gz], -1).reshape(-1, 3), CI["breach"], False))
    d = u.cfg.box_depth
    for x in (-d,):
        static.append((rect_outline(x, -hw, hw, 0.0, zt, 0.25, dev), CI["downed"], False))
    xs = torch.arange(-d, 0.0, 0.2, device=dev)
    for y in (-hw, hw):
        for z in (0.0, zt):
            static.append((torch.stack([xs, torch.full_like(xs, y), torch.full_like(xs, z)], -1), CI["downed"], False))
    static.append((rect_outline(0.0, -hw, hw, 0.0, zb, 0.25, dev), CI["downed"], False))  # the box below the face
    for name, r_in, r_out in (("engage", pcfg.r_engage_in, pcfg.r_engage_out), ("alert", pcfg.r_alert_in, pcfg.r_alert_out)):
        static.append((offset_line(hw, r_in, 0.25, dev), CI[name], True))
        static.append((offset_line(hw, r_out, 0.25, dev), CI[name], False))
    if u.room_tags is not None:  # the real room's walls, on the floor
        (x0, x1), (y0, y1) = u.cfg.arena_x, u.cfg.arena_y
        xs, ys = torch.arange(x0, x1 + 1e-6, 0.25, device=dev), torch.arange(y0, y1 + 1e-6, 0.25, device=dev)
        f = lambda v, c: torch.full_like(v, c)  # noqa: E731
        walls = torch.cat([torch.stack([xs, f(xs, y0), f(xs, 0.03)], -1), torch.stack([xs, f(xs, y1), f(xs, 0.03)], -1),
                           torch.stack([f(ys, x0), ys, f(ys, 0.03)], -1), torch.stack([f(ys, x1), ys, f(ys, 0.03)], -1)])
        static.append((walls, CI["white"], True))
    # tags by the axis their face looks along: walls facing x or y, floor tags facing up
    axis = tags[1].abs().argmax(-1)
    x_wall, y_wall, floor = axis == 0, axis == 1, axis == 2

    for name in ("attacker", "downed"):
        c = COLORS[name]
        mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=c, emissive_color=tuple(0.4 * x for x in c))
        mat_cfg.func(f"/Visuals/Materials/{name}", mat_cfg)
    painted = [None] * na

    def paint(alive):
        for j in range(na):
            name = "attacker" if bool(alive[j]) else "downed"
            if painted[j] != name:
                sim_utils.bind_visual_material(f"/World/envs/env_0/Attacker_{j}", f"/Visuals/Materials/{name}")
                painted[j] = name

    def role_name(i):
        r = int(pilot.role[0, i])
        if r in ROLE_NAME:
            return ROLE_NAME[r]
        return "strike" if bool(pilot.strike[0, i]) else "guard"

    def draw(d_pos, d_quat, d_alive, a_pos, a_alive, seen):
        big, big_c, small, small_c = [], [], [], []
        for pts, c, is_big in static:
            (big if is_big else small).append(pts)
            (big_c if is_big else small_c).append(torch.full((len(pts),), c, device=dev))
        yaw = yaw_from_quat(d_quat)
        cap_pos, cap_c, shadow_c = [], [], []

        def stem(p, c):
            k = int(float(p[2]) / 0.2)
            if k > 0:
                z = torch.arange(1, k + 1, device=dev) * 0.2
                small.append(torch.stack([p[0].expand(k), p[1].expand(k), z], -1))
                small_c.append(torch.full((k,), c, device=dev))

        for i in range(nd):
            c = CI[role_name(i)]
            cap_pos.append(d_pos[i] + torch.tensor([0.0, 0.0, 0.1], device=dev)); cap_c.append(c)
            shadow_c.append(c)
            stem(d_pos[i], c)
            if not bool(d_alive[i]):
                continue
            ahead = torch.stack([torch.cos(yaw[i]), torch.sin(yaw[i]), torch.zeros_like(yaw[i])])
            small.append(d_pos[i] + torch.tensor([0.25, 0.4, 0.55], device=dev).unsqueeze(-1) * ahead)
            small_c.append(torch.full((3,), CI["white"], device=dev))
            big.append(pilot.target[0, i : i + 1]); big_c.append(torch.tensor([c], device=dev))
            r = int(pilot.role[0, i])
            if r >= 0:
                t = torch.linspace(0.1, 0.9, 9, device=dev).unsqueeze(-1)
                small.append(d_pos[i] + t * (a_pos[r] - d_pos[i])); small_c.append(torch.full((9,), c, device=dev))
        for j in range(na):
            c = CI["attacker"] if bool(a_alive[j]) else CI["downed"]
            cap_pos.append(a_pos[j] + torch.tensor([0.0, 0.0, 0.1], device=dev)); cap_c.append(c)
            shadow_c.append(c)
            stem(a_pos[j], c)
        big_dots.visualize(torch.cat(big) + origin, marker_indices=torch.cat(big_c))
        small_dots.visualize(torch.cat(small) + origin, marker_indices=torch.cat(small_c))
        paint(a_alive)
        caps.visualize(torch.stack(cap_pos) + origin, marker_indices=torch.tensor(cap_c, device=dev))
        live = d_alive.nonzero().flatten()
        if len(live):
            shell_dots.visualize((d_pos[live].unsqueeze(1) + sphere_pts).reshape(-1, 3) + origin,
                                 marker_indices=torch.tensor(cap_c[:nd], device=dev)[live].repeat_interleave(len(sphere_pts)))
        else:
            shell_dots.visualize(torch.full((1, 3), -100.0, device=dev))
        floor_pts = torch.cat([d_pos, a_pos]) * torch.tensor([1.0, 1.0, 0.0], device=dev) + torch.tensor([0.0, 0.0, 0.01], device=dev)
        discs.visualize(floor_pts + origin, marker_indices=torch.tensor(shadow_c, device=dev))
        tag_c = torch.where(seen, CI["strike"], CI["downed"])
        tag_x.visualize(tags[0][x_wall] + origin, marker_indices=tag_c[x_wall])
        tag_y.visualize(tags[0][y_wall] + origin, marker_indices=tag_c[y_wall])
        if floor.any():
            tag_z.visualize(tags[0][floor] + origin, marker_indices=tag_c[floor])

    def states():
        d = lambda key, objs: torch.stack([getattr(x.data, key)[0] for x in objs])  # noqa: E731
        return (d("root_pos_w", u._defenders) - origin, d("root_lin_vel_w", u._defenders), d("root_quat_w", u._defenders),
                d("root_ang_vel_w", u._defenders), d("root_pos_w", u._attackers) - origin)

    viewport = get_active_viewport()
    u._site_markers.set_visibility(False)  # the base env's circular-site pad: not part of this task
    # broadside from the -y side: the face edge-on at the left, the attackers coming in from the right
    u.sim.set_camera_view(eye=[origin[0].item() + 2.6, origin[1].item() - 8.0, origin[2].item() + 4.2],
                          target=[origin[0].item() + 2.6, origin[1].item(), origin[2].item() + 0.9])
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20)
    frame = 0
    panels: dict[int, list[str]] = {}

    def capture(lines):
        nonlocal frame
        capture_viewport_to_file(viewport, os.path.join(args.frame_dir, f"frame_{frame:05d}.png"))
        panels[frame] = lines
        for _ in range(args.ticks_per_capture):
            u.sim.render()
        frame += 1

    max_steps = int(args.max_seconds / u.step_dt)
    obs, _ = env.reset()
    kind = "PPO" if learned else "aggressive" if u.cfg.aggressive else "diving / baiting"
    attempts = args.max_attempts if args.keep == "breach" else args.episodes
    kept = 0
    for ep in range(attempts):
        if kept >= args.episodes:
            break
        pilot.reset(torch.tensor([0], device=dev))
        tracker.reset(torch.tensor([0], device=dev), states()[4][None], u._attacker_alive.clone())
        has_fix = torch.ones(1, nd, dtype=torch.bool, device=dev)
        no_fix_s = torch.zeros(nd, device=dev)
        actions = {a: torch.zeros(1, 4, device=dev) for a in agents}
        first = frame
        if learned:
            stack = torch.zeros(1, n_stack, sum(obs[a].shape[-1] for a in agents), device=dev)  # zero = no history
        for step in range(max_steps):
            d_pos, d_vel, d_quat, d_ang, a_pos = states()
            if learned:
                stack = torch.cat([stack[:, 1:], torch.cat([obs[a] for a in agents], -1)[:, None]], 1)
                with torch.no_grad():
                    act = policy(stack.reshape(1, -1)).view(1, len(agents), -1)
                actions = {a: act[:, i] for i, a in enumerate(agents)}
            elif step % hold == 0:
                a_est, a_vel, a_alive = tracker.estimate()
                vel, yaw_rate = pilot.step(d_pos[None], d_vel[None], d_quat[None], u._defender_alive.clone(),
                                           a_est, a_vel, a_alive, has_fix)
                act = torch.cat([vel / u.cfg.command.max_lin_speed, (yaw_rate / u.cfg.command.max_yaw_rate).unsqueeze(-1)], -1)
                actions = {a: act[:, i].clamp(-1.0, 1.0) for i, a in enumerate(agents)}
            obs = env.step(actions)[0]
            if bool(u.outcome_valid[0]):  # the env has already reset: this frame would show the next start
                break
            d_pos, d_vel, d_quat, d_ang, a_pos = states()
            if not learned:
                tracker.step(a_pos[None], u._attacker_alive)
            rot = matrix_from_quat(d_quat)
            cam = d_pos + rot @ torch.tensor(sensor.camera_offset, device=dev)
            _, fix, mask = sighting(sensor, cam, camera_axes(rot, sensor.camera_pitch), *tags, noise=True,
                                    speed=torch.linalg.norm(d_vel, dim=-1), yaw_rate=d_ang[:, 2])
            has_fix = fix[None]
            no_fix_s = torch.where(fix, torch.zeros_like(no_fix_s), no_fix_s + u.step_dt)
            if step % args.capture_every:
                continue
            u.sim.render()
            d_alive, a_alive_true = u._defender_alive[0], u._attacker_alive[0]
            draw(d_pos, d_quat, d_alive, a_pos, a_alive_true, mask.any(0))
            u.sim.render()
            t = step * u.step_dt
            downed = int((~a_alive_true).sum())
            lines = [f"bus defender vs {kind} attackers  ep {ep + 1}  t={t:5.1f} s  attackers downed {downed}"]
            for i in range(nd):
                r = int(pilot.role[0, i])
                name = role_name(i)
                role = name if r < 0 else f"{name} A{r}  margin {float(pilot.margin[0, i, r]):+.2f}"
                if bool(u.taking_off[0, i]):
                    role = f"takeoff  z {float(d_pos[i, 2]):.2f} m"
                fixs = "fix" if bool(fix[i]) else f"no fix {no_fix_s[i]:.1f}s"
                lines.append(f"D{i}  {role:<24} {fixs:<12} tags {int(mask[i].sum())}" if bool(d_alive[i]) else f"D{i}  downed")
            for j in range(na):
                fd = float(face.distance(a_pos[j]))
                st = STATE_NAME[int(pilot.att_state[0, j])] if bool(a_alive_true[j]) else "downed"
                mode = "PPO" if learned else BAIT_NAME[int(u._bait_phase[0, j])]
                lines.append(f"A{j}  {st:<8} {fd:4.1f} m from face  {mode}")
            capture(lines)
        else:
            obs, _ = env.reset()
        if bool(u.outcome_valid[0]):
            breached = bool(u.outcome_breached[0])
            captures = int(u.outcome_captures[0])
            lost = int(u.outcome_defenders_lost[0])
            out, clash = int(u.outcome_out[0]), int(u.outcome_att_collisions[0])
        else:
            breached, captures, lost = False, int((~u._attacker_alive[0]).sum()), int((~u._defender_alive[0]).sum())
            out = clash = 0
        verdict = ("BREACH" if breached else "WON" if captures >= na else "attackers collided" if clash
                   else "attackers out of play" if out else "held")
        result = (f"episode {ep + 1}: {verdict} -- {captures} attackers downed by defenders, {lost} defenders lost"
                  f"{f', {clash} attackers hit each other' if clash else ''}"
                  f"{f', {out} drones out of play' if out else ''}, {step * u.step_dt:.1f} s")
        print(f"[rec] {result}", flush=True)
        if args.keep == "breach" and not breached:
            for i in range(first, frame):  # dropped: their files are removed before encoding
                panels.pop(i, None)
            continue
        kept += 1
        if frame > first:  # hold the last frame 1.5 s with the result
            last = os.path.join(args.frame_dir, f"frame_{frame - 1:05d}.png")
            for _ in range(200):
                if os.path.exists(last):
                    break
                u.sim.render()
            final = panels[frame - 1] + ["", result.split(": ", 1)[1]]  # the banner once, on every held frame
            for _ in range(int(1.5 / (u.step_dt * args.capture_every))):
                shutil.copyfile(last, os.path.join(args.frame_dir, f"frame_{frame:05d}.png"))
                panels[frame] = final
                frame += 1

    for _ in range(120):
        u.sim.render()
    for i, lines in panels.items():
        path = os.path.join(args.frame_dir, f"frame_{i:05d}.png")
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        dr = ImageDraw.Draw(img)
        w = max(dr.textlength(line, font=font) for line in lines) + 24
        dr.rectangle((8, 8, 8 + w, 16 + 26 * len(lines)), fill=(0, 0, 0))
        for k, line in enumerate(lines):
            dr.text((20, 14 + 26 * k), line, font=font, fill=(255, 255, 255))
        img.save(path)
    # the kept frames, numbered without gaps for ffmpeg; the rest are deleted
    for name in os.listdir(args.frame_dir):
        if name.startswith("frame_") and int(name[6:11]) not in panels:
            os.remove(os.path.join(args.frame_dir, name))
    k = 0
    for i in sorted(panels):
        src = os.path.join(args.frame_dir, f"frame_{i:05d}.png")
        if os.path.exists(src):
            os.rename(src, os.path.join(args.frame_dir, f"clip_{k:05d}.png"))
            k += 1
    frames = sorted(f for f in os.listdir(args.frame_dir) if f.startswith("clip_"))
    print(f"[rec] kept {kept} episodes, wrote {len(frames)} frames", flush=True)
    if not frames:
        return 1

    import imageio_ffmpeg

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fps = round(1.0 / (u.step_dt * args.capture_every))  # real time
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(fps), "-i", os.path.join(args.frame_dir, "clip_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", args.out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(args.out):
        print(f"[rec] ffmpeg failed: {proc.stderr[-600:]}", flush=True)
        return 1
    print(f"[rec] wrote {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(code)  # with livestream on, simulation_app.close() can hang after the video is written
