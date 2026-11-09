"""Record the zone defender on AS-Defend-3v2-v0, with its zones, roles and tag sightings drawn in.

Captures the Kit viewport like ``record_approach.py`` (livestream mode keeps the viewport
alive headless). The RTX viewport here renders neither marker opacity nor glass (both come
out opaque or black), so zones are dotted rings and safety spheres are dotted shells.

  floor rings   red = breach (1.5 m), orange = engage, yellow = alert; big dots at the
                enter radius, small dots at the leave radius (the hysteresis band)
  defenders     the Tello inside a dotted shell the size of its safety sphere, colored by
                role: blue hold, cyan goalie, magenta shadowing (attacker in alert), green
                engaging; a solid dot of the same color on top, white dots ahead along the
                camera heading, a big dot at its target point, and a dotted line to the
                assigned attacker
  attackers     the Tello itself painted red while alive, grey once downed (it then lands)
  shadows       every drone has a floor disc straight below it and a dotted stem up to it,
                so position and height read correctly from the angled camera
  tags          wall tags (hardcoded room) green while some defender decodes them, grey otherwise

    python -u scripts/record_zone.py --episodes 2 --out outputs/videos/zone_defense.mp4 \
        --frame_dir outputs/frames_zone > outputs/record_zone.log 2>&1
"""

import argparse
import os
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(description="Record the zone defender.")
parser.add_argument("--task", type=str, default="AS-Defend-3v2-v0", help="Or AS-Defend-3v2-Aggressive-v0.")
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--bait_prob", type=float, default=None)
parser.add_argument("--max_seconds", type=float, default=20.0)
parser.add_argument("--capture_every", type=int, default=2, help="Env steps per frame (60 Hz env: 2 = 30 fps).")
parser.add_argument("--ticks_per_capture", type=int, default=3)
parser.add_argument("--pilot_hz", type=float, default=13.5)
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
from action_space_kit.tasks.direct.tello_defend.zone_pilot import (  # noqa: E402
    ALERT, ENGAGE, ZonePilot, ZonePilotCfg, room_tags, yaw_from_quat)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

COLORS = {
    "breach": (0.9, 0.0, 0.0), "engage": (1.0, 0.12, 0.0), "alert": (0.9, 0.85, 0.0),
    "hold": (0.0, 0.1, 1.0), "goalie": (0.0, 0.9, 0.9), "shadow": (0.9, 0.0, 0.9), "engaging": (0.1, 0.9, 0.2),
    "attacker": (0.95, 0.05, 0.05), "downed": (0.45, 0.45, 0.45), "white": (1.0, 1.0, 1.0),
}
NAMES = list(COLORS)
CI = {k: i for i, k in enumerate(NAMES)}
ROLE_NAME = {-1: "hold", -2: "goalie", -3: "downed"}
STATE_NAME = {0: "outside", 1: "ALERT", 2: "ENGAGE"}
BAIT_NAME = {0: "dive", 1: "bait in", 2: "bait out"}


def markers(path, make):
    # emissive: under the dome light plain diffuse colors wash out (orange read as yellow)
    mat = lambda c: sim_utils.PreviewSurfaceCfg(diffuse_color=c, emissive_color=tuple(0.25 * x for x in c))  # noqa: E731
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path=path, markers={k: make(mat(c)) for k, c in COLORS.items()}))


def shell(radius: float, n: int, device) -> torch.Tensor:
    """(n, 3) points evenly spread over a sphere (Fibonacci lattice)."""
    k = torch.arange(n, device=device, dtype=torch.float32) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    rho = torch.sqrt(1.0 - z * z)
    return radius * torch.stack([rho * torch.cos(phi), rho * torch.sin(phi), z], -1)


def ring(radius: float, spacing: float, z: float, device) -> torch.Tensor:
    k = max(8, int(2 * math.pi * radius / spacing))
    a = torch.arange(k, device=device) * (2 * math.pi / k)
    return torch.stack([radius * torch.cos(a), radius * torch.sin(a), torch.full_like(a, z)], -1)


def main() -> int:
    os.makedirs(args.frame_dir, exist_ok=True)
    for stale in os.listdir(args.frame_dir):
        os.remove(os.path.join(args.frame_dir, stale))

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env_cfg.seed = args.seed
    if args.bait_prob is not None:
        env_cfg.bait_prob = args.bait_prob
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    dev, nd, na = u.device, u.cfg.num_defenders, u.cfg.num_attackers
    agents = u.cfg.possible_agents
    hold = max(1, round(1.0 / (args.pilot_hz * u.step_dt)))
    pcfg = ZonePilotCfg()
    pilot = ZonePilot(pcfg, 1, nd, na, hold * u.step_dt, dev)
    sensor = TagSensorCfg()
    tags = room_tags(pcfg, dev)
    origin = u.scene.env_origins[0]

    big_dots = markers("/Visuals/zone_big", lambda m: sim_utils.SphereCfg(radius=0.05, visual_material=m))
    small_dots = markers("/Visuals/zone_small", lambda m: sim_utils.SphereCfg(radius=0.025, visual_material=m))
    caps = markers("/Visuals/zone_caps", lambda m: sim_utils.SphereCfg(radius=0.06, visual_material=m))
    shell_dots = markers("/Visuals/zone_shells", lambda m: sim_utils.SphereCfg(radius=0.012, visual_material=m))
    sphere_pts = shell(pcfg.sphere_radius, 160, dev)
    discs = markers("/Visuals/zone_shadows", lambda m: sim_utils.CylinderCfg(radius=0.16, height=0.006, visual_material=m))
    tag_x = markers("/Visuals/zone_tags_x", lambda m: sim_utils.CuboidCfg(size=(0.03, 0.4, 0.4), visual_material=m))
    tag_y = markers("/Visuals/zone_tags_y", lambda m: sim_utils.CuboidCfg(size=(0.4, 0.03, 0.4), visual_material=m))

    floor = []  # (points, color, big?)
    for name, r_in, r_out in (("breach", pcfg.r_breach, None), ("engage", pcfg.r_engage_in, pcfg.r_engage_out),
                              ("alert", pcfg.r_alert_in, pcfg.r_alert_out)):
        floor.append((ring(r_in, 0.25, 0.03, dev), CI[name], True))
        if r_out is not None:
            floor.append((ring(r_out, 0.25, 0.03, dev), CI[name], False))
    x_wall = tags[1][:, 0].abs() > 0.5  # tags on the +-x walls face along x

    # attackers: no marker, the Tello itself is painted (a material bound over the model's own)
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

    def draw(d_pos, d_quat, a_pos, alive, seen):
        big, big_c, small, small_c = [], [], [], []
        for pts, c, is_big in floor:
            (big if is_big else small).append(pts)
            (big_c if is_big else small_c).append(torch.full((len(pts),), c, device=dev))
        role, state = pilot.role[0], pilot.att_state[0]
        yaw = yaw_from_quat(d_quat)
        cap_pos, cap_c, shadow_c = [], [], []

        def stem(p, c):
            k = int(float(p[2]) / 0.2)
            if k > 0:
                z = torch.arange(1, k + 1, device=dev) * 0.2
                small.append(torch.stack([p[0].expand(k), p[1].expand(k), z], -1))
                small_c.append(torch.full((k,), c, device=dev))

        for i in range(nd):
            r = int(role[i])
            name = ROLE_NAME.get(r) or ("engaging" if int(state[r]) == ENGAGE else "shadow")
            c = CI[name]
            cap_pos.append(d_pos[i] + torch.tensor([0.0, 0.0, 0.1], device=dev)); cap_c.append(c)
            shadow_c.append(c)
            stem(d_pos[i], c)
            ahead = torch.stack([torch.cos(yaw[i]), torch.sin(yaw[i]), torch.zeros_like(yaw[i])])
            small.append(d_pos[i] + torch.tensor([0.25, 0.4, 0.55], device=dev).unsqueeze(-1) * ahead)
            small_c.append(torch.full((3,), CI["white"], device=dev))
            big.append(pilot.target[0, i : i + 1]); big_c.append(torch.tensor([c], device=dev))
            if r >= 0:
                t = torch.linspace(0.1, 0.9, 9, device=dev).unsqueeze(-1)
                small.append(d_pos[i] + t * (a_pos[r] - d_pos[i])); small_c.append(torch.full((9,), c, device=dev))
        for j in range(na):
            c = CI["attacker"] if bool(alive[j]) else CI["downed"]
            shadow_c.append(c)
            stem(a_pos[j], c)
        big_dots.visualize(torch.cat(big) + origin, marker_indices=torch.cat(big_c))
        small_dots.visualize(torch.cat(small) + origin, marker_indices=torch.cat(small_c))
        paint(alive)
        caps.visualize(torch.stack(cap_pos) + origin, marker_indices=torch.tensor(cap_c, device=dev))
        shell_dots.visualize((d_pos.unsqueeze(1) + sphere_pts).reshape(-1, 3) + origin,
                             marker_indices=torch.tensor(cap_c, device=dev).repeat_interleave(len(sphere_pts)))
        floor_pts = torch.cat([d_pos, a_pos]) * torch.tensor([1.0, 1.0, 0.0], device=dev) + torch.tensor([0.0, 0.0, 0.01], device=dev)
        discs.visualize(floor_pts + origin, marker_indices=torch.tensor(shadow_c, device=dev))
        tag_c = torch.where(seen, CI["engaging"], CI["downed"])
        tag_x.visualize(tags[0][x_wall] + origin, marker_indices=tag_c[x_wall])
        tag_y.visualize(tags[0][~x_wall] + origin, marker_indices=tag_c[~x_wall])

    def states():
        d = lambda key, objs: torch.stack([getattr(x.data, key)[0] for x in objs])  # noqa: E731
        return (d("root_pos_w", u._defenders) - origin, d("root_lin_vel_w", u._defenders), d("root_quat_w", u._defenders),
                d("root_ang_vel_w", u._defenders), d("root_pos_w", u._attackers) - origin, d("root_lin_vel_w", u._attackers))

    viewport = get_active_viewport()
    # high over a corner, the whole alert ring in view
    u.sim.set_camera_view(eye=[origin[0].item() + 6.5, origin[1].item() - 8.0, origin[2].item() + 8.5],
                          target=[origin[0].item(), origin[1].item() + 0.5, origin[2].item() + 0.5])
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20)
    frame = 0
    panels: dict[int, list[str]] = {}

    def capture(lines):
        nonlocal frame
        capture_viewport_to_file(viewport, os.path.join(args.frame_dir, f"frame_{frame:05d}.png"))
        panels[frame] = lines
        for _ in range(args.ticks_per_capture):
            u.sim.render()  # renders without stepping physics
        frame += 1

    max_steps = int(args.max_seconds / u.step_dt)
    for ep in range(args.episodes):
        env.reset() if ep == 0 else None
        pilot.reset(torch.tensor([0], device=dev))
        has_fix = torch.ones(1, nd, dtype=torch.bool, device=dev)
        no_fix_s = torch.zeros(nd, device=dev)
        actions = {a: torch.zeros(1, 4, device=dev) for a in agents}
        first, captures, breached = frame, 0, False
        for step in range(max_steps):
            d_pos, d_vel, d_quat, d_ang, a_pos, a_vel = states()
            alive = u._attacker_alive[0]
            if step % hold == 0:
                vel, yaw_rate = pilot.step(d_pos[None], d_vel[None], d_quat[None], a_pos[None], a_vel[None], alive[None], has_fix,
                                             u._defender_alive)
                act = torch.cat([vel / u.cfg.command.max_lin_speed, (yaw_rate / u.cfg.command.max_yaw_rate).unsqueeze(-1)], -1)
                actions = {a: act[:, i].clamp(-1.0, 1.0) for i, a in enumerate(agents)}
            alive_before = alive.clone()
            env.step(actions)
            if bool(u.outcome_valid[0]):  # the env has already reset: this frame would show the next start
                breached = bool(u.outcome_breached[0])
                captures = int(u.outcome_captures[0])
                break
            captures += int((alive_before & ~u._attacker_alive[0]).sum())
            d_pos, d_vel, d_quat, d_ang, a_pos, a_vel = states()
            rot = matrix_from_quat(d_quat)
            cam = d_pos + rot @ torch.tensor(sensor.camera_offset, device=dev)
            _, fix, mask = sighting(sensor, cam, camera_axes(rot, sensor.camera_pitch), *tags, noise=True,
                                    speed=torch.linalg.norm(d_vel, dim=-1), yaw_rate=d_ang[:, 2])
            has_fix = fix[None]
            no_fix_s = torch.where(fix, torch.zeros_like(no_fix_s), no_fix_s + u.step_dt)
            if step % args.capture_every:
                continue
            u.sim.render()
            draw(d_pos, d_quat, a_pos, u._attacker_alive[0], mask.any(0))
            u.sim.render()
            t = step * u.step_dt
            lines = [f"zone defender vs {'aggressive' if u.cfg.aggressive else 'diving'} attackers  ep {ep + 1}/{args.episodes}  t={t:5.1f} s  downed {captures}"]
            for i in range(nd):
                r = int(pilot.role[0, i])
                role = ROLE_NAME.get(r) or f"-> A{r} ({'engage' if int(pilot.att_state[0, r]) == ENGAGE else 'shadow'})"
                fixs = "fix" if bool(fix[i]) else f"no fix {no_fix_s[i]:.1f}s"
                lines.append(f"D{i}  {role:<18} {fixs:<12} tags {int(mask[i].sum())}")
            for j in range(na):
                r = float(torch.linalg.norm(a_pos[j, :2]))
                st = STATE_NAME[int(pilot.att_state[0, j])] if bool(u._attacker_alive[0, j]) else "downed"
                lines.append(f"A{j}  {st:<8} r={r:4.1f} m  {BAIT_NAME[int(u._bait_phase[0, j])]}")
            capture(lines)
        else:
            breached = False
            env.reset()
        verdict = "BREACH" if breached else ("WON" if captures >= na and not u.cfg.respawn_waves else "held")
        lost = int(u.outcome_defenders_lost[0]) if bool(u.outcome_valid[0]) and hasattr(u, "outcome_defenders_lost") else int((~u._defender_alive[0]).sum())
        result = f"episode {ep + 1}: {verdict} -- {captures} attackers downed, {lost} defenders lost, {step * u.step_dt:.1f} s"
        print(f"[rec] {result}", flush=True)
        if frame > first:  # hold the last frame 1.5 s with the result
            last = os.path.join(args.frame_dir, f"frame_{frame - 1:05d}.png")
            for _ in range(200):
                if os.path.exists(last):
                    break
                u.sim.render()
            for _ in range(int(1.5 / (u.step_dt * args.capture_every))):
                shutil.copyfile(last, os.path.join(args.frame_dir, f"frame_{frame:05d}.png"))
                panels[frame] = panels[frame - 1] + ["", result.split(": ", 1)[1]]
                frame += 1

    for _ in range(120):
        u.sim.render()
    for i, lines in panels.items():
        path = os.path.join(args.frame_dir, f"frame_{i:05d}.png")
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        d = ImageDraw.Draw(img)
        w = max(d.textlength(line, font=font) for line in lines) + 24
        d.rectangle((8, 8, 8 + w, 16 + 26 * len(lines)), fill=(0, 0, 0))
        for k, line in enumerate(lines):
            d.text((20, 14 + 26 * k), line, font=font, fill=(255, 255, 255))
        img.save(path)
    frames = sorted(f for f in os.listdir(args.frame_dir) if f.endswith(".png"))
    print(f"[rec] wrote {len(frames)} frames", flush=True)
    if not frames:
        return 1

    import imageio_ffmpeg

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fps = round(1.0 / (u.step_dt * args.capture_every))  # real time
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(fps), "-i", os.path.join(args.frame_dir, "frame_%05d.png"),
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
