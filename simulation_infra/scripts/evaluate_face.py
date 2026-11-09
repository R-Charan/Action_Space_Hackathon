# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Score the bus defender (or a baseline) on the breach-face task, AS-Defend-Face-v0.

The defender sees attackers the way the hardware round provides them: delayed, noisy poses and
alive flags (``AttackerTracker``). Reports the match outcomes and what the bus is built to keep:

  guarantee   for every attacker with an assigned guard, the guarding margin (Isaacs, equal
              speeds) on the TRUE positions: negative means the attacker could reach part of
              the face first; below the latency buffer means only the capture radius saves it
  tags        share of defender time with a mapped tag decodable / a fix this frame / lost
  spheres     closest approach between defenders, time inside each other's sphere

    python -u scripts/evaluate_face.py --pilot bus --episodes 64 > outputs/eval_face_bus.log 2>&1
    python -u scripts/evaluate_face.py --task AS-Defend-Face-Aggressive-v0 --pilot bus --episodes 64
"""

import argparse
import ast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a defender on the breach-face task.")
parser.add_argument("--task", type=str, default="AS-Defend-Face-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--pilot", type=str, default="bus", choices=["bus", "classical59", "chase", "zero"],
                    help="classical59: the bus through the deploy runner's contract (59 inputs per defender -> 4 "
                         "outputs, classical_policy.py) plus the runner's safety layer")
parser.add_argument("--pilot_hz", type=float, default=13.5, help="Command rate (the real rc rate).")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="BusPilotCfg overrides.")
parser.add_argument("--env_set", nargs="*", default=[], metavar="KEY=VALUE", help="Env cfg overrides.")
parser.add_argument("--telemetry_set", nargs="*", default=[], metavar="KEY=VALUE", help="TelemetryCfg overrides.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import dataclasses

import gymnasium as gym
import torch

import action_space_kit.tasks  # noqa: F401
from action_space_kit.tags.sensing import TagSensorCfg, camera_axes, detectable, sighting
from action_space_kit.tasks.direct.tello_defend import classical_policy as cp59, room_layout
from action_space_kit.tasks.direct.tello_defend.bus_pilot import AttackerTracker, BusPilot, BusPilotCfg, TelemetryCfg, box_room_tags
from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.utils import parse_env_cfg

LOST_AFTER_S = 0.4


def apply(cfg, items, label):
    names = {f.name for f in dataclasses.fields(cfg)} if dataclasses.is_dataclass(cfg) else None
    for item in items:
        key, value = item.split("=", 1)
        if names is not None and key not in names or names is None and not hasattr(cfg, key):
            raise SystemExit(f"unknown {label} field: {key}")
        setattr(cfg, key, ast.literal_eval(value))
    return cfg


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    apply(env_cfg, args_cli.env_set, "env cfg")
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    n, nd, na, dev = raw.num_envs, raw.cfg.num_defenders, raw.cfg.num_attackers, raw.device
    agents = raw.cfg.possible_agents
    hold = max(1, round(1.0 / (args_cli.pilot_hz * raw.step_dt)))
    pcfg = BusPilotCfg()
    if raw.room_tags is not None:  # the real room: its side walls bound the pilot (--set still overrides)
        pcfg.y_min, pcfg.y_max = room_layout.pilot_y_range(raw.cfg)
    pcfg = apply(pcfg, args_cli.set, "BusPilotCfg")
    tcfg = apply(TelemetryCfg(), args_cli.telemetry_set, "TelemetryCfg")
    tags = raw.room_tags if raw.room_tags is not None else box_room_tags(pcfg, dev)
    pilot = BusPilot(pcfg, raw.face, n, nd, na, hold * raw.step_dt, dev, tags=tags)
    tracker = AttackerTracker(tcfg, n, na, raw.step_dt, dev)
    sensor = TagSensorCfg()
    if args_cli.pilot == "classical59":
        # the runner's world frame: the face-local frame turned back (room_layout); 1 action unit = top speed
        c = raw.cfg
        frame = cp59.FaceFrame(face_x=room_layout.face_x(c) if c.room_map else 0.0,
                               center_y=c.face_center_y if c.room_map else 0.0,
                               width=c.face_width, height=c.face_height, bottom=c.face_bottom)
        site = torch.tensor([frame.face_x, frame.center_y, c.face_bottom + 0.5 * c.face_height], device=dev)
        vmax = c.command.max_lin_speed
        policies = [cp59.ClassicalPolicy(site, frame, nd, na, 1.0 / (hold * raw.step_dt), vmax, pcfg, dev) for _ in range(n)]
        applied = torch.zeros(n, nd, 4, device=dev)
    print(f"[face] task {args_cli.task}, pilot {args_cli.pilot}, {1.0 / (hold * raw.step_dt):.1f} Hz, face "
          f"{raw.face.width}x{raw.face.height} m, speeds {raw.cfg.attacker_speed} (cap {raw.cfg.command.max_lin_speed})", flush=True)
    print(f"[face] pilot {pcfg}\n[face] telemetry {tcfg}", flush=True)

    def local(objs, key="root_pos_w"):
        v = torch.stack([getattr(o.data, key) for o in objs], 1)
        return v - raw.scene.env_origins.unsqueeze(1) if key == "root_pos_w" else v

    env.reset()
    tracker.reset(torch.arange(n, device=dev), local(raw._attackers), raw._attacker_alive.clone())
    actions = {a: torch.zeros(n, 4, device=dev) for a in agents}
    has_fix = torch.ones(n, nd, dtype=torch.bool, device=dev)
    no_fix_s = torch.zeros(n, nd, device=dev)
    pair_i, pair_j = torch.triu_indices(nd, nd, 1, device=dev)

    episodes = breaches = wins = 0
    downed = defenders_lost = teamkills = outs = clashes = 0.0
    m = dict(def_steps=0, visible=0, fixed=0, lost=0, pair_steps=0, in_sphere=0, guard_steps=0, neg=0, thin=0,
             uncovered=0, active_steps=0, beyond=0, att_steps=0, att_fast=0, def_fast=0)
    vlim = raw.cfg.attacker_max_speed  # both sides fly identical drones
    vmax_att = vmax_def = 0.0
    min_sep, min_margin, max_x, max_lost = float("inf"), float("inf"), 0.0, 0.0
    buffer = tcfg.max_speed * (tcfg.delay_s + hold * raw.step_dt) - raw.cfg.collision_radius
    step = 0
    while episodes < args_cli.episodes:
        d_pos, d_vel, d_quat = local(raw._defenders), local(raw._defenders, "root_lin_vel_w"), local(raw._defenders, "root_quat_w")
        d_alive = raw._defender_alive.clone()
        a_est, a_vel, a_alive = tracker.estimate()
        if step % hold == 0:
            if args_cli.pilot == "bus":
                vel, yaw_rate = pilot.step(d_pos, d_vel, d_quat, d_alive, a_est, a_vel, a_alive, has_fix)
            elif args_cli.pilot == "classical59":
                vel = torch.zeros(n, nd, 3, device=dev)
                for e in range(n):
                    w_d, w_a = frame.to_local(d_pos[e]), frame.to_local(a_est[e])  # the half turn is its own inverse
                    obs = cp59.build_obs(w_d, frame.vel_to_world(d_vel[e]), w_a, frame.vel_to_world(a_vel[e]), a_alive[e],
                                         site, applied[e], torch.zeros(nd, device=dev), d_alive[e])
                    cmd = cp59.safety(policies[e].act(obs), w_d, d_alive[e], action_speed=vmax)
                    applied[e] = cmd
                    vel[e] = frame.vel_to_local(cmd[:, :3] * vmax)
                yaw_rate = torch.zeros(n, nd, device=dev)
            elif args_cli.pilot == "chase":
                rel = a_est.unsqueeze(1) - d_pos.unsqueeze(2)
                dist = torch.linalg.norm(rel, dim=-1).masked_fill(~a_alive.unsqueeze(1), float("inf"))
                tgt = rel.gather(2, dist.argmin(-1).view(n, nd, 1, 1).expand(-1, -1, 1, 3)).squeeze(2)
                vel = tgt / torch.linalg.norm(tgt, dim=-1, keepdim=True).clamp(min=1e-6) * pcfg.max_speed
                yaw_rate = torch.zeros(n, nd, device=dev)
            else:
                vel, yaw_rate = torch.zeros(n, nd, 3, device=dev), torch.zeros(n, nd, device=dev)
            act = torch.cat([vel / raw.cfg.command.max_lin_speed, (yaw_rate / raw.cfg.command.max_yaw_rate).unsqueeze(-1)], -1)
            actions = {a: act[:, i].clamp(-1.0, 1.0) for i, a in enumerate(agents)}

        env.step(actions)
        step += 1
        a_true = local(raw._attackers)
        tracker.step(a_true, raw._attacker_alive)
        # true speeds of live drones against the limit (+2 % for the velocity loop's ripple)
        a_spd = torch.linalg.norm(local(raw._attackers, "root_lin_vel_w"), dim=-1)
        d_spd = torch.linalg.norm(local(raw._defenders, "root_lin_vel_w"), dim=-1)
        a_up, d_up = raw._attacker_alive, raw._defender_alive
        vmax_att = max(vmax_att, float((a_spd * a_up).max()))
        vmax_def = max(vmax_def, float((d_spd * d_up).max()))
        m["att_steps"] += int(a_up.sum())
        m["att_fast"] += int(((a_spd > 1.02 * vlim) & a_up).sum())
        m["def_fast"] += int(((d_spd > 1.02 * vlim) & d_up).sum())

        # -- metrics on the post-step state ------------------------------------------------
        d_pos, d_vel, d_quat = local(raw._defenders), local(raw._defenders, "root_lin_vel_w"), local(raw._defenders, "root_quat_w")
        d_ang = local(raw._defenders, "root_ang_vel_w")
        up = raw._defender_alive
        rot = matrix_from_quat(d_quat.reshape(-1, 4))
        cam = d_pos.reshape(-1, 3) + rot @ torch.tensor(sensor.camera_offset, device=dev)
        cam_rot = camera_axes(rot, sensor.camera_pitch)
        visible, _ = detectable(sensor, cam, cam_rot, *tags)
        _, fix, _ = sighting(sensor, cam, cam_rot, *tags, noise=True,
                             speed=torch.linalg.norm(d_vel.reshape(-1, 3), dim=-1), yaw_rate=d_ang.reshape(-1, 3)[:, 2])
        has_fix = fix.view(n, nd)
        no_fix_s = torch.where(has_fix, torch.zeros_like(no_fix_s), no_fix_s + raw.step_dt)
        m["def_steps"] += int(up.sum())
        m["visible"] += int((visible.any(-1).view(n, nd) & up).sum())
        m["fixed"] += int((has_fix & up).sum())
        m["lost"] += int(((no_fix_s > LOST_AFTER_S) & up).sum())
        max_lost = max(max_lost, float((no_fix_s * up).max()))
        max_x = max(max_x, float((d_pos[..., 0] * up).max()))
        m["beyond"] += int(((d_pos[..., 0] > pcfg.x_max + 0.3) & up).sum())
        sep = torch.linalg.norm(d_pos[:, pair_i] - d_pos[:, pair_j], dim=-1)
        both = up[:, pair_i] & up[:, pair_j]
        m["pair_steps"] += int(both.sum())
        m["in_sphere"] += int(((sep < 2.0 * pcfg.sphere_radius) & both).sum())
        if both.any():
            min_sep = min(min_sep, float(sep[both].min()))
        if args_cli.pilot == "bus":
            live = raw._attacker_alive & (pilot.att_state >= 1)
            true_margin = raw.face.margin(d_pos.unsqueeze(2), a_true.unsqueeze(1), pilot.pts)  # (N, D, A)
            guard = (pilot.role.unsqueeze(-1) == torch.arange(na, device=dev)) & live.unsqueeze(1) & up.unsqueeze(-1)
            m["guard_steps"] += int(guard.sum())
            m["neg"] += int(((true_margin < 0) & guard).sum())
            m["thin"] += int(((true_margin < buffer) & guard).sum())
            if guard.any():
                min_margin = min(min_margin, float(true_margin[guard].min()))
            m["active_steps"] += int(live.sum())
            m["uncovered"] += int((live & ~guard.any(1)).sum())

        finished = raw.outcome_valid.nonzero(as_tuple=False).flatten()
        if len(finished):
            episodes += len(finished)
            br = raw.outcome_breached[finished]
            caps = raw.outcome_captures[finished]
            breaches += int(br.sum())
            downed += float(caps.sum())
            wins += int(((caps >= na) & ~br).sum())  # interceptions only; out-of-arena attackers don't count
            defenders_lost += float(raw.outcome_defenders_lost[finished].sum())
            teamkills += float(raw.outcome_teamkills[finished].sum())
            outs += float(raw.outcome_out[finished].sum())
            clashes += float(raw.outcome_att_collisions[finished].sum())
            pilot.reset(finished)
            if args_cli.pilot == "classical59":
                for e in finished.tolist():
                    policies[e].reset()
                applied[finished] = 0.0
            tracker.reset(finished, local(raw._attackers)[finished], raw._attacker_alive[finished].clone())
            no_fix_s[finished] = 0.0
            has_fix[finished] = True

    pct = lambda a, b: 100.0 * a / max(b, 1)  # noqa: E731
    e = max(episodes, 1)
    print(f"\n=== {args_cli.pilot} on {args_cli.task} ===", flush=True)
    print(f"episodes:              {episodes}", flush=True)
    print(f"breached:              {pct(breaches, episodes):.1f}%", flush=True)
    print(f"won (both intercepted): {pct(wins, episodes):.1f}%   (rest held to the {raw.cfg.episode_length_s:.0f} s timeout or attackers left the arena)", flush=True)
    print(f"attackers intercepted: {downed / e:.2f} per episode; defenders lost {defenders_lost / e:.2f} "
          f"({teamkills / e:.2f} to teammates); drones out of arena {outs / e:.2f}; attackers hit each other {clashes / e:.2f}", flush=True)
    if args_cli.pilot == "bus":
        print(f"guarantee:             margin < 0 on {pct(m['neg'], m['guard_steps']):.2f}% of guarded attacker-steps, "
              f"< buffer ({buffer:.2f} m) on {pct(m['thin'], m['guard_steps']):.2f}%; worst {min_margin:.2f} m", flush=True)
        print(f"uncovered attackers:   {pct(m['uncovered'], m['active_steps']):.2f}% of active attacker-steps had no guard", flush=True)
    print(f"tag decodable:         {pct(m['visible'], m['def_steps']):.1f}%; fix this frame {pct(m['fixed'], m['def_steps']):.1f}%; "
          f"lost (> {LOST_AFTER_S} s) {pct(m['lost'], m['def_steps']):.2f}%, longest {max_lost:.2f} s", flush=True)
    print(f"speed (limit {vlim} m/s): attackers max {vmax_att:.2f} m/s, over +2 % on {pct(m['att_fast'], m['att_steps']):.2f}% "
          f"of live steps; defenders max {vmax_def:.2f} m/s, over on {pct(m['def_fast'], m['def_steps']):.2f}%", flush=True)
    print(f"closest defenders:     {min_sep:.2f} m; inside a sphere {pct(m['in_sphere'], m['pair_steps']):.2f}% of pair time", flush=True)
    print(f"furthest out:          x = {max_x:.2f} m; past the leash+0.3 m {pct(m['beyond'], m['def_steps']):.2f}% of defender time", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
