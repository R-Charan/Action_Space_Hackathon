# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Score the classical zone defender (or a baseline) on the 3-vs-2 defend task.

Besides interceptions and breaches it measures what the zone pilot is built to guarantee:

  tags        share of defender time with a mapped tag decodable (``tags/sensing.py``,
              hardcoded room in ``ZonePilotCfg``), and time spent lost (no fix > 0.4 s)
  spheres     closest approach between defenders, time inside each other's sphere,
              and collisions (< 0.25 m)
  bait        how far from the site defenders were drawn, and time past the leash

    python -u scripts/evaluate_zone.py --pilot zone  --episodes 40 > outputs/eval_zone.log 2>&1
    python -u scripts/evaluate_zone.py --pilot chase --episodes 40 > outputs/eval_zone_chase.log 2>&1
    python -u scripts/evaluate_zone.py --pilot zone --set r_engage_in=3.2 leash_radius=3.6
"""

import argparse
import ast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate the zone defender on AS-Defend-3v2-v0.")
parser.add_argument("--task", type=str, default="AS-Defend-3v2-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=40)
parser.add_argument("--pilot", type=str, default="zone", choices=["zone", "chase", "zero"])
parser.add_argument("--pilot_hz", type=float, default=13.5, help="Command rate (the real rc rate); 0 = every env step.")
parser.add_argument("--bait_prob", type=float, default=None, help="Override the env's chance that an attacker baits.")
parser.add_argument("--waves", type=int, default=None, choices=[0, 1], help="Override respawn_waves.")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ZonePilotCfg overrides.")
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
from action_space_kit.tasks.direct.tello_defend.zone_pilot import ZonePilot, ZonePilotCfg, room_tags
from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.utils import parse_env_cfg

LOST_AFTER_S = 0.4  # the room task's ``lost``: no fix for this long
COLLISION_M = 0.25


def pilot_cfg() -> ZonePilotCfg:
    cfg = ZonePilotCfg()
    fields = {f.name for f in dataclasses.fields(cfg)}
    for item in args_cli.set:
        key, value = item.split("=", 1)
        if key not in fields:
            raise SystemExit(f"unknown ZonePilotCfg field: {key}")
        setattr(cfg, key, ast.literal_eval(value))
    return cfg


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if args_cli.bait_prob is not None:
        env_cfg.bait_prob = args_cli.bait_prob
    if args_cli.waves is not None:
        env_cfg.respawn_waves = bool(args_cli.waves)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    n, nd, na, dev = raw.num_envs, raw.cfg.num_defenders, raw.cfg.num_attackers, raw.device
    agents = raw.cfg.possible_agents
    hold = 1 if args_cli.pilot_hz <= 0 else max(1, round(1.0 / (args_cli.pilot_hz * raw.step_dt)))
    tick_s = hold * raw.step_dt

    pcfg = pilot_cfg()
    pilot = ZonePilot(pcfg, n, nd, na, tick_s, dev)
    sensor = TagSensorCfg()
    tags = room_tags(pcfg, dev)
    print(f"[zone] task {args_cli.task}, pilot {args_cli.pilot}, {1.0 / tick_s:.1f} Hz commands "
          f"(every {hold} env steps), bait_prob {raw.cfg.bait_prob}, waves {raw.cfg.respawn_waves}", flush=True)
    print(f"[zone] cfg {pcfg}", flush=True)

    def def_alive():
        # under mutual_kill a defender can be downed (and lands); otherwise all are always in play
        return getattr(raw, "_defender_alive", torch.ones(n, nd, dtype=torch.bool, device=dev))

    def states():
        o = raw.scene.env_origins.unsqueeze(1)
        d = lambda key, objs: torch.stack([getattr(x.data, key) for x in objs], 1)  # noqa: E731
        return (d("root_pos_w", raw._defenders) - o, d("root_lin_vel_w", raw._defenders), d("root_quat_w", raw._defenders),
                d("root_ang_vel_w", raw._defenders), d("root_pos_w", raw._attackers) - o, d("root_lin_vel_w", raw._attackers))

    env.reset()
    actions = {a: torch.zeros(n, 4, device=dev) for a in agents}
    has_fix = torch.ones(n, nd, dtype=torch.bool, device=dev)
    no_fix_s = torch.zeros(n, nd, device=dev)
    pair_i, pair_j = torch.triu_indices(nd, nd, 1, device=dev)

    episodes = breaches = intercepted = attacker_slots = wins = 0
    defenders_lost = teamkills = 0.0
    intercept_times: list[float] = []
    alive_prev = raw._attacker_alive.clone()
    step_in_episode = torch.zeros(n, device=dev)
    ep_times: list[list[float]] = [[] for _ in range(n)]
    m = dict(def_steps=0, visible=0, fixed=0, lost=0, downed=0, beyond_leash=0, pair_steps=0, in_sphere=0, collide=0, att_contact=0)
    max_lost_s, min_sep, max_radius = 0.0, float("inf"), 0.0
    step = 0

    while episodes < args_cli.episodes:
        d_pos, d_vel, d_quat, d_angvel, a_pos, a_vel = states()
        alive = raw._attacker_alive
        if step % hold == 0:
            if args_cli.pilot == "zone":
                vel, yaw_rate = pilot.step(d_pos, d_vel, d_quat, a_pos, a_vel, alive, has_fix, def_alive())
                act = torch.cat([vel / raw.cfg.command.max_lin_speed,
                                 (yaw_rate / raw.cfg.command.max_yaw_rate).unsqueeze(-1)], -1)
            elif args_cli.pilot == "chase":
                rel = a_pos.unsqueeze(1) - d_pos.unsqueeze(2)  # (N, D, A, 3)
                dist = torch.linalg.norm(rel, dim=-1).masked_fill(~alive.unsqueeze(1), float("inf"))
                tgt = rel.gather(2, dist.argmin(-1).view(n, nd, 1, 1).expand(-1, -1, 1, 3)).squeeze(2)
                act = torch.cat([tgt / torch.linalg.norm(tgt, dim=-1, keepdim=True).clamp(min=1e-6),
                                 torch.zeros(n, nd, 1, device=dev)], -1)
            else:
                act = torch.zeros(n, nd, 4, device=dev)
            actions = {a: act[:, i].clamp(-1.0, 1.0) for i, a in enumerate(agents)}

        env.step(actions)
        step += 1
        step_in_episode += 1

        # -- metrics on the post-step state ------------------------------------------
        d_pos, d_vel, d_quat, d_angvel, a_post, _ = states()
        rot = matrix_from_quat(d_quat.reshape(-1, 4))
        cam_pos = d_pos.reshape(-1, 3) + rot @ torch.tensor(sensor.camera_offset, device=dev)
        cam_rot = camera_axes(rot, sensor.camera_pitch)
        visible, _ = detectable(sensor, cam_pos, cam_rot, *tags)
        _, fix, _ = sighting(sensor, cam_pos, cam_rot, *tags, noise=True,
                             speed=torch.linalg.norm(d_vel.reshape(-1, 3), dim=-1), yaw_rate=d_angvel.reshape(-1, 3)[:, 2])
        has_fix = fix.view(n, nd)
        no_fix_s = torch.where(has_fix, torch.zeros_like(no_fix_s), no_fix_s + raw.step_dt)
        # a crashed defender (on the floor, or tumbled past 60 deg) is not counted below
        upright = 1.0 - 2.0 * (d_quat[..., 1] ** 2 + d_quat[..., 2] ** 2)  # body z . world z
        in_play = def_alive()  # defenders downed by the rules land: neither crashed nor counted below
        up = (d_pos[..., 2] > 0.3) & (upright > 0.5) & in_play
        hit_att = torch.linalg.norm(d_pos.unsqueeze(2) - a_post.unsqueeze(1), dim=-1).min(-1).values < COLLISION_M
        m["att_contact"] += int((hit_att & up).sum())
        m["def_steps"] += int(up.sum())
        m["downed"] += int((~up & in_play).sum())
        m["visible"] += int((visible.any(-1).view(n, nd) & up).sum())
        m["fixed"] += int((has_fix & up).sum())
        m["lost"] += int(((no_fix_s > LOST_AFTER_S) & up).sum())
        max_lost_s = max(max_lost_s, float((no_fix_s * up).max()))
        r = torch.linalg.norm(d_pos[..., :2], dim=-1)
        m["beyond_leash"] += int(((r > pcfg.leash_radius + 0.3) & up).sum())
        max_radius = max(max_radius, float((r * up).max()))
        sep = torch.linalg.norm(d_pos[:, pair_i] - d_pos[:, pair_j], dim=-1)  # (N, pairs)
        both = up[:, pair_i] & up[:, pair_j]
        m["pair_steps"] += int(both.sum())
        m["in_sphere"] += int(((sep < 2.0 * pcfg.sphere_radius) & both).sum())
        m["collide"] += int(((sep < COLLISION_M) & both).sum())
        if both.any():
            min_sep = min(min_sep, float(sep[both].min()))

        newly_down = alive_prev & ~raw._attacker_alive
        for row in newly_down.any(dim=-1).nonzero(as_tuple=False).flatten().tolist():
            ep_times[row].append(step_in_episode[row].item() * raw.step_dt)
        alive_prev = raw._attacker_alive.clone()

        finished = raw.outcome_valid.nonzero(as_tuple=False).flatten()
        if len(finished):
            episodes += len(finished)
            breaches += int(raw.outcome_breached[finished].sum())
            intercepted += int(raw.outcome_captures[finished].sum())
            wins += int(((raw.outcome_captures[finished] >= na) & ~raw.outcome_breached[finished]).sum())
            if hasattr(raw, "outcome_defenders_lost"):
                defenders_lost += float(raw.outcome_defenders_lost[finished].sum())
                teamkills += float(raw.outcome_teamkills[finished].sum())
            attacker_slots += len(finished) * na
            for row in finished.tolist():
                intercept_times += ep_times[row]
                ep_times[row] = []
            step_in_episode[finished] = 0.0
            pilot.reset(finished)
            no_fix_s[finished] = 0.0
            has_fix[finished] = True

    pct = lambda a, b: 100.0 * a / max(b, 1)  # noqa: E731
    print(f"\n=== {args_cli.pilot} on {args_cli.task} ===", flush=True)
    print(f"episodes:              {episodes}", flush=True)
    print(f"breach rate:           {pct(breaches, episodes):.1f}% of episodes", flush=True)
    if not raw.cfg.respawn_waves:
        print(f"won (all {na} down):     {pct(wins, episodes):.1f}%; held to timeout {pct(episodes - wins - breaches, episodes):.1f}%", flush=True)
    print(f"attackers downed:      {intercepted / max(episodes, 1):.2f} per episode ({na} per wave)", flush=True)
    if getattr(raw.cfg, "mutual_kill", False):
        print(f"defenders lost:        {defenders_lost / max(episodes, 1):.2f} per episode of {nd} "
              f"({teamkills / max(episodes, 1):.2f} to teammate collisions)", flush=True)
    if intercept_times:
        print(f"time to intercept:     {sum(intercept_times) / len(intercept_times):.1f} s (mean of {len(intercept_times)})", flush=True)
    print(f"tag decodable:         {pct(m['visible'], m['def_steps']):.1f}% of defender time", flush=True)
    print(f"fix this frame:        {pct(m['fixed'], m['def_steps']):.1f}% (dropout, view angle, blur)", flush=True)
    print(f"lost (> {LOST_AFTER_S} s no fix):  {pct(m['lost'], m['def_steps']):.2f}% of defender time, longest {max_lost_s:.2f} s", flush=True)
    print(f"defenders crashed:     {pct(m['downed'], m['downed'] + m['def_steps']):.2f}% of defender time (floor or tumbled)", flush=True)
    print(f"attacker contact:      {m['att_contact']} defender-steps within {COLLISION_M} m of an attacker", flush=True)
    print(f"closest defenders:     {min_sep:.2f} m; inside a sphere ({2 * pcfg.sphere_radius:.1f} m) "
          f"{pct(m['in_sphere'], m['pair_steps']):.2f}% of pair time; collisions {m['collide']} steps", flush=True)
    print(f"furthest from site:    {max_radius:.2f} m; past leash+0.3 m {pct(m['beyond_leash'], m['def_steps']):.2f}% of defender time", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
