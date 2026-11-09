# Defense: 3 defenders vs 2 attackers (sim + deployment)

The hackathon's defense round: **3 defender Tellos guard a breach face, 2 attacker Tellos try to fly
through it.** Drones are identical (same speed). Two drones closer than 0.35 m are both downed
(mutual kill), so an interception trades one defender for one attacker; with 3 vs 2 the spare
defender wins that trade. Defenders get each attacker's pose and alive flag from the organizers
(no velocity: we estimate it).

Our defender is **classical** ("park the bus"), not learned: it is robust to attack plans it has
never seen. Learning is used on the other side, to train PPO attackers that look for its holes.

| Piece | Where |
|---|---|
| Sim tasks (Isaac Lab) | `source/action_space_kit/action_space_kit/tasks/direct/tello_defend/` |
| The defender (pure torch, no Isaac) | `tello_defend/bus_pilot.py` (+ `face_geometry.py`, `zone_pilot.py`, `tags/sensing.py`) |
| Real-room geometry | `tello_defend/room_layout.py` (reads `drone_deployment_infra/config/map.yaml`) |
| Score / record | `scripts/evaluate_face.py`, `scripts/record_face.py` (older circular site: `evaluate_zone.py`, `record_zone.py`) |
| Real drones (the deliverable) | `drone_deployment_infra/swarm_ops/swarm_defend.py` (see its README section "defend") |
| Teammate PPO runner's interface | `tello_defend/classical_policy.py`, `drone_deployment_infra/core/classical_defender.py`, `drone_deployment_infra/defender_classical_policy.patch` |

Run every command from `simulation_infra/` with the CLAUDE.md Step 2 environment, `python -u`,
logging to `outputs/`.

## Frames

| Frame | Axes |
|---|---|
| Room / sim world (`tags.room_map`) | X out of the tag-10 wall, Y along it, Z up from the floor |
| Face-local (the face tasks, `bus_pilot`) | face on the plane x = 0, box behind it (x < 0), attackers come from +x. In the room: `x = face_x - X`, `y = face_y - Y`, `z = Z` (half a turn about Z) |
| Map (`map.yaml`, `swarm_formation` telemetry) | x right, y up, z out of tag 10; world = `[z, x, y + 1.22]` (tag 10 centre is 1.22 m up) |
| `swarm_formation` internal | x right, y down, z into the wall; world = `[-p2, p0, -p1 + 1.22]` |

Yaw `psi`: 0 = facing tag 10, positive = clockwise from above.

## Tasks

| Task id | What |
|---|---|
| `AS-Defend-3v2-v0`, `-Aggressive-v0` | first version: circular breach site, zone pilot (`zone_pilot.py`) |
| `AS-Defend-Face-v0`, `-Aggressive-v0` | breach **face** (3 x 2 m rectangle, 0.5 m off the floor), open arena |
| `AS-Defend-Face-Room-v0`, `-Room-Aggressive-v0` | **the final arena**: the real room from `map.yaml`; face at room X = 4.2 m (far end, opposite tag 10), side walls at Y = -2.0 / 3.2; defenders take off in front of the face facing tag 10; attackers launch 0.3-0.6 m from the tag-10 wall |
| `AS-Attack-Face-v0`, `AS-Attack-Face-Room-v0` | the same match with **PPO flying both attackers** (central SB3 policy, `agents/sb3_ppo_attack_cfg.yaml`) against the bus inside the env |

Scripted attackers: default = 0.8-1.5 m/s runs at the face that weave (0.3-1.0 m, 0.3-0.8 Hz),
half of them baiting (in, back out, then dive) and closing as a pair; aggressive = they also jink,
switch aim point and speed at random, and dodge nearby defenders. Every attacker command is capped
at 1.5 m/s overall.

## The bus defender (`bus_pilot.py`)

Equal speeds make this Isaacs' target-guarding game: a defender guarantees an attacker can't reach
the face while every point of the face is closer to the defender than to the attacker (the
**margin** > 0). Each tick (13.5 Hz, the real rc rate), batched in torch:

1. **Telemetry**: `AttackerTracker`, a constant-velocity Kalman filter per attacker, turns the
   delayed (0.2 s), noisy poses + alive flags into position and velocity.
2. **Zones** by distance to the face, with hysteresis: OUTSIDE / ALERT (`r_alert_in` / `_out`) /
   ENGAGE (`r_engage_in` / `_out`).
3. **Assignment**: every partial one-to-one assignment of live attackers to live defenders is scored
   by its worst (bottleneck) guarding margin; the best is kept unless a new one gains
   `switch_margin` after `min_hold_s`. Left-over defender = **spare**.
4. **Motion**: a guard moves to the candidate point that maximizes its margin (`face_grid`,
   `lookahead_s`); an engaged attacker within `strike_radius` of its guard is **struck** (head-on
   intercept, `strike_lead_s`); unassigned defenders **hold** posts 0.8 m in front of the face.
   Defenders stay in a box (`x_min` .. `x_max` = the leash, room walls), keep safety spheres between
   themselves, and do not brake for attackers (braking froze strikes short of contact).
5. **Tags**: heading keeps a mapped tag in the camera (`tags/sensing.py`, the real camera model); no
   fix for 0.4 s = lost: no translation, turn back to the tags.

Sim defaults: alert 4.0 m, engage 1.5 m, strike 1.3 m, leash 3.0 m. **The real-drone file overrides
these to "chase"** (below).

## Results (512 episodes each unless noted)

| Matchup (final arena, `-Room-`) | Breach |
|---|---|
| bus vs default scripted attackers | 3.9 % |
| bus vs aggressive scripted attackers | 12.7 % |
| bus vs PPO attackers (105 M steps, trained in the open arena) | 82.4 % |
| `classical59` (the bus through the teammate runner's 59 -> 4 interface) vs bus, aggressive | 16.6 % vs 16.8 % (parity) |

PPO attackers beat the bus by rushing side by side at full speed: guard margins fall from +0.9 m
(2.5 m out) to -0.3 m (1 m out) before the 1.3 m strike fires, one attacker trades itself for a
defender and the other scores. That is why the real-drone defender engages early (chase).

Teammate PPO defenders on the same envs (`AS-Defend-Role-v0`, from `defend_policy_addon.zip`):
`shared_stage3_best.pt` 17.3 % vs classical 4.7 % with attrition (the competition rule);
`scratch_attr_s1.pt` 95.5 % breach in the room. Videos: `outputs/videos/scratch_attr_s1_vs_{classical,stage3}.mp4`.

## Commands

```bash
# score (pilots: bus | classical59 | chase | zero)
python -u scripts/evaluate_face.py --task AS-Defend-Face-Room-v0 --pilot bus --episodes 512 > outputs/eval_room_bus.log 2>&1
python -u scripts/evaluate_face.py --task AS-Defend-Face-Room-Aggressive-v0 --pilot bus --episodes 512 > outputs/eval_room_aggr.log 2>&1
grep -E "episodes:|breached:" outputs/eval_room_bus.log
# tune: --set <BusPilotCfg field>=<v>  --env_set <env cfg field>=<v>  --telemetry_set <TelemetryCfg field>=<v>
python -u scripts/evaluate_face.py --task AS-Defend-Face-Room-v0 --set r_engage_in=3.0 r_engage_out=3.5 strike_radius=2.5 > outputs/eval_room_chase.log 2>&1

# record (always --out and --frame_dir); --keep breach keeps only breached episodes
python -u scripts/record_face.py --task AS-Defend-Face-Room-Aggressive-v0 --episodes 2 \
  --out outputs/videos/room_aggr.mp4 --frame_dir outputs/frames_room > outputs/record_room_aggr.log 2>&1
grep "\[rec\]" outputs/record_room_aggr.log

# PPO attackers: train against the bus, then score / record them
setsid nohup python -u scripts/sb3/train.py --task AS-Attack-Face-Room-v0 --headless --num_envs 4096 > outputs/train_attack.log 2>&1 &
python -u scripts/sb3/play.py --task AS-Attack-Face-Room-v0 --headless --num_envs 64 --episodes 512 \
  --checkpoint <run>/model.zip --report outputs/play_attack.json > outputs/play_attack.log 2>&1
python -u scripts/record_face.py --task AS-Attack-Face-Room-v0 --checkpoint <run>/model.zip \
  --out outputs/videos/ppo_vs_bus.mp4 --frame_dir outputs/frames_ppo > outputs/record_ppo.log 2>&1
```
Pass: `evaluate_face.py` ends with an `episodes:` / `breached:` block; `record_face.py` prints
`[rec] wrote <out> (... bytes)`. Delete the frame dir afterwards. One Isaac process at a time on a
6 GB GPU. Training throughput: ~3.3k fps at 256 envs on an RTX 3060; ~30 min to 105 M steps at 8192
envs on an H100.

Fixes made along the way (already in the tree): `tello_defend_env.py` rotates the controller's
world-frame force into the body frame (a yawed drone used to run away); `scripts/evaluate.py` and
`record_clip.py` skip downed attackers and render with `env.sim.render()` (a bare app update stepped
physics without controller forces).

## To the real drones: `swarm_defend.py`

`drone_deployment_infra/swarm_ops/swarm_defend.py` is the single file we hand the organizers. It
runs as its own process under their umbrella (`swarm_policies.py --run defend:6,7,9`), like their
`formation` attackers:

- **Planner**: `bus_pilot.py`, `face_geometry.py`, `zone_pilot.py` and `tags/sensing.py` are baked
  in verbatim by `swarm_ops/build_defend.py` (run it after changing any of them; `--selftest` fails
  while the baked copy and the sim differ). Needs `torch` (CPU) and `numpy`, no Isaac.
- **Localization / link / video**: `swarm_formation.py`'s drone class (tags, Kalman, preflight).
- **Engagement = chase**: alert / engage / strike radii 50 m (the whole room), leash to 0.2 m from
  the tag-10 wall, top speed 0.89 m/s (the most the rc model reaches). The sim zones held guards
  back from hovering attackers, which looked like a stall after ~20 s.
- **Checks without drones**: `--selftest`, `--bench 200` (closed loop on the fitted rc model; add
  `--att-speed 1.0 1.5`, `--loiter 10`), `--mock --fly` (simulated drones + scripted feed).

Bench (150 episodes, chase defaults): breach 0.0 % vs 0.6-1.0 m/s rushes, 0.7 % vs 0.89 m/s,
6.7 % vs 1.0-1.5 m/s (old sim zones: 4.7 / 5.3 / 24.7 %); attackers that hover forever: 1.88 of 2
caught (old zones: 0.02).

Open items: the attacker feed (it currently reads JSON on UDP 9200; the organizers' umbrella
publishes attacker poses as telemetry `formation/drone<N>`), the `--ignore-tags` default vs the new
stepping-stone tags, and a keepout around tag stands. See `swarm_ops/README.md`.
