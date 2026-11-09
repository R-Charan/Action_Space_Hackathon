# Handoff: defense work (3 vs 2), moving to the fork

Written 2026-10-25 at the end of the work session in the upstream clone
(`~/Personal/College/WPI/Competitions/Action_Space`, remote jaromero3rd/Action_Space, which we can't
push to). Work continues in the fork `Action_Space_Hackathon` (remote R-Charan/Action_Space_Hackathon).
**Status:** the port is done. It's in the fork's working tree on `main`, uncommitted. Upstream's
newer umbrella (telemetry hub) has our `defend` hook re-applied on top. Fork checks pass:
`--selftest`, and `swarm_policies.py --run defend:6,7,9 --fly --defend-args "--mock --headless"`.

Start a new session there by reading this file, `simulation_infra/DEFENSE.md` and the "defend" part of
`drone_deployment_infra/swarm_ops/README.md`.

## Where things stand

- **Deliverable:** `drone_deployment_infra/swarm_ops/swarm_defend.py`, one file for the organizers.
  It runs as its own process under their umbrella (`swarm_policies.py --run defend:6,7,9 ...`), like
  their `formation` attackers. It contains the sim's classical bus defender (baked in by
  `build_defend.py`) plus `swarm_formation.py`'s link, video and tag localization. Engagement is
  **chase**: every attacker gets a defender from the first tick. The spare holds the face. Top speed
  0.89 m/s. It prints an event log and an end-of-run summary.
- **Verified without drones:**
  - selftest; pyflakes clean.
  - bench (breach 0.0-6.7 %).
  - 45 s mock flight: no stall.
  - Umbrella mock run: 3/3 PASS, FLY, Ctrl+C lands all.
  - A clean copy of `drone_deployment_infra` without `simulation_infra`.
  - Faulty, silent and missing attacker feeds.
  - Not yet flown on real drones with this version.
- **Sim side:** see `simulation_infra/DEFENSE.md` (tasks, results, commands).
- **Nothing is committed.** The user commits only when they ask.

## Decisions the user made

- **Classical defender for the real run.** PPO is used for attackers, and for a later "PPO defender"
  development stream.
- **Chase, full stop.** "The attackers don't move away, even if they move: chase". Don't reintroduce
  wait-for-it zones or anti-bait logic unless asked.
- **Speed cap 0.89 m/s**, the highest the drones reached on the rc model.
- **Separate process under the organizers' umbrella.** We hand them one python file.
- **Later:** branches in the fork for development streams (classical defender, PPO defender, ...).

## First real run on the organizers' host (2026-10-24, log `logs_swarm_defend.txt` at the repo root)

The defenders never flew (`defending 0.0 s`):
- **Takeoff judged failed, so we landed everything at t = 2.1 s.**
  - drone4 replied `error`, and was at 10 cm one second later.
  - drone5's reply was `82`, a late `battery?` answer, and it was at 0 cm one second later.
  - drone3's reply timed out after 20 s.
  - Our check (one try, then height >= 30 cm after 1 s) is too strict.
  - **Fix:** copy formation's newer takeoff (`swarm_formation.py` around line 1215): 2 tries,
    `height?` on a lost reply, airborne from the status stream (h >= 60 cm for 0.5 s). Don't send
    `land` while a takeoff reply is still pending.
  - **DONE 2026-11-04.** The takeoff now matches formation's. On a failure the land-all waits for
    pending takeoff replies (airborne drones hover meanwhile), and no retry goes out after landing
    starts. Mock-tested with `--mock --fake-feed --fly`: normal, a lost reply (airborne -> ok),
    and a real failure plus a 6 s slow reply (waited, then landed all). Logs:
    `outputs/to_mock_{ok,lost,fail}.log`.
- **The attacker feed arrived**, sent by their wrapper (group `swarmdefend`) in our JSON format on
  localhost. But formation's telemetry `p` is in the frame of its `--target` tag (that run: tag 6,
  "map re-expressed around it"), not tag 10's.
  - Read as tag-10 coordinates, the attacker sat at world (4.16, -1.34, -1.49), under the floor.
  - Decoded from the tag-6 frame with formation's map code, it sat at (4.75, -1.95, 0.41).
  - **Fix:** read `formation/drone<N>` telemetry directly and convert from the target tag's frame
    (`--feed-origin-tag`), or get the organizers to publish in tag 10's frame. Ask them which
    `--target` they fly.
- **Their map is off.** Formation's map check rejected tags 7, 23, 12, 2, 11 and 1 (0.2-2.6 m off
  their mapped spots). The same check drops them for our defenders, which leaves fewer tags.
- **Small bug:** an attacker's first appearance prints `feed RESUMED` instead of `LIVE`, because
  `prev_fresh` is updated before the attacker is known. **DONE 2026-11-04.**

## Borrowed from the teammate's PPO repo (2026-11-04)

The teammate repo is `../RL-based-drone-interceptors` (N-Raghav). It's read-only for us: never edit it.
`swarm_defend.py` stays the deliverable.
- **Taken: evasive attackers for the bench.** `PatternAttackers` is a numpy port of his attacker
  scripts, with the same ranges. Each attacker gets one of straight / swerve / arc / feint / weave /
  jink, plus an altitude sinusoid, dodging, a speed burst (capped at our 1.5 m/s rules ceiling) and
  a 0-4 s launch delay. Run `swarm_defend.py --bench 200 --patterns [--pattern feint]`. The
  default bench and the selftest are unchanged.
- **Results at 1.0-1.5 m/s, 200 episodes** (`outputs/bench_patterns.log`):

  | Attackers | Breach |
  |---|---|
  | Straight | 5.5 % |
  | Evasive mix | 14.5 % |
  | Straight + modifiers | 11.5 % |
  | Swerve | 11.5 % |
  | Weave | 13.5 % |
  | Arc | 18.5 % |
  | Jink | 20.0 % |
  | **Feint** | **27.5 %** |

  Feint, then jink, is the weak spot to tune against.
- **Not taken, already covered or better in ours:**
  - the feed parser: the same JSON format, and ours already takes the list form and `alive`;
  - the map frames (`swarm_formation`'s code);
  - lost-fix hold, height from the range sensor, video delay;
  - the safety layer: our spheres brake on closing speed, he uses plain repulsion.
- **Not taken, conflicts with a decision:**
  - his zones/jobs planner (we chase);
  - the PPO network (classical for the real run);
  - turning to face the flight direction (it loses the tags; his real test failed);
  - the drone_gui adapter (we run under the umbrella).

## Next steps, in order

1. ~~**Port to the fork.**~~ Done (see Status). Copy our new files. Re-apply our small `defend` hook onto the fork's newer
   umbrella files instead of overwriting them:
   - `swarm_ops/swarm_policies.py`: the `defend` branch in `parse_runs`, `--defend-args`, help text.
   - `swarm_ops/policies/__init__.py`: register `DefendPolicy`.
   - `swarm_ops/README.md`: the defend section.

   The fork = upstream `origin/main` = our base `cf62a13` + 3 commits:
   - `d0ef8d2`: telemetry.
   - `0ca7536`: bundle-adjusted map.
   - `cd6536a`: stepping-stone tags 12, 22-25, plus keepout and A* path planning in formation.

   Our local `map.yaml` / `tags.yaml` already equal the fork's.
2. **Attacker feed = umbrella telemetry.** `swarm_defend.py` reads JSON on UDP 9200 (a guessed
   format), and nobody sends that.
   - The organizers' umbrella publishes `formation/drone<N>`: `p` in the map frame (`None` before the
     first fix), `psi_deg`, `fix_age`, `state`, `height_cm`, `rc`. It travels over localhost UDP via
     `policies/telemetry.py`. External processes get `SWARM_TELEM`, and read it with
     `TelemetryClient.from_env()` / `.since("formation/", max_age=...)`.
   - There's no alive flag. Infer "down" from a stale topic, or from `height_cm` near the floor.
   - Keep UDP 9200 as a fallback.
   - Test under the real umbrella with formation in mock mode, if possible.
3. **`--ignore-tags` default `1,4,7,22,24`.** It came from the 10-24 map, where 22 and 24 were
   "floor tags mapped face-down". In the new map, 22-25 are stepping-stone tags. Re-check against the
   new `map.yaml`.
4. **Keepout around tag stands.** Formation now has `--keepout` and A* path planning. If the
   stepping-stone tags stand in the flying area, the chase needs a keepout too. Open question to the
   user: are 22-25 on stands in the arena, or on walls?
5. **Possible tuning.** In mock runs, the chasers swap targets every 2-5 s. Raise `switch_margin` /
   `min_hold_s` if that shows up in real flights.
6. **Later:** fork branches per development stream.

## Files that are ours (port these)

| File | What |
|---|---|
| `drone_deployment_infra/swarm_ops/swarm_defend.py` | the deliverable |
| `drone_deployment_infra/swarm_ops/build_defend.py` | bakes the sim planner into it |
| `drone_deployment_infra/swarm_ops/policies/defend.py` | umbrella hook (launches swarm_defend.py) |
| `drone_deployment_infra/swarm_ops/{swarm_policies.py,policies/__init__.py,README.md,requirements.txt}` | hook edits; re-apply onto the fork's versions |
| `drone_deployment_infra/core/classical_defender.py`, `drone_deployment_infra/defender_classical_policy.patch` | the bus behind the teammate PPO runner's 59 -> 4 interface |
| `drone_deployment_infra/README.md`, `README.md`, `CLAUDE.md` | doc pointers |
| `simulation_infra/DEFENSE.md` | defense guide |
| `simulation_infra/scripts/{evaluate_face,record_face,evaluate_zone,record_zone}.py` | score / record |
| `simulation_infra/scripts/{evaluate.py,record_clip.py}` | fixes: skip downed attackers, render without stepping physics |
| `simulation_infra/source/.../tello_defend/`: `__init__.py`, `tello_defend_env.py` (force-frame fix), `bus_pilot.py`, `classical_policy.py`, `face_geometry.py`, `room_layout.py`, `zone_pilot.py`, `tello_defend_3v2_env*.py`, `tello_defend_face_env*.py`, `tello_attack_face_env*.py`, `agents/sb3_ppo_attack_cfg.yaml` | sim tasks + defender |

**Not ours, don't port by default:**
- `defend_policy_addon.zip`, `simulation_infra/scripts/evaluate_raghav.py`: the teammate's PPO
  defender add-on.
- `swarm_ops/policies/defender.py`, `policies/defender_trial.py`: the teammate's earlier wrapper.
- The `*.pt` checkpoints at the repo root (`scratch_attr_s1.pt`, `shared_room_best.pt`,
  `shared_stage3_best.pt`). They belong to a future PPO-defender branch, if anywhere.
- `simulation_infra/outputs/` (gitignored): logs and videos, including the PPO attacker checkpoints
  in `outputs/sf_attack/`.

## Environment notes

- **Isaac Sim** runs in distrobox `ros2` on an RTX 3060 Laptop with 6 GB. Run one Isaac process at a
  time.
- **Python with torch:** `~/isaac/env_isaaclab/bin/python`. The system `python3` has no torch.
- **Shell setup for sim commands:** run them from `simulation_infra/`, after the CLAUDE.md Step 2
  line.
- **Logging:** run with `python -u`, log to `outputs/`, and never write to `/tmp` or `/mnt/data`.
- **Others' worktrees:** don't edit the teammate's worktree (`worktree-defend-policy-eval`).
- **Other sessions:** another Claude session also edited `swarm_ops` here, adding the first
  `defend` registration and `--defend-args`.
