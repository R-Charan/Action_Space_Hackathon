# swarm_ops: Tello swarm commands

All commands run from `drone_deployment_infra/swarm_ops/`. Config lives in `../config/`
(`dongles.json`, `map.yaml`, `tags.yaml`, `camera/tello_<n>.yaml`).

```
cd ~/Action_Space/drone_deployment_infra/swarm_ops
source <your venv>/bin/activate
```

---

## 1. One-time setup (per laptop)

```
pip install -r requirements.txt
echo "net.ipv4.conf.all.rp_filter=2" | sudo tee /etc/sysctl.d/99-tello.conf
sudo sysctl --system
```
- Plug the USB Wi-Fi adapters into a **powered hub** (adapter names in `dongles.json`
  belong to those exact dongles).
- Create every drone's Wi-Fi profile from `dongles.json` (drones can be OFF):
```
python connect.py --create --all
```
  Each profile is locked to its adapter + its drone's radio (BSSID), auto-connects,
  uses static IP `192.168.10.10<n>`, 2.4 GHz only.

**New drone?** Add its entry in `../config/dongles.json` (`iface`, `drone`, `ssid`, `bssid`),
then `python connect.py --create <n>`.
Find a drone's BSSID: `nmcli -f SSID,BSSID dev wifi list | grep TELLO`.

---

## 2. Connect drones (every session)

Turn the drones on, wait ~20-30 s (they join by themselves), then check:
```
python connect.py 4 5 6 7 9          # verify; connects any that aren't up yet
python connect.py --all              # every drone in dongles.json
python connect.py 4 5 6 7 9 --wait 60   # run FIRST, then power on: each joins as it appears
python connect.py 4 5 6 7 9 --down   # disconnect
```
Every drone must say `OK`. `NOT DETECTED` = adapter unplugged / hub unpowered.

Tip: chain it so a flight only starts when all drones are ready:
```
python connect.py 4 5 6 7 9 && python swarm_formation.py --only 4,5,6,7,9 --fly --fast --headless
```

---

## 3. Swarm formation (`swarm_formation.py`)

Drones take off, find tag 10 (or any mapped tag), fly to slots on an arc in front of
tag 10, hold, land. **Without `--fly` it is a dry-run** (no takeoff, no motion).

| Mode | Command |
|---|---|
| Dry-run (props off, check tags/video) | `python swarm_formation.py --only 4,5,6` |
| Normal (slow, careful) | `python swarm_formation.py --only 4,5,6 --fly` |
| Fast | `python swarm_formation.py --only 4,5,6 --fly --fast` |
| Headless (no windows, terminal only) | `python swarm_formation.py --only 4,5,6 --fly --fast --headless` |
| Flip wave (back flip one by one) | `python swarm_formation.py --only 4,5,6 --fly --fast --flip` |
| Recommended | `python swarm_formation.py --only 4,5,6 --fly --fast --headless --rest-radius 0.15 --hold 1` |

**Options**
| Flag | Meaning |
|---|---|
| `--only 4,5,6` | which drones |
| `--fly` | actually fly (without it: dry-run) |
| `--fast` | faster search/approach/turns (same final precision) |
| `--headless` | no windows; keys in terminal: `q`+Enter land all, `e`+Enter EMERGENCY |
| `--no-video` | no camera windows, only the top-view map |
| `--display-hz 5` | refresh windows less often (saves CPU) |
| `--rest-radius 0.15` | slot "close enough" zone in m (default 0.08); inside it the drone just holds |
| `--hold 1` | seconds to hold the formation before landing (default 5) |
| `--flip` | back flip wave once in formation (needs battery >= 55 %, ~1 m free behind each slot) |
| `--radius 1.3` | slot distance from tag 10 in m (don't go below ~1.2) |
| `--slot-offset 0` | slot height vs tag 10 center, cm |
| `--no-map` | use tag 10 only (ignore map.yaml) |
| `--ignore-tags 6,7` | skip specific mapped tags |

**Keys** (window mode): `q` = land all, `SPACE` = EMERGENCY motor stop. `Ctrl+C` = land all.

**Placement:** drones 2-3 m in front of tag 10 (within +-60 deg), 1 m+ apart,
**facing tag 10** (search becomes instant). More drones -> the arc moves further out
(5 drones ~1.5 m, 6 drones ~2 m from tag 10).

**Logs:** `logs/formation_<date>_<time>.csv` (state, position, commands, battery per drone).

---

## 4. Multi hover (`multi_hover.py`, no video)

All selected drones take off together, hover, land. Good for testing many drones.
```
python multi_hover.py --config ../config/dongles.json --only 1,2,3,4,5,8,9 --hover 10
```
All-or-nothing preflight (one failure -> nobody takes off). `Ctrl+C` lands all.
Drones ~1.5 m apart.

---

## 5. Attack + defense, scripted policies (`swarm_policies.py`, umbrella)

Runs one or more policies at once, each on its own drones: `--run <policy>:<drones>`,
repeatable. **Without `--fly` it is a dry-run.** One FLY prompt for everyone,
all-or-nothing preflight (formation drones included), `Ctrl+C` lands all.
Policies live one per file in `policies/` (add one there and register it in `policies/__init__.py`).

| Policy | What it does |
|---|---|
| `spin` | defense dummy: take off, hold the takeoff height, spin in place (`--yaw`, default 40) |
| `levels` | lower-numbered half at 3 ft (91 cm), the rest at 6 ft (183 cm), hover |
| `hover` | hold the takeoff height, or `--height <cm>` (40-250) |
| `formation` | `swarm_formation.py` on those drones, as its own process |
| `defend` | `swarm_defend.py` on those drones, as its own process: our 3v2 defenders (the sim's classical bus), own video + tag fixes, attackers from the UDP feed; chases from the first tick (`simulation_infra/DEFENSE.md`) |

```
python swarm_policies.py --run spin:6,7,9 --run formation:4,5           # dry-run
python swarm_policies.py --run spin:6,7,9 --run formation:4,5 --fly     # defense spins, attack forms up
python swarm_policies.py --run hover:4,5,6,7,9 --height 200 --fly --duration 60   # photo hover at 2 m
python swarm_policies.py --run levels:4,5,6,7,9 --fly
python swarm_policies.py --run defend:6,7,9 --run formation:4,5 --fly --defend-args "--feed-port 9200"
```

**Options**
| Flag | Meaning |
|---|---|
| `--run spin:6,7,9` | a policy and its drones (repeat per group; a drone in one group only) |
| `--fly` | actually fly (without it: dry-run, 15 s) |
| `--duration 60` | land all after this many s (default 20; with `formation`: until it lands) |
| `--height 200` | `hover` height above the floor, cm |
| `--yaw -60` | `spin` speed and direction (-100..100) |
| `--formation-args "--fast --hold 15"` | passed to `swarm_formation.py` |
| `--defend-args "--feed-port 9200 ..."` | passed to `swarm_defend.py`: `--feed-port` (9200), `--feed-frame map\|world`, `--attackers A1,A2` (feed names; default: as first seen), face geometry `--face-x 4.2 --face-width 3 --face-height 2 --face-bottom 0.5`, `--max-speed` (0.89 m/s), `--wall-gap` (0.2 m), `--alert-radius`/`--engage-radius`/`--strike-radius` (50 m: chase everywhere), `--start-height` (130 cm), `--lost-land` (6 s), `--headless` |

With `formation`, its camera windows and keys (`q`, `SPACE`) still work, but its
`--headless` terminal keys don't (its stdin is the umbrella): use `Ctrl+C` in the
umbrella's terminal. Formation drones only avoid each other: keep the other groups out
of the space in front of tag 10. Safety: a drone with no status stream for 3 s or
battery < 15 % lands on its own; takeoff needs battery >= 40 %.

**Telemetry between policies** (`policies/telemetry.py`). Any policy can publish live data
and read what the others publish; only the latest value per topic is kept, with its age.
```python
self.bus.publish("spin/drone6", {"yaw_rc": 40})        # in step() or anywhere
hit = self.bus.latest("formation/drone4", max_age=0.5)  # (data, age_s) or None if missing/stale
self.bus.since("formation/")                            # {topic: (data, age_s)}
```
In-process policies get the umbrella's bus as `self.bus`. A policy in its own process uses
`TelemetryClient.from_env()` (same calls, over localhost UDP to the umbrella; a private bus
when run alone). Published automatically every tick:
| Topic | From | Data |
|---|---|---|
| `status/drone<N>` | umbrella, its own drones | `policy`, `height_cm`, `battery`, `rc` |
| `formation/drone<N>` | `swarm_formation.py` | `state`, `p` (m, map frame; `None` = no fix yet), `psi_deg`, `fix_age`, `height_cm`, `rc` |

Steer on telemetry only while it is fresh (`max_age`): a stalled publisher must not steer a drone.

With `defend`: it needs no drone_gui (it owns its drones' link, video and tags, like
`formation`). Each defender climbs to `--start-height`, turns until it sees a mapped tag,
then the planner (the sim's classical bus, baked in by `build_defend.py`) takes over and
chases: every attacker in the feed gets one defender, which flies straight at it (head-on
intercept) from the first tick, wherever it is in the room, out to 0.2 m from the tag-10 wall
(`--wall-gap`); the spare drone stays in front of the face and takes over when a chaser is
lost. (Zones are still there for tuning: `--alert-radius`, `--engage-radius`,
`--strike-radius`, all 50 m = the whole room by default.) A defender without a fresh
fix (< 0.4 s) only turns back to the tags; with none for `--lost-land` (6 s) it lands and is
out, and so is one found on the floor. Attackers not yet in the feed are left out (defenders
hold posts 0.8 m in front of the face). Top speed `--max-speed` 0.89 m/s. Once a second it prints, per drone, its
state, role, position, fix age and rc, plus each attacker's feed state and zone: read these
first when a defender doesn't engage. Log: `logs/defend_<time>.jsonl` (every tick: each
defender's state, role, pose, fix age, rc and target; the attackers' feed state and zones).

Every change is also printed as one `[event t=...]` line (t = s since FLY), and the run ends
with a `SUMMARY` block (per drone: last state and role, how it ended, top speed; per attacker:
down / alive / never seen):
| Event | Meaning |
|---|---|
| `[feed] FIRST PACKET from ...: b'...'` | the raw first datagram: check the organizers' format here |
| `[feed] BAD PACKET #n: ...` | not JSON / no `attackers` / no `p` (first 5 printed) |
| `[feed] attacker 'X' -> A0`, `... IGNORED` | feed names mapped to slots; extras beyond `--num-attackers` |
| `A0: feed LIVE / STALE / RESUMED` | no update for 1 s = stale: its position is held |
| `WARNING: no attacker packets on UDP :9200 ...` | 3 s into defending and nothing heard: defenders only hold posts |
| `drone6: takeoff ok`, `CLIMB -> SEARCH (...)`, `no trusted tag in view -> turning` | startup |
| `drone6: LOST TAGS` / `tags back after 1.2 s` / `LOST for 6.0 s ... LANDING, OUT` | localization |
| `planner ON`, `drone6: role spare -> STRIKE A0: A0 at (...), 2.3 m away` | who chases whom |
| `A1: DOWN at (...), nearest defender drone9 0.30 m` | a capture (or the feed's alive=false) |

**Attacker feed** (`swarm_defend.py --feed-port`, default 9200): JSON datagrams
`{"attackers": {"TELLO-3": {"p": [x, y, z], "alive": true, "age": 0.05}, ...}}` (or a list of
`{"name", "p", "alive"}`), `p` in the map frame (`--feed-frame world`: X out of the tag-10 wall,
Y along it, Z up). In the 2026-10-24 host run the organizers' wrapper sent this format from
formation's telemetry, but formation's `p` is in its **`--target` tag's frame** (that run: tag 6),
not tag 10's, so positions were off by metres (attacker 1.5 m under the floor). To fix next: read
`formation/drone<N>` telemetry directly and convert from the target tag's frame (or have formation
publish in tag 10's frame).

Check without drones:
```
python swarm_defend.py --selftest                       # baked planner == sim, frames, rc, feed, roles; pass: SELFTEST PASSED
python swarm_defend.py --bench 200 --att-speed 1.0 1.5  # closed loop on the rc model; --loiter 10 = attackers hover first
python swarm_defend.py --only 6,7,9 --mock --fly --headless --duration 30   # simulated drones + scripted feed; type FLY
python swarm_policies.py --run defend:6,7,9 --fly --defend-args "--mock --headless"   # the same through the umbrella
python swarm_defend.py --only 6,7,9 --fake-feed         # real drones, dry run (no takeoff), scripted attackers
```
After changing `simulation_infra`'s `bus_pilot.py`, `face_geometry.py`, `zone_pilot.py` or
`tags/sensing.py`: `python build_defend.py`, then `--selftest`.

---

## 6. Map the tags (`map_tags.py`, no flying)

Needed whenever tags are added or moved. Hold the drone (props off) so it sees
tag 10 together with each other tag until all are green, then press `q` to save.
```
python connect.py 4
python map_tags.py --drone 4 --tags-list 10,6,7
```
Every tag needs its size in `../config/tags.yaml` (`sizes_mm`, black square in mm).
Check: dry-run shows no `MAP CHECK` warnings.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `NOT DETECTED` / `OSError: No such device` | adapter unplugged or hub unpowered: replug, `nmcli dev status \| grep wlx` |
| `drone not broadcasting` | drone off / asleep / flat battery: power it on |
| Drones switch off while connecting | idle auto-off / heat: power on just before flying, connect right away |
| Takeoff failed (height 0) | low battery, hot, tilted surface, or IMU calibration needed (Tello app) |
| Pixelated video with 5+ drones | 2.4 GHz crowding: power drones on ~10 s apart, spread adapters, EDU video lower (see `EDU_MODIF_README.md`) |
| `Map: tags [...]` doesn't match the wall | re-map with `map_tags.py` |
| Wi-Fi name `TELLO-TELLO-6` | in the Tello app type only `6` (it adds `TELLO-`) |
