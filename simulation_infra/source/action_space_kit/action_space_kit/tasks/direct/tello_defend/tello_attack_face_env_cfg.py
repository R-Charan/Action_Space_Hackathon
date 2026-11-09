"""Train attackers against the bus defender: the adversary that hunts for holes in the defence.

Same face, arena, drones, starts and mutual-kill rules as ``AS-Defend-Face-v0``, but the agents
are the two attackers (one central policy flies both, agent cfg ``central: true``) and the three
defenders are flown by ``bus_pilot.BusPilot`` on ``AttackerTracker`` telemetry, exactly as
``scripts/evaluate_face.py`` scores it.

Reward (team, summed over the two attackers by the central wrapper):
  outcomes   +rew_breach once on a breach; per drone downed: +rew_defender_down, -rew_attacker_down
             (and -rew_out more if it left play); -rew_timeout if the defence holds to the end
  shaping    potential-based, so it can't change which policy is best (Ng et al. 1999):
             Phi = -rew_face_dist * face distance - rew_margin * clamp(defence margin), summed
             over the live attackers; 0 for a downed attacker and at the end of the episode
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .tello_defend_face_env_cfg import NUM_ATTACKERS, NUM_DEFENDERS, TelloDefendFaceEnvCfg

# obs per attacker: own (pos rel. face centre 3, vel 3, alive 1), partner (rel pos 3, rel vel 3,
# alive 1), each defender (rel pos 3, rel vel 3, alive 1), own face distance 1, time left 1
_OBS_DIM = 7 + 7 + 7 * NUM_DEFENDERS + 2
_DECIMATION = 8  # 120 Hz physics -> 15 Hz policy, about the real 13.5 Hz rc rate


@configclass
class TelloAttackFaceEnvCfg(TelloDefendFaceEnvCfg):
    possible_agents = [f"attacker_{i}" for i in range(NUM_ATTACKERS)]
    # room-frame velocity command [-1, 1] -> +-command.max_lin_speed (1.5 m/s) per axis, no yaw
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}
    decimation = _DECIMATION
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=_DECIMATION)

    # the learned attackers fly themselves: no scripted behaviour, no separation helper
    bait_prob = 0.0
    attacker_separation = 0.0
    speed_cap_norm = True
    """Scale a command longer than 1 back to 1 (the drones' top speed is 1.5 m/s in any direction)."""

    # - the defence (overrides: BusPilotCfg / TelemetryCfg fields, as evaluate_face --set)
    pilot_hz = 13.5
    pilot_set: dict = {}
    telemetry_set: dict = {}

    # - rewards
    rew_breach = 10.0
    rew_defender_down = 1.0
    rew_attacker_down = 1.0
    rew_out = 1.0
    rew_timeout = 2.0
    rew_face_dist = 0.5  # [1/m]
    rew_margin = 0.3  # [1/m]
    margin_clip = (-1.0, 3.0)  # [m]


@configclass
class TelloAttackFaceRoomEnvCfg(TelloAttackFaceEnvCfg):
    """The attack task in the final arena (TelloDefendFaceRoomEnvCfg)."""

    # the final arena: the real room from map.yaml, breach opposite tag 10, attackers launch beside
    # tag 10, defenders take off from three floor markers 0.8 m in front of the face
    room_map = True
    defender_takeoff = True
    defender_start_x = 0.8
    attacker_start_z = [1.0, 2.0]
