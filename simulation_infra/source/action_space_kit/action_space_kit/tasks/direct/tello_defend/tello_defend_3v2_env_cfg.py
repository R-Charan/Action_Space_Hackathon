"""The organizers' format: 3 defenders vs 2 attackers, with attackers that may bait."""

from __future__ import annotations

import gymnasium as gym

from isaaclab.utils import configclass

from .tello_defend_env_cfg import TelloDefendEnvCfg

NUM_DEFENDERS = 3
NUM_ATTACKERS = 2
_OBS_DIM = 9 + 7 * NUM_ATTACKERS


@configclass
class TelloDefend3v2EnvCfg(TelloDefendEnvCfg):
    possible_agents = [f"defender_{i}" for i in range(NUM_DEFENDERS)]
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}

    num_defenders = NUM_DEFENDERS
    num_attackers = NUM_ATTACKERS
    defender_radius = 2.2  # [m] start ring; defenders start evenly spaced on it (random rotation)

    # - baiting attackers: fly in to bait_depth, turn back out past bait_retreat, then dive
    bait_prob = 0.5  # per attacker, per wave
    bait_depth = [2.8, 4.2]  # [m] from the site, horizontal
    bait_retreat = 5.5  # [m]

    # - the match: one wave of 2 attackers. Two drones within collision_radius of each other are
    #   both downed: an interception costs the defender too, and so does hitting a teammate.
    #   Win = both attackers down without a breach; a breach ends it as a loss.
    respawn_waves = False
    mutual_kill = True
    collision_radius = 0.35  # [m] centre to centre, any two drones (the capture radius)
    rew_scale_defender_lost = -15.0  # a defender downed by an interception (per defender)
    rew_scale_teamkill = -30.0  # extra, two defenders downing each other (each)

    # - a downed drone (intercepted, or in a collision) is sent `land`
    downed_descent_speed = 0.5  # [m/s], the Tello's land descent
    downed_floor_height = 0.15  # [m] below this it just holds (motors idle on the real drone)

    # - aggressive flight (off here; on in TelloDefend3v2AggressiveEnvCfg). Per attacker, per wave:
    aggressive = False
    weave_amp = [0.3, 1.0]  # sideways weave, as a fraction of the forward direction (1.0 = 45 deg)
    weave_freq = [0.3, 0.8]  # [Hz]
    jink_amp = 0.8  # sudden sideways swerves, drawn in [-jink_amp, jink_amp] ...
    jink_interval = [0.4, 1.2]  # [s] ... and held this long
    speed_interval = [0.5, 1.5]  # [s] between new speeds drawn from attacker_speed
    altitude_range = [0.8, 2.6]  # [m] target heights ...
    altitude_interval = [1.0, 3.0]  # [s] ... changed this often
    altitude_gain = 1.5  # [1/s]
    max_climb_rate = 0.6  # [m/s]
    dodge_gain = 0.0  # sideways swerve away from defenders inside dodge_radius (0 = off)
    dodge_radius = 1.5  # [m]


@configclass
class TelloDefend3v2AggressiveEnvCfg(TelloDefend3v2EnvCfg):
    """Attackers weave, jink, change speed and height at random, and dodge nearby defenders."""

    aggressive = True
    attacker_speed = [0.4, 1.0]  # [m/s] bursts up to the defenders' own top speed
    dodge_gain = 1.0
