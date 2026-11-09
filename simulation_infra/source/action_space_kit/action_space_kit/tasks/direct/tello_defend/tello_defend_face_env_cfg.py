"""Defend a breach face: the hardware-round format, 3 defenders vs 2 identical attackers.

Sizes are placeholders until the organizers' numbers arrive: a 3 m x 2 m face (bottom edge
0.5 m up) on the front of a box, attackers starting ~6 m away, both sides up to 1.5 m/s.
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from action_space_kit.control import TelloCommandCfg

from .tello_defend_env_cfg import TelloDefendEnvCfg

NUM_DEFENDERS = 3
NUM_ATTACKERS = 2
_OBS_DIM = 9 + 7 * NUM_ATTACKERS


@configclass
class TelloDefendFaceEnvCfg(TelloDefendEnvCfg):
    possible_agents = [f"defender_{i}" for i in range(NUM_DEFENDERS)]
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}
    num_defenders = NUM_DEFENDERS
    num_attackers = NUM_ATTACKERS

    episode_length_s = 30.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=16.0, replicate_physics=True)
    # identical drones: the same command limits on both sides (rc 100 -> 1.5 m/s here)
    command: TelloCommandCfg = TelloCommandCfg(max_lin_speed=1.5)

    # - the breach face (env-local): plane x = 0, |y| <= width/2, bottom <= z <= bottom + height
    face_width = 3.0
    face_height = 2.0
    face_bottom = 0.5
    box_depth = 1.0  # [m] the box behind the face (visual only)
    # an attacker crossing x = 0 outside the face (around the box) is out of play, not a breach

    # - the arena: any drone leaving it (or dropping below the floor margin) is downed
    arena_x = [-1.5, 8.0]
    arena_y = [-5.0, 5.0]
    arena_z = [0.2, 3.2]

    # - starts
    attacker_start_x = [5.5, 6.5]
    attacker_start_y = [-3.0, 3.0]
    attacker_start_z = [1.0, 2.5]
    attacker_start_separation = 1.0  # [m] attackers never start closer than this to each other
    defender_start_x = 1.0
    defender_start_y = [-1.0, 0.0, 1.0]
    defender_start_z = 1.5
    start_jitter = 0.15  # [m]
    # defenders on the floor (their start markers) and the SDK `takeoff` at the whistle: the firmware
    # climbs at takeoff_speed to takeoff_height ignoring sticks, then the pilot flies (room tasks)
    defender_takeoff = False
    takeoff_speed = 0.5  # [m/s]
    takeoff_height = (0.8, 1.0)  # [m] per drone per episode
    floor_start_z = 0.08  # [m] a landed Tello's centre

    # - the real room (room_layout.py), off by default: map.yaml's tags, its walls as the arena, the box
    #   against the far wall opposite tag 10. When on, the arena and start boxes above are derived from these.
    room_map = False
    room_depth = 5.2  # [m] tag-10 wall to the far wall: everything is inside the tags (the furthest is 5.18 m out)
    room_y = (-2.0, 3.2)  # [m] side walls, room Y (along the tag-10 wall; 0 under tag 10)
    face_center_y = 0.0  # [m] room Y of the face centre: the base is directly opposite tag 10
    attacker_start_room_x = (0.3, 0.6)  # [m] attackers start this far out from tag 10 ...
    attacker_start_room_y = (-1.5, 1.5)  # [m] ... and within this of it along the wall (room Y)
    room_start_margin = 0.5  # [m] attackers start at least this far from the side walls
    tag10_height = 1.22  # [m] tag 10 centre above the floor (measured)

    # - scripted attackers: fly at a random point of the face
    attacker_speed = [0.8, 1.5]  # [m/s]
    attacker_max_speed = 1.5  # [m/s] the drones' limit: every attacker command is capped to this speed
    bait_prob = 0.5  # per attacker: fly in to bait_depth from the face, back out past bait_retreat_x, then dive
    bait_depth = [1.0, 3.0]  # [m] from the face
    bait_retreat_x = 5.0  # [m]
    aim_interval = [0.0, 0.0]  # [s] re-pick the aim point this often (0 = never; the aggressive cfg sets it)
    aggressive = False
    weave_amp = [0.3, 1.0]
    weave_freq = [0.3, 0.8]  # [Hz]
    jink_amp = 0.8
    jink_interval = [0.4, 1.2]  # [s]
    speed_interval = [0.5, 1.5]  # [s]
    dodge_gain = 0.0
    dodge_radius = 1.5  # [m]
    # attackers keep clear of each other (a real team doesn't fly its drones together) but may fly
    # close as a pair: free beyond attacker_separation, all closing speed gone by attacker_min_gap,
    # which stays outside collision_radius
    attacker_separation = 0.6  # [m] (0 = off)
    attacker_min_gap = 0.45  # [m]
    attacker_react_s = 0.3  # [s] the gap is judged this far ahead at the actual closing speed
    attacker_separation_push = 0.5  # [m/s] push apart at attacker_min_gap, fading to 0 at attacker_separation

    # - the match: one wave; two drones within collision_radius are both downed
    respawn_waves = False
    collision_radius = 0.35  # [m] centre to centre
    rew_scale_defender_lost = -15.0
    rew_scale_teamkill = -30.0
    downed_descent_speed = 0.5  # [m/s] a downed drone is sent `land`
    downed_floor_height = 0.15


@configclass
class TelloDefendFaceAggressiveEnvCfg(TelloDefendFaceEnvCfg):
    """Attackers weave, jink, switch aim point and speed at random, and dodge nearby defenders."""

    aggressive = True
    aim_interval = [0.8, 2.0]
    dodge_gain = 1.0


@configclass
class TelloDefendFaceRoomEnvCfg(TelloDefendFaceEnvCfg):
    # the final arena: the real room from map.yaml, breach opposite tag 10, attackers launch beside
    # tag 10, defenders take off from three floor markers 0.8 m in front of the face
    room_map = True
    defender_takeoff = True
    defender_start_x = 0.8
    attacker_start_z = [1.0, 2.0]


@configclass
class TelloDefendFaceRoomAggressiveEnvCfg(TelloDefendFaceAggressiveEnvCfg):
    # the final arena: the real room from map.yaml, breach opposite tag 10, attackers launch beside
    # tag 10, defenders take off from three floor markers 0.8 m in front of the face
    room_map = True
    defender_takeoff = True
    defender_start_x = 0.8
    attacker_start_z = [1.0, 2.0]
