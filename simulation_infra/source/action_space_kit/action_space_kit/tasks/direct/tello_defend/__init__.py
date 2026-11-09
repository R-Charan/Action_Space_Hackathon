"""Multi-agent defend task: the Action Space hackathon scenario."""

import gymnasium as gym

from . import agents

gym.register(
    id="AS-Defend-v0",
    entry_point=f"{__name__}.tello_defend_env:TelloDefendEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_defend_env_cfg:TelloDefendEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="AS-Defend-3v2-v0",
    entry_point=f"{__name__}.tello_defend_3v2_env:TelloDefend3v2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_defend_3v2_env_cfg:TelloDefend3v2EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="AS-Defend-3v2-Aggressive-v0",
    entry_point=f"{__name__}.tello_defend_3v2_env:TelloDefend3v2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_defend_3v2_env_cfg:TelloDefend3v2AggressiveEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

for _name, _cfg in (("AS-Defend-Face-v0", "TelloDefendFaceEnvCfg"), ("AS-Defend-Face-Aggressive-v0", "TelloDefendFaceAggressiveEnvCfg"),
                    ("AS-Defend-Face-Room-v0", "TelloDefendFaceRoomEnvCfg"),
                    ("AS-Defend-Face-Room-Aggressive-v0", "TelloDefendFaceRoomAggressiveEnvCfg")):
    gym.register(
        id=_name,
        entry_point=f"{__name__}.tello_defend_face_env:TelloDefendFaceEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tello_defend_face_env_cfg:{_cfg}",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )

for _name, _cfg in (("AS-Attack-Face-v0", "TelloAttackFaceEnvCfg"), ("AS-Attack-Face-Room-v0", "TelloAttackFaceRoomEnvCfg")):
    gym.register(
        id=_name,
        entry_point=f"{__name__}.tello_attack_face_env:TelloAttackFaceEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tello_attack_face_env_cfg:{_cfg}",
            "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_attack_cfg.yaml",
        },
    )
