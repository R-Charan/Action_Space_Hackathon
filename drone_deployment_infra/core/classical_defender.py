"""Load the classical bus defender (simulation_infra's ``tello_defend/classical_policy.py``) without Isaac.

The bus is plain torch, but its package's ``__init__`` files register Isaac Lab tasks. This
registers bare package stubs for the path down to it, so only the pure-torch modules run:

    from core.classical_defender import load_classical
    cp = load_classical()                                   # the classical_policy module
    policy = cp.ClassicalPolicy.for_room(site=(4.2, 0.0, 1.5))
    act = policy.act(obs)                                   # (3, 59) -> (3, 4), like SharedPolicy.act

Self-test (no drones, no Isaac): ``python3 drone_deployment_infra/core/classical_defender.py``
(pass: ``SELFTEST PASSED``).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "simulation_infra" / "source" / "action_space_kit" / "action_space_kit"


def load_classical(kit: Path = KIT):
    """The ``classical_policy`` module, imported without running the kit's Isaac-dependent ``__init__`` files."""
    for name, sub in (("action_space_kit", ""), ("action_space_kit.tags", "tags"),
                      ("action_space_kit.tasks", "tasks"), ("action_space_kit.tasks.direct", "tasks/direct"),
                      ("action_space_kit.tasks.direct.tello_defend", "tasks/direct/tello_defend")):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(kit / sub)]
            sys.modules[name] = mod
    return importlib.import_module("action_space_kit.tasks.direct.tello_defend.classical_policy")


def _selftest() -> None:
    import torch

    cp = load_classical()
    site = torch.tensor([4.2, 0.0, 1.5])
    policy = cp.ClassicalPolicy.for_room(site=site)
    # defenders hovering 0.8 m in front of the face (world X 3.4), attackers launching beside tag 10
    d_pos = torch.tensor([[3.4, -1.0, 1.0], [3.4, 0.0, 1.0], [3.4, 1.0, 1.0]])
    a_pos = torch.tensor([[0.5, -0.5, 1.5], [0.5, 0.8, 1.2]])
    a_vel = torch.tensor([[1.4, 0.0, 0.0], [1.4, 0.0, 0.0]])
    alive = torch.ones(2, dtype=torch.bool)
    for _ in range(30):
        a_pos = a_pos + a_vel / 13.5
        obs = cp.build_obs(d_pos, torch.zeros(3, 3), a_pos, a_vel, alive, site, torch.zeros(3, 4), torch.zeros(3))
        assert obs.shape == (3, 59), obs.shape
        act = policy.act(obs)
        assert act.shape == (3, 4) and act.abs().max() <= 1.0 and torch.isfinite(act).all(), act
        d_pos = d_pos + cp.safety(act, d_pos)[:, :3] / 13.5
    # the round decoded back: every view agrees with the state it was built from
    obs = cp.build_obs(d_pos, torch.zeros(3, 3), a_pos, a_vel, alive, site, torch.zeros(3, 4), torch.zeros(3))
    dp, _, da, ap, _, aa = policy.decode(obs)
    assert torch.allclose(dp, d_pos, atol=1e-4) and torch.allclose(ap, a_pos, atol=1e-4) and da.all() and aa.all()
    # a defender that is out: decoded as out, commanded nothing, and the others still fly
    dead = torch.tensor([True, False, True])
    obs = cp.build_obs(d_pos, torch.zeros(3, 3), a_pos, a_vel, alive, site, torch.zeros(3, 4), torch.zeros(3), dead)
    act = policy.act(obs)
    assert not policy.decode(obs)[2][1] and act[1].abs().sum() == 0 and act[[0, 2], :3].abs().sum() > 0, act
    print(f"[classical] defenders fell back between the attackers and the face: world X {d_pos[:, 0].tolist()}")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    _selftest()
