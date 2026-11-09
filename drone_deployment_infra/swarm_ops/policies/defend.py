"""defend: swarm_defend.py on its drones (our defenders), as its own process (like formation).

The defenders need video + tag localization and the attacker feed, so they aren't stepped
per tick: swarm_policies.py launches swarm_defend.py with --only <its drones>, relays its
output, answers its FLY prompt once you have typed FLY, and sends it one SIGINT (= land all)
when the run ends. --defend-args passes its options (feed port/frame, face geometry, ...).
"""

import os
import shlex
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DefendPolicy:
    name = "defend"
    external = True

    def __init__(self, extra=""):
        self.extra = shlex.split(extra)

    def command(self, nums, fly, config):
        return ([sys.executable, "-u", os.path.join(HERE, "swarm_defend.py"),
                 "--only", ",".join(str(n) for n in nums), "--config", str(config)]
                + (["--fly"] if fly else []) + self.extra)
