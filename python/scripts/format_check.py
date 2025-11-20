# python/scripts/format_check.py

import os
import sys

# Ensure parent dir (.../python) is on sys.path when running from scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from noob.envs.noob_drone_nav_env import NoobNavOpenSpaceEnv
from noob.paths import SIM_CONFIG, scene


def main():
    env = NoobNavOpenSpaceEnv(
        scene_config=scene("scene_basic_drone.jsonc"),
        sim_config_root=SIM_CONFIG,
        debug_state_structure=True,
    )

    try:
        obs, info = env.reset()
        print("Initial obs:", obs)
        print("Info:", info)
    finally:
        print("Closing env and disconnecting client...")
        env.close()


if __name__ == "__main__":
    main()
