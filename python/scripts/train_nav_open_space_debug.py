# python/scripts/train_nav_open_space_debug.py

import os
import sys
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Ensure parent dir (.../python) is on sys.path when running from scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from noob.envs.noob_drone_nav_env_speed_debug import NoobNavOpenSpaceEnv
from noob.paths import SIM_CONFIG, scene


def make_env(scene_name: str, action_dt: float, max_episode_steps: int):
    """
    Factory to create a fresh env instance.
    SB3 will wrap this in DummyVecEnv (for now, single env).
    """

    def _thunk():
        env = NoobNavOpenSpaceEnv(
            scene_config=scene(scene_name),
            sim_config_root=SIM_CONFIG,
            action_dt=action_dt,
            max_episode_steps=max_episode_steps,
            debug_state_structure=True,  # keep True for now to see reset/ground logs
        )
        return env

    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="scene_basic_drone_fast.jsonc",
        help="Scene JSONC filename (looked up via noob.paths.scene).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50_000,
        help="Total PPO training timesteps.",
    )
    parser.add_argument(
        "--action-dt",
        type=float,
        default=0.1,
        help="Control time step (seconds) per RL action.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=500,
        help="Max env steps per episode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for PPO ('cpu' or 'cuda').",
    )
    args = parser.parse_args()

    # Single-env setup for now.
    env = DummyVecEnv(
        [make_env(args.scene, args.action_dt, args.max_episode_steps)]
    )
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join("runs", "tensorboard"),
        device=args.device,
    )

    print("Starting PPO training on device:", args.device)
    model.learn(
        total_timesteps=args.total_timesteps,
    )

    # Save final model
    model.save("ppo_nav_open_space_debug_final")
    env.close()


if __name__ == "__main__":
    main()
