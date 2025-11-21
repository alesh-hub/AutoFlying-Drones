# python/scripts/train_nav_open_space.py

import os
import sys
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import wandb
from wandb.integration.sb3 import WandbCallback

# Ensure parent dir (.../python) is on sys.path when running from scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from noob.envs.noob_drone_nav_env import NoobNavOpenSpaceEnv
from noob.paths import SIM_CONFIG, scene


def make_env(scene_name: str, action_dt: float, max_episode_steps: int, waypoint_pattern: str):
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
            waypoint_pattern=waypoint_pattern,
            debug_state_structure=False,  # no printing during training
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
        default=500_000,
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
        "--project",
        type=str,
        default="airsim-drone-rl",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="nav_open_space_500k",
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for PPO ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--wp-pattern",
        type=str,
        default="line_20m",
        help="Named waypoint pattern (line_20m, line_60m, square_20m, alt_stairs).",
    )
    args = parser.parse_args()

    # W&B init
    wandb_run = wandb.init(
        project=args.project,
        name=args.run_name,
        config={
            "algo": "PPO",
            "task": "nav_open_space",
            "scene": args.scene,
            "action_dt": args.action_dt,
            "max_episode_steps": args.max_episode_steps,
            "total_timesteps": args.total_timesteps,
            "device": args.device,
            "waypoint_pattern": args.wp_pattern,
        },
        sync_tensorboard=True,
    )

    # Ensure dirs exist
    os.makedirs("./checkpoints", exist_ok=True)

    # Single-env setup for now.
    env = DummyVecEnv(
        [make_env(args.scene, args.action_dt, args.max_episode_steps, args.wp_pattern)]
    )
    # Wrap with VecMonitor (this is SB3's vec-level monitor, not gymnasium.Monitor)
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join("runs", "tensorboard"),
        device=args.device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints",
        name_prefix="ppo_nav_open_space_20m_rebuilt",
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path="models",
        verbose=1,
    )

    print("Starting PPO training on device:", args.device)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, wandb_callback],
    )

    # Save final model
    model.save("ppo_nav_open_space_500k_rebuilt")

    wandb_run.finish()
    env.close()


if __name__ == "__main__":
    main()
