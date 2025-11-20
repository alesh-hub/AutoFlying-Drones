# python/scripts/eval_nav_open_space.py

import os
import sys
import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure parent dir (.../python) is on sys.path when running from scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from noob.envs.noob_drone_nav_env import NoobNavOpenSpaceEnv
from noob.paths import SIM_CONFIG, scene


def make_env(scene_name: str,
             action_dt: float,
             max_episode_steps: int,
             debug_state_structure: bool = False):
    """
    Factory to create a fresh env instance.
    Used for DummyVecEnv so the setup matches training.
    """
    def _thunk():
        env = NoobNavOpenSpaceEnv(
            scene_config=scene(scene_name),
            sim_config_root=SIM_CONFIG,
            action_dt=action_dt,
            max_episode_steps=max_episode_steps,
            waypoint_pattern="line_20m",
            debug_state_structure=debug_state_structure,
        )
        return env

    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="scene_basic_drone.jsonc",
        help="Scene JSONC filename (looked up via noob.paths.scene).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ppo_nav_open_space_basic.zip",
        help="Path to the trained PPO model .zip.",
    )
    parser.add_argument(
        "--action-dt",
        type=float,
        default=0.1,
        help="Control time step (seconds) per RL action (must match training).",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=500,
        help="Max env steps per episode (must match training).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for PPO ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--debug-env",
        action="store_true",
        help="If set, enables env debug_state_structure (lots of prints).",
    )
    parser.add_argument(
        "--slowdown",
        type=float,
        default=0.0,
        help=(
            "Optional sleep factor per step: "
            "sleep = action_dt / slowdown. "
            "Use >1.0 to slow wall-clock down, 0 = no sleep."
        ),
    )

    args = parser.parse_args()

    # Create vec env (same structure as training)
    env = DummyVecEnv([
        make_env(
            args.scene,
            args.action_dt,
            args.max_episode_steps,
            debug_state_structure=args.debug_env,
        )
    ])

    # Load trained model
    print(f"Loading model from: {args.model_path} on device={args.device}")
    model = PPO.load(
        args.model_path,
        env=env,
        device=args.device,
        print_system_info=False,
    )

    print(f"Starting evaluation for {args.n_episodes} episodes...")
    # For convenience, get underlying single env
    base_env = env.envs[0]

    for ep in range(1, args.n_episodes + 1):
        obs = env.reset()
        done = False
        ep_rew = 0.0
        ep_len = 0

        # Optional info about reset location / waypoint index etc.
        print(f"\n=== EVAL EPISODE {ep} ===")

        while not done:
            # Deterministic policy: no extra exploration noise
            action, _ = model.predict(obs, deterministic=True)

            obs, rewards, dones, infos = env.step(action)

            r = float(rewards[0])
            d = bool(dones[0])
            info = infos[0]

            ep_rew += r
            ep_len += 1
            done = d

            # Optional small sleep to slow things to human-watchable speed
            if args.slowdown > 0.0:
                time.sleep(args.action_dt / args.slowdown)

            if done:
                # Try to pull interesting flags from info
                success = info.get("success", False)
                ground_hit = info.get("ground_hit", False)
                dist_to_wp = info.get("dist_to_wp", None)

                suffix_bits = []
                if success:
                    suffix_bits.append("success=True")
                if ground_hit:
                    suffix_bits.append("ground_hit=True")
                if dist_to_wp is not None:
                    suffix_bits.append(f"final_dist_to_wp={dist_to_wp:.2f}")

                suffix = ""
                if suffix_bits:
                    suffix = " (" + ", ".join(suffix_bits) + ")"

                print(
                    f"[EVAL] Episode {ep} finished: "
                    f"len={ep_len}, rew={ep_rew:.2f}{suffix}"
                )

                # Safety: break if we somehow exceed max_episode_steps
                if ep_len >= args.max_episode_steps:
                    print(
                        f"[EVAL] Reached max_episode_steps={args.max_episode_steps}, "
                        "forcing episode termination."
                    )
                break

    env.close()
    print("Evaluation done.")


if __name__ == "__main__":
    main()
