# python/noob/envs/noob_drone_nav_env.py

import time
import math
import asyncio
from typing import Optional, Tuple, Dict, Any

import gym
from gym import spaces

from projectairsim import ProjectAirSimClient, World, Drone


class NoobNavOpenSpaceEnv(gym.Env):
    """
    Minimal Gym-style env for open-space waypoint navigation with a single quadrotor.

    Key points:
      - On construction, we connect to ProjectAirSim and capture the drone's
        initial *geodetic* spawn pose (lat, lon, alt + rotation).
      - If no waypoint_list is provided, we build a straight line of waypoints
        in front of the spawn, at ~3m above ground.
      - On every reset(), we teleport back to that geo pose using
        Drone.set_geo_pose(..., reset_kinematics=True) and then do a takeoff.
      - Episodes terminate when:
          * we get near the ground (within GROUND_HIT_TOL of GROUND_Z_AT_REST), or
          * we reach all waypoints, or
          * we hit max_episode_steps.
    """

    # Measured once via debug_ground_z.py for this scene.
    GROUND_Z_AT_REST = -0.187682
    # How far below that we still consider "near ground" (meters, NED).
    GROUND_HIT_TOL = 0.60

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        scene_config: str,
        sim_config_root: Optional[str] = None,
        waypoint_list: Optional[Tuple[Tuple[float, float, float], ...]] = None,
        action_dt: float = 0.1,
        max_episode_steps: int = 500,
        max_vel_mps: float = 5.0,
        max_yaw_rate_rad_s: float = math.radians(45.0),
        workspace_radius_m: float = 500.0,
        min_altitude_m: float = -200.0,
        max_altitude_m: float = 0.0,
        debug_state_structure: bool = False,
    ):
        super().__init__()

        self.scene_config = scene_config
        self.sim_config_root = sim_config_root
        self.action_dt = action_dt
        self.max_episode_steps = max_episode_steps
        self.max_vel_mps = max_vel_mps
        self.max_yaw_rate_rad_s = max_yaw_rate_rad_s
        self.workspace_radius_m = workspace_radius_m
        self.min_altitude_m = min_altitude_m
        self.max_altitude_m = max_altitude_m

        self.debug_state_structure = debug_state_structure
        self._debug_state_structure_done = False

        # Count how many times reset() has been called
        self._reset_call_count = 0

        # Whether user supplied custom waypoints
        self._custom_waypoints = waypoint_list is not None

        # Waypoints in world frame (x, y, z)
        # If user provided a list, use it directly.
        # Otherwise, we will create a line in front of the spawn
        # after we capture the spawn pose.
        if waypoint_list is not None:
            self.waypoints: Tuple[Tuple[float, float, float], ...] = waypoint_list
        else:
            # Placeholder; will be replaced after we capture spawn pose.
            self.waypoints = tuple()

        # Observation scaling (for normalization)
        obs_high = [
            workspace_radius_m,                             # |dx_to_wp|
            workspace_radius_m,                             # |dy_to_wp|
            abs(min_altitude_m) + abs(max_altitude_m),      # |dz_to_wp|
            max_vel_mps,                                    # |vx|
            max_vel_mps,                                    # |vy|
            max_vel_mps,                                    # |vz|
        ]
        self._obs_scale = obs_high

        # Normalized obs space
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=float
        )

        # Normalized action space (body-frame velocities + yaw rate)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=float
        )

        # ProjectAirSim client & world
        self.client: Optional[ProjectAirSimClient] = None
        self.world: Optional[World] = None
        self.drone: Optional[Drone] = None

        # Name of the drone in scene JSONC
        self._drone_name: str = "Drone1"

        # Visualize the intended path from waypoints
        self._waypoint_viz_drawn = False

        self._episode_step = 0
        self._current_wp_idx = 0
        self._prev_dist_to_wp = None

        # Track last position for things like ground-contact checks
        self._last_position = (0.0, 0.0, 0.0)

        # Geodetic spawn pose: dict with lat, lon, alt, rot
        self._spawn_geo: Optional[Dict[str, Any]] = None

        # --- Speed / timing diagnostics ---
        # Wall-clock start time (set on first reset)
        self._wall_start: Optional[float] = None
        # Global step counter across all episodes
        self._global_step: int = 0
        # Nominal simulated time from our RL loop (steps * action_dt)
        self._sim_time_nominal: float = 0.0
        # Sim-time from ProjectAirSim timestamps
        self._sim_ts_start: Optional[int] = None
        self._sim_ts_last: Optional[int] = None
        self._sim_time_from_ts: float = 0.0
        # How often to print speed stats (in env steps)
        self._speed_log_every_steps: int = 500

        self._connect()
        self._capture_spawn_geo_pose()
        self._draw_waypoint_debug_viz()

    # ------------------------------------------------------------------
    # ProjectAirSim wiring
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        """Connect to a running ProjectAirSim server and load scene + drone."""
        self.client = ProjectAirSimClient()
        self.client.connect()

        if self.sim_config_root is not None:
            self.world = World(
                self.client,
                self.scene_config,
                sim_config_path=self.sim_config_root,
                delay_after_load_sec=2,
            )
        else:
            self.world = World(
                self.client,
                self.scene_config,
                delay_after_load_sec=2,
            )

        # Assumes your scene JSONC has a robot named "Drone1"
        self.drone = Drone(self.client, self.world, self._drone_name)
        self.drone.enable_api_control()
        self.drone.arm()

    def _extract_yaw_from_quaternion(self, rot: Dict[str, float]) -> float:
        """
        Extract yaw (rotation about Z, in radians) from a quaternion.

        Assumes standard ENU/NED-like convention with yaw around Z.
        """
        w = rot["w"]
        x = rot["x"]
        y = rot["y"]
        z = rot["z"]

        # Yaw from quaternion (Z-Y-X / "yaw-pitch-roll" convention)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def _build_default_waypoints_from_spawn(
        self,
        spawn_x: float,
        spawn_y: float,
        spawn_rot: Dict[str, float],
    ) -> Tuple[Tuple[float, float, float], ...]:
        """
        Build a simple straight-line waypoint path "in front" of the spawn pose.

        - Use the spawn yaw to define a forward direction.
        - Place 4 waypoints at distances [5, 10, 15, 20] meters ahead.
        - Set their altitude to ~3 meters above ground.
        """
        yaw = self._extract_yaw_from_quaternion(spawn_rot)

        # Forward direction from yaw in XY plane
        fwd_x = math.cos(yaw)
        fwd_y = math.sin(yaw)

        # Target waypoint altitude: ~3m above ground
        z_wp = self.GROUND_Z_AT_REST - 3.0

        distances = [5.0, 10.0, 15.0, 20.0]
        wps = []
        for d in distances:
            x_wp = spawn_x + d * fwd_x
            y_wp = spawn_y + d * fwd_y
            wps.append((x_wp, y_wp, z_wp))

        if self.debug_state_structure:
            print("==== [NoobNavOpenSpaceEnv] Default waypoints from spawn ====")
            for i, (wx, wy, wz) in enumerate(wps):
                print(f"  WP{i}: x={wx:.2f}, y={wy:.2f}, z={wz:.2f}")

        return tuple(wps)

    def _capture_spawn_geo_pose(self) -> None:
        """
        Capture the initial geodetic spawn pose once,
        so we can teleport back to it on every reset via Drone.set_geo_pose().

        Uses:
          - Drone.get_ground_truth_geo_location() -> dict with keys
                "latitude", "longitude", "altitude"
          - Drone.get_ground_truth_pose() -> dict with keys
                "rotation" (Quaternion) and "translation" (Vector3)
        """
        assert self.drone is not None

        geo = self.drone.get_ground_truth_geo_location()
        pose = self.drone.get_ground_truth_pose()
        rot = pose["rotation"]
        translation = pose["translation"]

        spawn_x = float(translation["x"])
        spawn_y = float(translation["y"])
        spawn_z = float(translation["z"])

        self._spawn_geo = {
            "lat": float(geo["latitude"]),
            "lon": float(geo["longitude"]),
            "alt": float(geo["altitude"]),
            "rot": {
                "w": float(rot["w"]),
                "x": float(rot["x"]),
                "y": float(rot["y"]),
                "z": float(rot["z"]),
            },
            "world": {
                "x": spawn_x,
                "y": spawn_y,
                "z": spawn_z,
            },
        }

        if self.debug_state_structure:
            print("==== [NoobNavOpenSpaceEnv] Captured spawn geo pose ====")
            print(self._spawn_geo)

        # If user did NOT provide custom waypoints, build a line in front of the spawn
        if not self._custom_waypoints:
            self.waypoints = self._build_default_waypoints_from_spawn(
                spawn_x=spawn_x,
                spawn_y=spawn_y,
                spawn_rot=self._spawn_geo["rot"],
            )

    def close(self):
        if self.client is not None:
            try:
                if self.drone is not None:
                    # Best-effort cleanup
                    self.drone.disable_api_control()
                    self.drone.disarm()
            except Exception:
                pass
            self.client.disconnect()
        super().close()

    # ------------------------------------------------------------------
    # RL API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        self._episode_step = 0
        self._current_wp_idx = 0
        self._prev_dist_to_wp = None

        self._reset_call_count += 1
        if self.debug_state_structure:
            print(f"\n=== RESET CALLED #{self._reset_call_count} ===")

        # Initialize wall-clock & sim-time accumulators on first reset
        if self._wall_start is None:
            self._wall_start = time.time()
            self._sim_time_nominal = 0.0
            self._sim_ts_start = None
            self._sim_ts_last = None
            self._sim_time_from_ts = 0.0

        # One-time debug print of full state structure
        if self.debug_state_structure and not self._debug_state_structure_done:
            self._debug_print_state_structure()
            self._debug_state_structure_done = True

        async def _do_reset():
            assert self.drone is not None

            # 1) Teleport drone to initial spawn geo pose, if we captured it
            if self._spawn_geo is not None:
                lat = self._spawn_geo["lat"]
                lon = self._spawn_geo["lon"]
                alt = self._spawn_geo["alt"]
                rot = self._spawn_geo["rot"]

                if self.debug_state_structure:
                    print("[RESET] Teleporting drone via Drone.set_geo_pose()")

                try:
                    ok = self.drone.set_geo_pose(
                        latitude=lat,
                        longitude=lon,
                        altitude=alt,
                        rotation=rot,  # dict with keys w,x,y,z
                        reset_kinematics=True,
                    )
                    if self.debug_state_structure and not ok:
                        print(
                            "[RESET] WARNING: set_geo_pose returned False "
                            "(teleport may have failed)"
                        )
                except Exception as e:
                    if self.debug_state_structure:
                        print("[RESET] WARNING: set_geo_pose raised:", repr(e))

            # 2) Now do a takeoff so every episode starts airborne
            if self.debug_state_structure:
                print("[RESET] calling takeoff_async()")
            takeoff_task = await self.drone.takeoff_async()
            await takeoff_task

            # After takeoff, log the z position once per reset
            if self.debug_state_structure:
                pose_after = self.drone.get_ground_truth_pose()
                z_after = float(pose_after["translation"]["z"])
                vertical_dist = abs(z_after - self.GROUND_Z_AT_REST)
                print(
                    f"[RESET] post-takeoff z={z_after:.3f}, "
                    f"GROUND_Z_AT_REST={self.GROUND_Z_AT_REST:.3f}, "
                    f"vertical_dist_from_ground={vertical_dist:.3f}, "
                    f"GROUND_HIT_TOL={self.GROUND_HIT_TOL:.3f}"
                )

        asyncio.run(_do_reset())

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._episode_step += 1
        self._global_step += 1

        # Nominal simulated time: assume each step is action_dt seconds of sim
        self._sim_time_nominal += self.action_dt

        # Map normalized action -> physical command
        a_vfwd, a_vright, a_vdown, a_yaw = action
        v_forward = float(a_vfwd) * self.max_vel_mps
        v_right = float(a_vright) * self.max_vel_mps
        v_down = float(a_vdown) * self.max_vel_mps
        yaw_rate = float(a_yaw) * self.max_yaw_rate_rad_s

        if self._episode_step % 250 == 0 and self.debug_state_structure:
            print(
                f"[STEP {self._episode_step}] "
                f"action={action}, "
                f"v_forward={v_forward:.2f}, v_right={v_right:.2f}, "
                f"v_down={v_down:.2f}, yaw_rate={yaw_rate:.2f}"
            )

        # Send command and advance sim
        self._apply_action_and_step_sim(
            v_forward, v_right, v_down, yaw_rate
        )

        obs = self._get_obs()
        reward, done, info = self._compute_reward_and_done(obs)

        truncated = False
        if self._episode_step >= self.max_episode_steps:
            truncated = True
            done = True

        # --- Speed diagnostics: print every _speed_log_every_steps global steps ---
        if (
            self.debug_state_structure
            and self._wall_start is not None
            and self._global_step % self._speed_log_every_steps == 0
        ):
            wall_now = time.time()
            wall_elapsed = wall_now - self._wall_start
            if wall_elapsed <= 0:
                wall_elapsed = 1e-9

            steps_per_sec = self._global_step / wall_elapsed
            sim_nominal = self._sim_time_nominal
            sim_per_real_nominal = sim_nominal / wall_elapsed

            sim_ts = self._sim_time_from_ts
            sim_per_real_ts = sim_ts / wall_elapsed if sim_ts > 0.0 else 0.0

            print(
                "[SPEED] steps={:d}, wall={:.1f}s, "
                "sim_nominal={:.1f}s, sim_ts={:.1f}s, "
                "steps/s={:.1f}, sim/real_nom={:.1f}x, sim/real_ts={:.1f}x".format(
                    self._global_step,
                    wall_elapsed,
                    sim_nominal,
                    sim_ts,
                    steps_per_sec,
                    sim_per_real_nominal,
                    sim_per_real_ts,
                )
            )

        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _apply_action_and_step_sim(
        self,
        v_forward: float,
        v_right: float,
        v_down: float,
        yaw_rate: float,
    ) -> None:
        """
        Apply a velocity command and advance the sim by action_dt.
        """
        async def _do():
            assert self.drone is not None

            move_task = await self.drone.move_by_velocity_body_frame_async(
                v_forward=v_forward,
                v_right=v_right,
                v_down=v_down,
                duration=self.action_dt,
            )
            await move_task

            if abs(yaw_rate) > 1e-3:
                yaw_task = await self.drone.rotate_by_yaw_rate_async(
                    yaw_rate, self.action_dt
                )
                await yaw_task

        asyncio.run(_do())

    def _debug_print_state_structure(self) -> None:
        """
        One-time print of pose and kinematics dicts so you can verify key names.
        """
        assert self.drone is not None

        pose = self.drone.get_ground_truth_pose()
        kin = self.drone.get_ground_truth_kinematics()

        print("==== [NoobNavOpenSpaceEnv] Sample ground-truth pose ====")
        print(pose)
        print("==== [NoobNavOpenSpaceEnv] Sample ground-truth kinematics ====")
        print(kin)

    def _get_obs(self):
        """
        Build and normalize the observation vector using ProjectAirSim ground truth.
        """
        assert self.drone is not None, "Drone is not connected"

        # --- Pose / position (world frame) ---
        pose = self.drone.get_ground_truth_pose()
        translation = pose["translation"]
        x = float(translation["x"])
        y = float(translation["y"])
        z = float(translation["z"])

        # Remember last position so reward/done logic can use it
        self._last_position = (x, y, z)

        # --- Kinematics / linear velocity (world frame) ---
        kin = self.drone.get_ground_truth_kinematics()

        # Sim-time from ProjectAirSim timestamps (nanoseconds -> seconds)
        ts = kin.get("time_stamp", None)
        if isinstance(ts, (int, float)):
            ts = int(ts)
            if self._sim_ts_start is None:
                # First timestamp we see: init baseline
                self._sim_ts_start = ts
                self._sim_ts_last = ts
            else:
                dt_sim = (ts - self._sim_ts_last) / 1e9
                # Guard against negative or insane jumps
                if 0.0 <= dt_sim < 10.0:
                    self._sim_time_from_ts += dt_sim
                self._sim_ts_last = ts

        lin_vel = kin["twist"]["linear"]
        vx = float(lin_vel["x"])
        vy = float(lin_vel["y"])
        vz = float(lin_vel["z"])

        # --- Distance to current waypoint ---
        wp = self.waypoints[self._current_wp_idx]
        dx = wp[0] - x
        dy = wp[1] - y
        dz = wp[2] - z

        # --- Assemble and normalize ---
        obs_raw = [dx, dy, dz, vx, vy, vz]
        obs = [v / s for v, s in zip(obs_raw, self._obs_scale)]

        return obs

    def _compute_reward_and_done(self, obs):
        """
        Simple shaping + altitude shaping + early termination if we get too close to ground.

        Reward components:
          - 0.5 * progress toward current waypoint (change in distance)
          - small per-step time penalty
          - + waypoint bonuses
          - + altitude_bonus when altitude is in [1.9m, 4.0m] above ground
          - -5 and done=True when we get within GROUND_HIT_TOL of the ground
        """
        # Latest position from _get_obs (NED-ish: z ~ 0 near ground, negative is up)
        x, y, z = self._last_position

        # Denormalize first 3 obs entries into approximate dx, dy, dz
        dx, dy, dz = [obs[i] * self._obs_scale[i] for i in range(3)]
        dist_to_wp = math.sqrt(dx * dx + dy * dy + dz * dz)

        prev_dist = self._prev_dist_to_wp
        if prev_dist is None:
            prev_dist = dist_to_wp
        progress = prev_dist - dist_to_wp
        self._prev_dist_to_wp = dist_to_wp

        time_penalty = -0.01
        reward = 0.5 * progress + time_penalty

        # Altitude above ground (positive upwards)
        altitude_m = abs(z - self.GROUND_Z_AT_REST)

        # Altitude shaping: small bonus for being between 1.9m and 4m above ground
        altitude_low = 1.9
        altitude_high = 4.0
        altitude_bonus = 0.02
        if altitude_low <= altitude_m <= altitude_high:
            reward += altitude_bonus

        done = False
        info: Dict[str, Any] = {
            "dist_to_wp": dist_to_wp,
            "current_wp_idx": self._current_wp_idx,
            "altitude_m": altitude_m,
        }

        # 1) Waypoint reached?
        wp_reached = dist_to_wp < 1.0
        if wp_reached:
            reward += 8.0
            self._current_wp_idx += 1
            self._prev_dist_to_wp = None

            if self._current_wp_idx >= len(self.waypoints):
                reward += 20.0
                done = True
                info["success"] = True

        # 2) Ground-hit termination (unchanged logic)
        ground_threshold = self.GROUND_Z_AT_REST - self.GROUND_HIT_TOL

        if z > ground_threshold:
            reward -= 5.0
            done = True
            info["ground_hit"] = True

            if self.debug_state_structure:
                print(
                    f"[GROUND HIT] z={z:.3f}, ground_z={self.GROUND_Z_AT_REST:.3f}, "
                    f"vertical_dist={altitude_m:.3f}, tol={self.GROUND_HIT_TOL:.3f}, "
                    f"dist_to_wp={dist_to_wp:.2f}"
                )

        return reward, done, info

    def _draw_waypoint_debug_viz(self) -> None:
        """
        Visualize waypoints using ProjectAirSim World debug plotting APIs.
        """
        if self.world is None:
            print("[NoobNavOpenSpaceEnv] Debug viz skipped: world is None.")
            return

        world = self.world

        if not hasattr(world, "plot_debug_points"):
            print(
                "[NoobNavOpenSpaceEnv] Debug viz: World has no 'plot_debug_points'. "
                "Skipping waypoint visualization."
            )
            return

        if not self.waypoints:
            print(
                "[NoobNavOpenSpaceEnv] Debug viz: no waypoints defined yet, "
                "skipping visualization."
            )
            return

        points = [
            [float(x), float(y), float(z)]
            for (x, y, z) in self.waypoints
        ]

        color_rgba = [0.0, 1.0, 0.0, 1.0]
        duration = 0.0
        is_persistent = True

        try:
            world.plot_debug_points(
                points=points,
                color_rgba=color_rgba,
                size=25.0,
                duration=duration,
                is_persistent=is_persistent,
            )

            if hasattr(world, "plot_debug_solid_line"):
                world.plot_debug_solid_line(
                    points=points,
                    color_rgba=color_rgba,
                    thickness=5.0,
                    duration=duration,
                    is_persistent=is_persistent,
                )

            if hasattr(world, "plot_debug_strings"):
                labels = [f"WP{i}" for i in range(len(points))]
                world.plot_debug_strings(
                    strings=labels,
                    positions=points,
                    scale=1.0,
                    color_rgba=[1.0, 1.0, 1.0, 1.0],
                    duration=0.0,
                )

        except Exception as e:
            print(
                "[NoobNavOpenSpaceEnv] Warning: exception during waypoint debug viz:",
                repr(e),
            )
