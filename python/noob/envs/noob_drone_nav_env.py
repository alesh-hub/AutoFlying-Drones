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
        waypoint_pattern: Optional[str] = None,   
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

        self._last_position = (0.0, 0.0, 0.0)
        self._last_velocity = (0.0, 0.0, 0.0)   
        self._last_yaw = 0.0                    

        self.waypoint_pattern = waypoint_pattern

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
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def _build_waypoints_from_pattern(
        self,
        spawn_x: float,
        spawn_y: float,
        spawn_rot: Dict[str, float],
    ) -> Tuple[Tuple[float, float, float], ...]:
        """
        Build waypoints from a named body-frame pattern.

        Patterns are specified as (dx, dy, alt_above_ground_m) in the drone's
        forward-right-up frame at spawn, then rotated into world XY using yaw.
        """

        if self.waypoint_pattern is None:
            return self._build_default_waypoints_from_spawn(
                spawn_x=spawn_x,
                spawn_y=spawn_y,
                spawn_rot=spawn_rot,
            )

        # Body-frame waypoint definitions: (dx, dy, altitude_above_ground_m)
        # dx along forward, dy to the right.
        pattern_defs: Dict[str, Tuple[Tuple[float, float, float], ...]] = {
            # Similar to your original 5/10/15/20m line, all at 3m AGL
            "line_20m": (
                (5.0, 0.0, 3.0),
                (10.0, 0.0, 3.0),
                (15.0, 0.0, 3.0),
                (20.0, 0.0, 3.0),
            ),
            # Longer straight line out to 60m, still at 3m AGL
            "line_60m": (
                (10.0, 0.0, 3.0),
                (20.0, 0.0, 3.0),
                (40.0, 0.0, 3.0),
                (60.0, 0.0, 3.0),
            ),
            # 20m square at 3m AGL
            "square_20m": (
                (0.0,   0.0, 3.0),
                (20.0,  0.0, 3.0),
                (20.0, 20.0, 3.0),
                (0.0,  20.0, 3.0),
                (0.0,   0.0, 3.0),
            ),
            # “Stairs” in altitude along the forward direction
            "alt_stairs": (
                (10.0, 0.0, 2.0),
                (20.0, 0.0, 4.0),
                (30.0, 0.0, 2.0),
                (40.0, 0.0, 4.0),
            ),
        }

        if self.waypoint_pattern not in pattern_defs:
            if self.debug_state_structure:
                print(
                    f"[WAYPOINTS] Unknown pattern '{self.waypoint_pattern}', "
                    "falling back to default line."
                )
            return self._build_default_waypoints_from_spawn(
                spawn_x=spawn_x,
                spawn_y=spawn_y,
                spawn_rot=spawn_rot,
            )

        body_points = pattern_defs[self.waypoint_pattern]

        yaw = self._extract_yaw_from_quaternion(spawn_rot)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        wps = []
        for dx_b, dy_b, alt_agl in body_points:
            # Rotate body-frame XY into world XY
            x_wp = spawn_x + dx_b * cos_y - dy_b * sin_y
            y_wp = spawn_y + dx_b * sin_y + dy_b * cos_y

            # Convert altitude-above-ground (positive up) to world Z (negative up)
            z_wp = self.GROUND_Z_AT_REST - alt_agl

            wps.append((x_wp, y_wp, z_wp))

        if self.debug_state_structure:
            print(f"==== [NoobNavOpenSpaceEnv] Waypoints pattern '{self.waypoint_pattern}' ====")
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

        # If user did NOT provide custom waypoints, build from pattern or default
        if not self._custom_waypoints:
            if self.waypoint_pattern is not None:
                self.waypoints = self._build_waypoints_from_pattern(
                    spawn_x=spawn_x,
                    spawn_y=spawn_y,
                    spawn_rot=self._spawn_geo["rot"],
                )
            else:
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
        rot = pose["rotation"]                # <-- NEW
        x = float(translation["x"])
        y = float(translation["y"])
        z = float(translation["z"])

        # Remember last position
        self._last_position = (x, y, z)

        # Cache yaw (from quaternion) for reward shaping
        yaw = self._extract_yaw_from_quaternion(
            {"w": float(rot["w"]), "x": float(rot["x"]),
             "y": float(rot["y"]), "z": float(rot["z"])}
        )
        self._last_yaw = yaw

        # --- Kinematics / linear velocity (world frame) ---
        kin = self.drone.get_ground_truth_kinematics()
        lin_vel = kin["twist"]["linear"]
        vx = float(lin_vel["x"])
        vy = float(lin_vel["y"])
        vz = float(lin_vel["z"])

        # Cache velocity
        self._last_velocity = (vx, vy, vz)

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
          - +15 on waypoint reached, +30 when final waypoint reached
          - +altitude_bonus when altitude is in [1.9m, 4.0m] above ground
          - -50 and done=True when we get within GROUND_HIT_TOL of the ground
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

        time_penalty = -0.05
        reward = 2.0 * progress + time_penalty

        # Altitude above ground (positive upwards)
        altitude_m = abs(z - self.GROUND_Z_AT_REST)

        # # Altitude shaping: small bonus for being between 1.7m and 4m above ground
        # altitude_low = 1.7
        # altitude_high = 4.0
        # altitude_bonus = 0.02
        # if altitude_low <= altitude_m <= altitude_high:
        #     reward += altitude_bonus

        done = False
        info: Dict[str, Any] = {
            "dist_to_wp": dist_to_wp,
            "current_wp_idx": self._current_wp_idx,
            "altitude_m": altitude_m,
        }

        # 1) Waypoint reached?
        wp_reached = dist_to_wp < 1.0
        if wp_reached:
            reward += 30.0
            self._current_wp_idx += 1
            self._prev_dist_to_wp = None

            if self._current_wp_idx >= len(self.waypoints):
                reward += 60.0
                done = True
                info["success"] = True

        # 2) Ground-hit termination (unchanged logic)
        ground_threshold = self.GROUND_Z_AT_REST - self.GROUND_HIT_TOL

        if z > ground_threshold:
            reward -= 50.0
            done = True
            info["ground_hit"] = True

            if self.debug_state_structure:
                print(
                    f"[GROUND HIT] z={z:.3f}, ground_z={self.GROUND_Z_AT_REST:.3f}, "
                    f"vertical_dist={altitude_m:.3f}, tol={self.GROUND_HIT_TOL:.3f}, "
                    f"dist_to_wp={dist_to_wp:.2f}"
                )

        # 3) Optional workspace bounds (still off for now)
        return reward, done, info
    
    # def _compute_reward_and_done(self, obs):
    #     """
    #     Reward components:
    #       - 0.5 * progress toward current waypoint (change in distance)
    #       - per-step time penalty (discourage loitering)
    #       - + speed_bonus for moving fast in the body-forward direction
    #         while roughly facing the waypoint
    #       - +20 on waypoint reached (within 1.0m), +40 when final waypoint reached
    #       - -50 and done=True when we get within GROUND_HIT_TOL of the ground
    #     """
    #     # Latest position from _get_obs (NED-ish: z ~ 0 near ground, negative is up)
    #     x, y, z = self._last_position
    #     vx, vy, vz = self._last_velocity
    #     yaw = self._last_yaw

    #     # Denormalize first 3 obs entries into approximate dx, dy, dz
    #     dx, dy, dz = [obs[i] * self._obs_scale[i] for i in range(3)]
    #     dist_to_wp = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6  # avoid zero

    #     prev_dist = self._prev_dist_to_wp
    #     if prev_dist is None:
    #         prev_dist = dist_to_wp
    #     progress = prev_dist - dist_to_wp
    #     self._prev_dist_to_wp = dist_to_wp

    #     # Base reward: progress + small time penalty
    #     time_penalty = -0.02
    #     reward = 0.5 * progress + time_penalty

    #     # -------- Heading-aware forward-speed bonus --------
    #     # Body-forward speed from world-frame velocity and yaw.
    #     # (Assumes AirSim NED: x forward, y right, z down)
    #     v_forward = vx * math.cos(yaw) + vy * math.sin(yaw)

    #     # Direction from drone to waypoint in XY (normalized)
    #     dir_wp_x = dx / dist_to_wp
    #     dir_wp_y = dy / dist_to_wp

    #     # Drone heading unit vector in XY plane
    #     heading_x = math.cos(yaw)
    #     heading_y = math.sin(yaw)

    #     # Alignment ∈ [-1, 1]; 1 == facing waypoint, -1 == facing away.
    #     alignment = dir_wp_x * heading_x + dir_wp_y * heading_y
    #     alignment = max(alignment, 0.0)  # ignore if facing away

    #     # Only reward positive forward speed, scaled by alignment
    #     # If v_forward ≈ 5 m/s and alignment≈1, bonus ≈ 0.1
    #     speed_weight = 0.02
    #     speed_bonus = speed_weight * max(v_forward, 0.0) * alignment
    #     reward += speed_bonus

    #     done = False
    #     info: Dict[str, Any] = {
    #         "dist_to_wp": dist_to_wp,
    #         "current_wp_idx": self._current_wp_idx,
    #         "altitude_m": abs(z - self.GROUND_Z_AT_REST),
    #         "v_forward": v_forward,
    #         "heading_alignment": alignment,
    #     }

    #     # 1) Waypoint reached?
    #     wp_reached = dist_to_wp < 1.0
    #     if wp_reached:
    #         reward += 15.0
    #         self._current_wp_idx += 1
    #         self._prev_dist_to_wp = None

    #         if self._current_wp_idx >= len(self.waypoints):
    #             reward += 30.0
    #             done = True
    #             info["success"] = True

    #     # 2) Ground-hit termination
    #     ground_threshold = self.GROUND_Z_AT_REST - self.GROUND_HIT_TOL

    #     if z > ground_threshold:
    #         reward -= 50.0
    #         done = True
    #         info["ground_hit"] = True

    #         if self.debug_state_structure:
    #             print(
    #                 f"[GROUND HIT] z={z:.3f}, ground_z={self.GROUND_Z_AT_REST:.3f}, "
    #                 f"vertical_dist={info['altitude_m']:.3f}, tol={self.GROUND_HIT_TOL:.3f}, "
    #                 f"dist_to_wp={dist_to_wp:.2f}"
    #             )

    #     return reward, done, info



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
