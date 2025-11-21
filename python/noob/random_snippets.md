### front-speed bonus reward:

    def _compute_reward_and_done(self, obs):
        """
        Reward components:
          - 0.5 * progress toward current waypoint (change in distance)
          - per-step time penalty (discourage loitering)
          - + speed_bonus for moving fast in the body-forward direction
            while roughly facing the waypoint
          - +20 on waypoint reached (within 1.0m), +40 when final waypoint reached
          - -50 and done=True when we get within GROUND_HIT_TOL of the ground
        """
        # Latest position from _get_obs (NED-ish: z ~ 0 near ground, negative is up)
        x, y, z = self._last_position
        vx, vy, vz = self._last_velocity
        yaw = self._last_yaw

        # Denormalize first 3 obs entries into approximate dx, dy, dz
        dx, dy, dz = [obs[i] * self._obs_scale[i] for i in range(3)]
        dist_to_wp = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6  # avoid zero

        prev_dist = self._prev_dist_to_wp
        if prev_dist is None:
            prev_dist = dist_to_wp
        progress = prev_dist - dist_to_wp
        self._prev_dist_to_wp = dist_to_wp

        # Base reward: progress + small time penalty
        time_penalty = -0.02
        reward = 0.5 * progress + time_penalty

        # -------- Heading-aware forward-speed bonus --------
        # Body-forward speed from world-frame velocity and yaw.
        # (Assumes AirSim NED: x forward, y right, z down)
        v_forward = vx * math.cos(yaw) + vy * math.sin(yaw)

        # Direction from drone to waypoint in XY (normalized)
        dir_wp_x = dx / dist_to_wp
        dir_wp_y = dy / dist_to_wp

        # Drone heading unit vector in XY plane
        heading_x = math.cos(yaw)
        heading_y = math.sin(yaw)

        # Alignment ∈ [-1, 1]; 1 == facing waypoint, -1 == facing away.
        alignment = dir_wp_x * heading_x + dir_wp_y * heading_y
        alignment = max(alignment, 0.0)  # ignore if facing away

        # Only reward positive forward speed, scaled by alignment
        # If v_forward ≈ 5 m/s and alignment≈1, bonus ≈ 0.1
        speed_weight = 0.02
        speed_bonus = speed_weight * max(v_forward, 0.0) * alignment
        reward += speed_bonus

        done = False
        info: Dict[str, Any] = {
            "dist_to_wp": dist_to_wp,
            "current_wp_idx": self._current_wp_idx,
            "altitude_m": abs(z - self.GROUND_Z_AT_REST),
            "v_forward": v_forward,
            "heading_alignment": alignment,
        }

        # 1) Waypoint reached?
        wp_reached = dist_to_wp < 1.0
        if wp_reached:
            reward += 15.0
            self._current_wp_idx += 1
            self._prev_dist_to_wp = None

            if self._current_wp_idx >= len(self.waypoints):
                reward += 30.0
                done = True
                info["success"] = True

        # 2) Ground-hit termination
        ground_threshold = self.GROUND_Z_AT_REST - self.GROUND_HIT_TOL

        if z > ground_threshold:
            reward -= 50.0
            done = True
            info["ground_hit"] = True

            if self.debug_state_structure:
                print(
                    f"[GROUND HIT] z={z:.3f}, ground_z={self.GROUND_Z_AT_REST:.3f}, "
                    f"vertical_dist={info['altitude_m']:.3f}, tol={self.GROUND_HIT_TOL:.3f}, "
                    f"dist_to_wp={dist_to_wp:.2f}"
                )

        return reward, done, info




#### altitude bonus:

        # Altitude shaping: small bonus for being between 1.7m and 4m above ground
        altitude_low = 1.7
        altitude_high = 4.0
        altitude_bonus = 0.02
        if altitude_low <= altitude_m <= altitude_high:
            reward += altitude_bonus
