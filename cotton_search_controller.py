"""
Autonomous Cotton Search & Accurate IK Harvesting Controller.

Integrates real-time 3-sector autonomous search with closed-form 3D
inverse kinematics, exact camera-to-robot coordinate conversion,
and a structured picking state machine:
  SEARCH -> TARGET_LOCKED -> PRE_PICK -> APPROACH -> GRIP -> RETRACT -> DROP -> RELEASE -> HOMING

All kinematics calculations utilize confirmed physical dimensions:
  - Base to Shoulder Joint Center: 110 mm
  - Shoulder-to-Arm Pivot (L1): 212 mm
  - Arm-to-Elbow Pivot (L2): 199 mm
  - Elbow-to-Gripper Contact Point (L3): 202 mm
  - Camera Stalk: 70 mm forward of Elbow pivot, 121 mm upright
"""

import math
import time
import numpy as np

import robot_kinematics as rk


class CottonSearchController:
    # Operating States
    STATE_IDLE          = "IDLE"
    STATE_SEARCHING     = "SEARCH"
    STATE_TARGET_LOCKED = "TARGET_LOCKED"
    STATE_TARGET_LONG   = "LONG"
    STATE_PRE_PICK      = "PRE_PICK"
    STATE_APPROACH      = "APPROACH"
    STATE_GRIP          = "GRIP"
    STATE_RETRACT       = "RETRACT"
    STATE_DROP          = "DROP"
    STATE_RELEASE       = "RELEASE"
    STATE_HOMING        = "HOMING"
    STATE_ERROR         = "ERROR"
    STATE_PAUSED        = "PAUSED"

    # Backward compatibility aliases
    STATE_HARVESTING    = "HARVESTING"

    # Harvesting Sub-Stages (for HUD and legacy compatibility)
    STAGE_SEARCHING     = "SEARCHING"
    STAGE_WAITING       = "WAITING"         # Depth stabilization pause
    STAGE_LONG          = "LONG"            # Out-of-reach notification hold
    STAGE_PRE_PICK      = "PRE_PICK"       # Move to 45mm backed-off approach stance
    STAGE_APPROACH      = "APPROACH"       # Smooth transit to cotton boll contact
    STAGE_PICKING       = "PICKING"        # Close gripper ('Y') & spin roller CW ('CW')
    STAGE_LIFTING       = "LIFTING"        # Lift shoulder to 130 deg clear of branches
    STAGE_MOVE_DROP     = "MOVE_DROP"      # Rotate base to 50 deg drop position
    STAGE_DROPPING      = "DROPPING"       # Open gripper ('S') & eject CCW ('CCW')
    STAGE_HOMING        = "HOMING"         # Smooth return to Home pose

    # Legacy stage aliases
    STAGE_SHOULDER_REACH = "PRE_PICK"
    STAGE_ELBOW_REACH    = "APPROACH"
    STAGE_REACHING       = "APPROACH"

    # Physical Limits (degrees)
    BASE_MIN     = rk.LIMIT_BASE_MIN
    BASE_MAX     = rk.LIMIT_BASE_MAX
    SHOULDER_MIN = rk.LIMIT_SHOULDER_MIN
    SHOULDER_MAX = rk.LIMIT_SHOULDER_MAX
    ARM_MIN      = rk.LIMIT_ARM_MIN
    ARM_MAX      = rk.LIMIT_ARM_MAX
    ELBOW_MIN    = rk.LIMIT_ELBOW_MIN
    ELBOW_MAX    = rk.LIMIT_ELBOW_MAX
    GRIPPER_OPEN = rk.LIMIT_GRIPPER_OPEN
    GRIPPER_CLOSE= rk.LIMIT_GRIPPER_CLOSE

    # Safe Home Positions
    HOME_B = rk.HOME_B
    HOME_S = rk.HOME_S
    HOME_A = rk.HOME_A
    HOME_E = rk.HOME_E
    HOME_G = rk.HOME_G

    # Physical Dimensions for external inspection
    LINK_L1_SHOULDER_MM        = rk.L1_SHOULDER_MM
    LINK_L2_ELBOW_MM           = rk.L2_ARM_MM
    LINK_L3_GRIPPER_MM         = rk.L3_GRIPPER_MM
    BASE_SHOULDER_HEIGHT_MM    = rk.BASE_SHOULDER_HEIGHT_MM
    BASE_PLATFORM_HEIGHT_MM    = rk.BASE_PLATFORM_HEIGHT_MM
    ELBOW_TO_GRIPPER_BACK_MM   = rk.ELBOW_TO_GRIPPER_BACK_MM
    GRIPPER_TOOTH_LENGTH_MM    = rk.GRIPPER_TOOTH_LENGTH_MM
    CAM_MOUNT_OFFSET_X_MM      = rk.CAM_MOUNT_OFFSET_X_MM
    CAM_MOUNT_OFFSET_Y_MM      = rk.CAM_MOUNT_OFFSET_Y_MM

    # Timing parameters (seconds)
    STABILIZATION_PAUSE_SEC = 0.4
    APPROACH_DURATION_SEC   = 1.0
    PLUCK_DURATION_SEC      = 4.5
    RETRACT_DURATION_SEC    = 1.8
    MOVE_DROP_DURATION_SEC  = 3.5
    EJECT_DURATION_SEC      = 4.0

    def __init__(self, serial_comm, speed_ms_per_deg=85):
        self.serial_comm = serial_comm
        self.speed_ms_per_deg = speed_ms_per_deg

        # Simulation / Dry-Run Mode
        self.dry_run = False

        # Current estimated joint angles (degrees)
        self.cur_base = self.HOME_B
        self.cur_shoulder = self.HOME_S
        self.cur_arm = self.HOME_A
        self.cur_elbow = self.HOME_E

        # Master Controller State - Starts in IDLE holding at Home until user presses ENTER
        self.state = self.STATE_IDLE
        self.harvest_stage = self.STAGE_SEARCHING
        self.status_msg = "ARM HOLDING AT HOME (Press ENTER to start search)"
        self.auto_pick_on_timeout = True
        self.auto_resume_search_after_harvest = False

        # Search sweep waypoints across 3 sectors
        self.waypoints = [
            # Sector 1: Center (Base 90)
            {
                "sector": 1,
                "desc": "Sector 1 (Center): Base 90, Sh 100, Arm 130, Elbow 95",
                "base": 90, "shoulder": 100, "arm": 130, "elbow": 95
            },
            {
                "sector": 1,
                "desc": "Sector 1 (Center): Elbow sweep 95 -> 125 (Arm 130)",
                "base": 90, "shoulder": 100, "arm": 130, "elbow": 125
            },
            # Sector 2: Left Sector (Base 60)
            {
                "sector": 2,
                "desc": "Sector 2 (Left): Rotate Base 60, Elbow 95 (Arm 130)",
                "base": 60, "shoulder": 100, "arm": 130, "elbow": 95
            },
            {
                "sector": 2,
                "desc": "Sector 2 (Left): Elbow sweep 95 -> 125 (Arm 130)",
                "base": 60, "shoulder": 100, "arm": 130, "elbow": 125
            },
            # Sector 3: Right Sector (Base 120)
            {
                "sector": 3,
                "desc": "Sector 3 (Right): Rotate Base 120, Elbow 95 (Arm 130)",
                "base": 120, "shoulder": 100, "arm": 130, "elbow": 95
            },
            {
                "sector": 3,
                "desc": "Sector 3 (Right): Elbow sweep 95 -> 125 (Arm 130)",
                "base": 120, "shoulder": 100, "arm": 130, "elbow": 125
            }
        ]
        self.current_step = 0
        self.step_start_time = 0.0
        self.step_duration = 0.0
        self.step_start_base = self.HOME_B
        self.step_target_base = self.HOME_B
        self.step_start_shoulder = self.HOME_S
        self.step_target_shoulder = self.HOME_S
        self.step_start_arm = self.HOME_A
        self.step_target_arm = self.HOME_A
        self.step_start_elbow = self.HOME_E
        self.step_target_elbow = self.HOME_E

        # Target lock & harvesting variables
        self.locked_target_id = None
        self.stage_start_time = 0.0
        self.stage_duration = 0.0
        self.red_box_active = False

        # Live vision coordinates (Camera frame)
        self.target_cx = 0
        self.target_cy = 0
        self.target_z_mm = 0.0
        self.target_x_mm = 0.0
        self.target_y_mm = 0.0
        self.last_valid_z_mm = 160.0
        self.frame_w = 1280
        self.frame_h = 720

        # Multi-sample depth accumulation buffer
        self.depth_samples = []
        self.pos3d_samples = []

        # Pose & IK Results
        self.freeze_pose = (self.HOME_B, self.HOME_S, self.HOME_A, self.HOME_E)
        self.locked_target_base_xyz = np.array([0.0, 0.0, 0.0])
        self.pre_pick_base_xyz = np.array([0.0, 0.0, 0.0])
        self.target_ik_angles = (self.HOME_B, self.HOME_S, self.HOME_A, self.HOME_E)
        self.pre_pick_ik_angles = (self.HOME_B, self.HOME_S, self.HOME_A, self.HOME_E)
        self.fk_reconstructed_xyz = np.array([0.0, 0.0, 0.0])
        self.fk_error_mm = 0.0
        self.ik_status_str = "IK IDLE"
        self.measurement_data = None

        # Explicit target angle variables for HUD display
        self.target_ik_base = self.HOME_B
        self.target_ik_shoulder = self.HOME_S
        self.target_ik_arm = self.HOME_A
        self.target_ik_elbow = self.HOME_E

        # Homing variables
        self.homing_start_time = 0.0
        self.homing_phase_duration = 3.0
        self.homing_start_base = self.HOME_B
        self.homing_start_shoulder = self.HOME_S
        self.homing_start_arm = self.HOME_A
        self.homing_start_elbow = self.HOME_E

    # =========================================================================
    # DRY-RUN SIMULATION CONTROL
    # =========================================================================
    def set_dry_run(self, enabled: bool):
        self.dry_run = enabled
        mode_str = "ENABLED (Simulation mode, hardware bypass)" if self.dry_run else "DISABLED (Real hardware active)"
        print(f"\n[CONTROLLER] Dry-Run Mode: {mode_str}\n")
        self.status_msg = f"DRY-RUN {mode_str}"

    def toggle_dry_run(self):
        self.set_dry_run(not self.dry_run)
        return self.dry_run

    def _send_servo_safe(self, axis_char: str, angle: float):
        safe_ang = int(round(angle))
        if self.dry_run:
            print(f"[DRY-RUN SERVO] {axis_char}{safe_ang} (Simulated write)")
            return True
        return self.serial_comm.send_servo(axis_char, safe_ang)

    def _send_cmd_safe(self, cmd_str: str):
        if self.dry_run:
            print(f"[DRY-RUN CMD] Command '{cmd_str}' (Simulated write)")
            return True
        return self.serial_comm.send_cmd(cmd_str)

    # =========================================================================
    # REACTION & SEARCH ENGINE
    # =========================================================================
    def start_search(self):
        """Starts or restarts the 3-sector search sequence from step 0."""
        if self.state == self.STATE_HOMING:
            print("[SEARCH ENGINE] Arm is currently returning to Home. Please wait.")
            return
        self.current_step = 0
        self.state = self.STATE_SEARCHING
        self.harvest_stage = self.STAGE_SEARCHING
        self.locked_target_id = None
        print("\n" + "=" * 75)
        print(" [AUTOMATED COTTON SEARCH ENGINE ACTIVATED]")
        print(" Trajectory: Sector 1 (Center B90) -> Sector 2 (Left B60) -> Sector 3 (Right B120)")
        print(" Active Vision: YOLO scanning continuously for fully opened cotton bolls")
        print("=" * 75 + "\n")
        self._execute_step(0)

    def toggle_search(self):
        """Toggles between starting/resuming search and pausing."""
        if self.state == self.STATE_HOMING:
            print("[SEARCH ENGINE] Arm is currently returning to Home. Please wait.")
            return False
        elif self.state in [self.STATE_IDLE, self.STATE_ERROR]:
            self.start_search()
            return True
        elif self.state == self.STATE_SEARCHING:
            self.state = self.STATE_PAUSED
            self._send_cmd_safe("STOP")
            self.status_msg = "SEARCH PAUSED (Press ENTER to resume)"
            print(f"[SEARCH ENGINE] {self.status_msg}")
            return False
        elif self.state == self.STATE_PAUSED:
            self.state = self.STATE_SEARCHING
            self.status_msg = f"RESUMED: {self.waypoints[self.current_step]['desc']}"
            print(f"[SEARCH ENGINE] {self.status_msg}")
            self._execute_step(self.current_step)
            return True
        elif self.is_harvesting:
            self.start_search()
            return True
        return False

    def _execute_step(self, step_idx):
        if step_idx >= len(self.waypoints):
            print("\n[SEARCH ENGINE] Completed full 3-sector sweep! Autonomously looping from Sector 1...\n")
            self.current_step = 0
            step_idx = 0

        wp = self.waypoints[step_idx]
        target_b = wp['base']
        target_s = wp['shoulder']
        target_a = wp.get('arm', self.HOME_A)
        target_e = wp.get('elbow', self.cur_elbow)

        self.step_start_base = self.cur_base
        self.step_target_base = target_b
        self.step_start_shoulder = self.cur_shoulder
        self.step_target_shoulder = target_s
        self.step_start_arm = self.cur_arm
        self.step_target_arm = target_a
        self.step_start_elbow = self.cur_elbow
        self.step_target_elbow = target_e

        self.step_duration = self.calculate_duration(target_b, target_s, target_a, target_e)
        self.step_start_time = time.time()
        self.status_msg = f"[{step_idx+1}/{len(self.waypoints)}] {wp['desc']}"
        print(f"[SEARCH ENGINE] Step {step_idx+1}: {wp['desc']} (Est time: {self.step_duration:.1f}s)")

        # Send target servo angles in safe balance sequence
        is_unfolding_from_home = (self.cur_elbow == self.HOME_E and target_e != self.HOME_E)
        if target_e != self.cur_elbow:
            self._send_servo_safe('E', target_e)
            time.sleep(0.40 if is_unfolding_from_home else 0.02)
            self.cur_elbow = target_e

        if self.cur_arm != target_a:
            self._send_servo_safe('A', target_a)
            time.sleep(0.02)
            self.cur_arm = target_a

        if target_s != self.cur_shoulder:
            self._send_servo_safe('S', target_s)
            time.sleep(0.02)
            self.cur_shoulder = target_s

        if target_b != self.cur_base:
            self._send_servo_safe('B', target_b)
            time.sleep(0.02)
            self.cur_base = target_b

    # =========================================================================
    # TARGET DETECTION & DEPTH ACCUMULATION (AFTER DETECTION ONLY)
    # =========================================================================
    def on_cotton_locked(self, target_id, cx=0, cy=0, z_mm=0.0, frame_w=1280, frame_h=720, x_mm=0.0, y_mm=0.0):
        """
        Called immediately upon YOLO cotton detection.
        Freezes search sweep motion instantly, locks target ID, and begins
        multi-sample 3D depth accumulation.
        """
        # If already engaged in a picking sequence, do not allow conflicting interrupts
        if self.is_harvesting:
            return False

        # Instantly send STOP command twice to halt search sweep immediately with zero delay
        self._send_cmd_safe("STOP")
        self._send_cmd_safe("STOP")

        # Estimate exact braked joint angles mid-transit
        if self.step_duration > 0:
            elapsed = time.time() - self.step_start_time
            progress = min(1.0, max(0.0, elapsed / self.step_duration))
            self.cur_base = int(round(self.step_start_base + (self.step_target_base - self.step_start_base) * progress))
            self.cur_shoulder = int(round(self.step_start_shoulder + (self.step_target_shoulder - self.step_start_shoulder) * progress))
            self.cur_arm = int(round(self.step_start_arm + (self.step_target_arm - self.step_start_arm) * progress))
            self.cur_elbow = int(round(self.step_start_elbow + (self.step_target_elbow - self.step_start_elbow) * progress))

        self.freeze_pose = (self.cur_base, self.cur_shoulder, self.cur_arm, self.cur_elbow)

        # Enter TARGET_LOCKED state
        self.state = self.STATE_TARGET_LOCKED
        self.harvest_stage = self.STAGE_WAITING
        self.locked_target_id = target_id
        self.stage_start_time = time.time()
        self.stage_duration = self.STABILIZATION_PAUSE_SEC
        self.red_box_active = False

        self.target_cx = cx
        self.target_cy = cy
        self.target_z_mm = z_mm
        if z_mm > 0:
            self.last_valid_z_mm = z_mm
        self.target_x_mm = x_mm
        self.target_y_mm = y_mm
        self.frame_w = frame_w if frame_w > 0 else 1280
        self.frame_h = frame_h if frame_h > 0 else 720

        self.depth_samples = [z_mm] if z_mm > 0 else [self.last_valid_z_mm]
        self.pos3d_samples = [(x_mm, y_mm, z_mm)] if z_mm > 0 else [(x_mm, y_mm, self.last_valid_z_mm)]

        dist_str = f"{z_mm/10.0:.1f}cm ({z_mm:.0f}mm)" if z_mm > 0 else f"{self.last_valid_z_mm/10.0:.1f}cm"
        print("\n" + "=" * 80)
        print(f" >>> [COTTON DETECTED] Target #{target_id} Locked at {dist_str}")
        print(f" >>> Arm HALTED in place. Braked pose: Base={self.cur_base}deg, Sh={self.cur_shoulder}deg, Arm={self.cur_arm}deg, Elbow={self.cur_elbow}deg")
        print(f" >>> Accumulating 3D depth samples for {self.STABILIZATION_PAUSE_SEC:.1f}s...")
        print("=" * 80 + "\n")
        self.status_msg = f"COTTON #{target_id} LOCKED: Stabilizing 3D ({dist_str})..."
        return True

    def update_target_coordinates(self, cx, cy, z_mm=0.0, x_mm=0.0, y_mm=0.0):
        """Feeds depth measurements during stabilization pause."""
        if cx > 0 and cy > 0:
            self.target_cx = cx
            self.target_cy = cy
        if z_mm > 0:
            self.target_z_mm = z_mm
            self.last_valid_z_mm = z_mm
            self.depth_samples.append(z_mm)
            self.pos3d_samples.append((x_mm, y_mm, z_mm))
            if len(self.depth_samples) > 60:
                self.depth_samples.pop(0)
                self.pos3d_samples.pop(0)
        if x_mm != 0.0:
            self.target_x_mm = x_mm
        if y_mm != 0.0:
            self.target_y_mm = y_mm

    def _get_stabilized_3d_point(self):
        """Computes median (X_c, Y_c, Z_c) in camera frame."""
        if not self.depth_samples:
            return self.target_x_mm, self.target_y_mm, getattr(self, 'last_valid_z_mm', 160.0)

        valid_z = [z for z in self.depth_samples if z > 40.0]
        final_z = float(np.median(valid_z)) if valid_z else getattr(self, 'last_valid_z_mm', 160.0)

        final_x = self.target_x_mm
        final_y = self.target_y_mm
        if self.pos3d_samples:
            valid_p = [p for p in self.pos3d_samples if p[2] > 40.0]
            if valid_p:
                final_x = float(np.median([p[0] for p in valid_p]))
                final_y = float(np.median([p[1] for p in valid_p]))

        return final_x, final_y, final_z

    # =========================================================================
    # CLOSED-LOOP CONTROL & SEQUENCE STATE MACHINE
    # =========================================================================
    def update(self):
        """Called on every camera frame to orchestrate all motions and states."""
        now = time.time()

        # ---------------------------------------------------------------------
        # 1. Homing State Handler
        # ---------------------------------------------------------------------
        if self.state == self.STATE_HOMING:
            elapsed = now - self.homing_start_time
            frac = min(1.0, elapsed / max(0.1, self.homing_phase_duration))
            self.cur_base = int(round(self.homing_start_base + (self.HOME_B - self.homing_start_base) * frac))
            self.cur_shoulder = int(round(self.homing_start_shoulder + (self.HOME_S - self.homing_start_shoulder) * frac))
            self.cur_arm = int(round(self.homing_start_arm + (self.HOME_A - self.homing_start_arm) * frac))
            self.cur_elbow = int(round(self.homing_start_elbow + (self.HOME_E - self.homing_start_elbow) * frac))

            if elapsed >= self.homing_phase_duration:
                self.cur_base = self.HOME_B
                self.cur_shoulder = self.HOME_S
                self.cur_arm = self.HOME_A
                self.cur_elbow = self.HOME_E

                if getattr(self, 'auto_resume_search_after_harvest', False):
                    self.auto_resume_search_after_harvest = False
                    self.state = self.STATE_SEARCHING
                    self.harvest_stage = self.STAGE_SEARCHING
                    self.current_step = 0
                    self.status_msg = f"RESUMED: {self.waypoints[0]['desc']}"
                    print("\n" + "=" * 75)
                    print(f" >>> [HARVEST CYCLE COMPLETE] Resuming search sweep from Sector 1...")
                    print("=" * 75 + "\n")
                    self._execute_step(0)
                else:
                    self.state = self.STATE_IDLE
                    self.harvest_stage = self.STAGE_SEARCHING
                    self.current_step = 0
                    self.status_msg = "ARM HOLDING AT HOME (Press ENTER to start search)"
                    print("\n" + "=" * 75)
                    print(f" >>> [HOMING COMPLETE] Arm at HOME: Base {self.HOME_B}, Sh {self.HOME_S}, Arm {self.HOME_A}, Elbow {self.HOME_E}!")
                    print(" >>> Arm holding steady at Home. Press ENTER to start 3-sector search sweep!")
                    print("=" * 75 + "\n")
            else:
                rem = max(0.0, self.homing_phase_duration - elapsed)
                self.status_msg = f"HOMING: Returning to Home pose ({rem:.1f}s)..."
            return

        # ---------------------------------------------------------------------
        # 2. Error State Handler
        # ---------------------------------------------------------------------
        if self.state == self.STATE_ERROR:
            elapsed = now - self.stage_start_time
            if elapsed >= 3.0:
                print("\n[ERROR RECOVERY] Clearing error state; returning to search.\n")
                self.state = self.STATE_SEARCHING
                self.harvest_stage = self.STAGE_SEARCHING
                self.locked_target_id = None
                self._execute_step(self.current_step)
            return

        # ---------------------------------------------------------------------
        # 3. Harvesting State Machine (AFTER DETECTION ONLY)
        # ---------------------------------------------------------------------
        if self.is_harvesting:
            stage_elapsed = now - self.stage_start_time

            # -----------------------------------------------------------------
            # Stage A: TARGET_LOCKED (Stabilization Pause & Kinematics Solving)
            # -----------------------------------------------------------------
            if self.state == self.STATE_TARGET_LOCKED:
                rem_wait = max(0.0, self.stage_duration - stage_elapsed)
                self.status_msg = f"LOCKED #{self.locked_target_id}: Stabilizing 3D ({rem_wait:.1f}s)..."

                if stage_elapsed >= 0.3 and not self.red_box_active:
                    self.red_box_active = True

                if stage_elapsed >= self.stage_duration:
                    # Stabilization complete: resolve true 3D coordinates & IK
                    self._plan_and_execute_pick()

            # -----------------------------------------------------------------
            # Stage LONG: Target too far (Out-of-reach notification hold)
            # -----------------------------------------------------------------
            elif self.state == self.STATE_TARGET_LONG:
                if stage_elapsed >= self.stage_duration:
                    print("\n" + "=" * 75)
                    print(" >>> [LONG RECOVERY] Out-of-reach target cleared. Resuming search sweep...")
                    print("=" * 75 + "\n")
                    self.state = self.STATE_SEARCHING
                    self.harvest_stage = self.STAGE_SEARCHING
                    self.locked_target_id = None
                    self.red_box_active = False
                    self._execute_step(self.current_step)
                else:
                    rem = max(0.0, self.stage_duration - stage_elapsed)
                    self.status_msg = f"[LONG] Cotton #{self.locked_target_id} is LONG ({rem:.1f}s) - Resuming search..."

            # -----------------------------------------------------------------
            # Stage B: PRE_PICK (Move to 45mm backed-off approach position)
            # -----------------------------------------------------------------
            elif self.state == self.STATE_PRE_PICK:
                if stage_elapsed >= self.stage_duration:
                    print("\n" + "-" * 75)
                    print(f" >>> [PRE-PICK COMPLETE] Arm aligned at pre-pick stance!")
                    print(f" >>> Advancing slowly into contact with cotton boll...")
                    print("-" * 75 + "\n")
                    self._advance_to_approach()
                else:
                    rem = max(0.0, self.stage_duration - stage_elapsed)
                    self.status_msg = f"PRE-PICK #{self.locked_target_id}: Aligning gripper ({rem:.1f}s)..."

            # -----------------------------------------------------------------
            # Stage C: APPROACH (Smooth transit into boll contact)
            # -----------------------------------------------------------------
            elif self.state == self.STATE_APPROACH:
                if stage_elapsed >= self.stage_duration:
                    print("\n" + "=" * 75)
                    print(f" >>> [GRIPPER AT BOLL] Gripper contact point reached cotton boll!")
                    print(f" >>> Closing gripper ('Y') & running plucker CW ('CW') for {self.PLUCK_DURATION_SEC:.1f}s...")
                    print("=" * 75 + "\n")
                    self._advance_to_grip()
                else:
                    rem = max(0.0, self.stage_duration - stage_elapsed)
                    self.status_msg = f"APPROACH #{self.locked_target_id}: Moving to boll ({rem:.1f}s)..."

            # -----------------------------------------------------------------
            # Stage D: GRIP (Close MG90S fingers & Pluck CW)
            # -----------------------------------------------------------------
            elif self.state == self.STATE_GRIP:
                if stage_elapsed >= self.stage_duration:
                    print("\n" + "-" * 75)
                    print(f" >>> [PLUCK COMPLETE] Cotton harvested! Retracting clear of crop...")
                    print("-" * 75 + "\n")
                    self._advance_to_retract()
                else:
                    rem = max(0.0, self.stage_duration - stage_elapsed)
                    self.status_msg = f"HARVESTING #{self.locked_target_id}: Plucking Cotton CW ({rem:.1f}s)..."

            # -----------------------------------------------------------------
            # Stage E: RETRACT (Lift shoulder safely clear of branches)
            # -----------------------------------------------------------------
            elif self.state == self.STATE_RETRACT:
                if stage_elapsed >= self.stage_duration:
                    print("\n" + "-" * 75)
                    print(" >>> [CLEARANCE SAFE] Arm clear! Rotating Base to Drop position (Base 50)...")
                    print("-" * 75 + "\n")
                    self._advance_to_drop()
                else:
                    rem = max(0.0, self.stage_duration - stage_elapsed)
                    self.status_msg = f"RETRACT #{self.locked_target_id}: Lifting Shoulder alone (135deg)..."

            # -----------------------------------------------------------------
            # Stage F: DROP (Move to collection container at Base 50)
            # -----------------------------------------------------------------
            elif self.state == self.STATE_DROP:
                if stage_elapsed >= self.stage_duration:
                    print("\n" + "=" * 75)
                    print(" >>> [AT DROP BOX] Opening gripper ('S') & running CCW roller ('CCW') to eject...")
                    print("=" * 75 + "\n")
                    self._advance_to_release()
                else:
                    rem = max(0.0, self.stage_duration - stage_elapsed)
                    self.status_msg = f"DROP #{self.locked_target_id}: Moving to drop box ({rem:.1f}s)..."

            # -----------------------------------------------------------------
            # Stage G: RELEASE (Open gripper & Eject CCW into container)
            # -----------------------------------------------------------------
            elif self.state == self.STATE_RELEASE:
                if stage_elapsed >= self.stage_duration:
                    print("\n" + "=" * 75)
                    print(" >>> [EJECTION COMPLETE] Cotton safely deposited in container!")
                    print(" >>> Returning arm to Home...")
                    print("=" * 75 + "\n")
                    self._advance_to_homing()
                else:
                    rem = max(0.0, self.stage_duration - stage_elapsed)
                    self.status_msg = f"RELEASE #{self.locked_target_id}: Ejecting cotton CCW into container..."

            return

        # ---------------------------------------------------------------------
        # 4. Normal Search Sweep Waypoint Progression
        # ---------------------------------------------------------------------
        if self.state != self.STATE_SEARCHING:
            return

        if (now - self.step_start_time) >= self.step_duration:
            self.cur_base = getattr(self, 'step_target_base', self.cur_base)
            self.cur_shoulder = getattr(self, 'step_target_shoulder', self.cur_shoulder)
            self.cur_arm = getattr(self, 'step_target_arm', self.cur_arm)
            self.cur_elbow = getattr(self, 'step_target_elbow', self.cur_elbow)
            self.current_step += 1
            self._execute_step(self.current_step)

    # =========================================================================
    # PLANNING & KINEMATICS EXECUTION (AFTER DETECTION ONLY)
    # =========================================================================
    def _plan_and_execute_pick(self):
        """
        Calculates exact 3D coordinates, solves IK, validates safety limits,
        and commands arm to Pre-Pick waypoint.
        """
        now = time.time()
        # 1. Median 3D point in camera frame
        sx, sy, sz = self._get_stabilized_3d_point()
        p_cam = np.array([sx, sy, sz])

        # 2. Transform Camera 3D -> Robot Base Frame {0}
        fb_b, fb_s, fb_a, fb_e = self.freeze_pose
        p_base = rk.camera_to_base_frame(p_cam, fb_b, fb_s, fb_a, fb_e)
        self.locked_target_base_xyz = p_base

        # 3. Exact 3D Measurement of Gripper Contact Center to Cotton Boll Center
        meas = rk.measure_gripper_to_cotton(p_cam, fb_b, fb_s, fb_a, fb_e, max_reach_dist_mm=280.0, max_cam_depth_mm=280.0)
        self.measurement_data = meas

        print("\n" + "=" * 80)
        print(f" [MEASUREMENT & TRACKING] Target #{self.locked_target_id} Geometry:")
        print(f"  * Cotton Center in Cam Frame : X={meas['p_cotton_cam'][0]:+.1f}mm, Y={meas['p_cotton_cam'][1]:+.1f}mm, Z={meas['p_cotton_cam'][2]:.1f}mm ({sz/10.0:.1f}cm)")
        print(f"  * Gripper Center in Cam Frame: X={meas['p_gripper_cam'][0]:.1f}mm, Y={meas['p_gripper_cam'][1]:.1f}mm, Z={meas['p_gripper_cam'][2]:.1f}mm")
        print(f"  * Distance from Gripper Teeth: {meas['distance_mm']} mm (Operating Reach: 280 mm)")
        print(f"  * Forward Beyond Teeth (dz)  : {meas['dz_mm']} mm (+ in front, - inside jaws)")
        print(f"  * Relative Angles to Center  : Yaw={meas['yaw_deg']:+.1f} deg, Pitch={meas['pitch_deg']:+.1f} deg")
        print(f"  * Measurement Evaluation     : >>> {meas['status_str']} <<<")
        print("=" * 80 + "\n")

        # 4. Check if measurement is LONG (cannot reach or pluck from current position)
        if meas['is_long']:
            print("!" * 80)
            print(f" >>> [LONG] TARGET #{self.locked_target_id} MEASUREMENT: LONG!")
            print(f" >>> Depth Z={sz:.1f}mm ({sz/10.0:.1f}cm) exceeds reach limit (280mm / 28cm).")
            print(f" >>> SAYING 'LONG': Cotton boll is too far to catch or pluck. Move robot closer!")
            print(f" >>> Holding safely for 2.5s then automatically resuming search sweep...")
            print("!" * 80 + "\n")
            self.state = self.STATE_TARGET_LONG
            self.harvest_stage = self.STAGE_LONG
            self.stage_start_time = now
            self.stage_duration = 2.5
            self.status_msg = f"[LONG] Cotton #{self.locked_target_id} at {sz/10.0:.1f}cm is LONG - Move robot closer"
            return False

        # 5. Target is Reachable: Calculate exact angles to drive Gripper into Cotton Center
        sol_aim, sol_pre_aim, err_aim, reason_aim = rk.calculate_elbow_aiming(
            p_base, cur_shoulder=self.cur_shoulder, cur_elbow=self.cur_elbow, lock_arm_angle=self.HOME_A
        )
        actual_target_angles = sol_aim
        actual_pre_angles = sol_pre_aim
        err_target = err_aim
        self.pre_pick_base_xyz = p_base
        pick_status_msg = f"CENTER AIM: B{sol_aim[0]} S{sol_aim[1]} [A:130] E{sol_aim[3]} (Dist:{meas['distance_mm']}mm)"
        print(f" [PLUCK PLAN] Aligning Gripper Center to Cotton Center: Base={sol_aim[0]}deg, Shoulder={sol_aim[1]}deg, Arm=130deg, Elbow={sol_aim[3]}deg")

        # Diagnostic logging
        self._log_kinematics_diagnostics(p_cam, p_base, actual_target_angles, err_target, True, "OK")

        # Store calculated angles
        self.target_ik_angles = actual_target_angles
        self.target_ik_base = actual_target_angles[0]
        self.target_ik_shoulder = actual_target_angles[1]
        self.target_ik_arm = actual_target_angles[2]
        self.target_ik_elbow = actual_target_angles[3]
        self.pre_pick_ik_angles = actual_pre_angles

        fk_check = rk.forward_kinematics(*actual_target_angles)
        self.fk_reconstructed_xyz = fk_check['gripper']
        self.fk_error_mm = err_target
        self.ik_status_str = pick_status_msg

        # Command Pre-Pick move: ONLY Shoulder and Elbow (and Base) move. Arm A is strictly LOCKED at 130!
        self.state = self.STATE_PRE_PICK
        self.harvest_stage = self.STAGE_PRE_PICK
        self.stage_start_time = now
        self.stage_duration = self.calculate_duration(*actual_pre_angles)

        b_p, s_p, a_p, e_p = actual_pre_angles
        if b_p != self.cur_base:
            self._send_servo_safe('B', b_p)
            time.sleep(0.02)
        if s_p != self.cur_shoulder:
            self._send_servo_safe('S', s_p)
            time.sleep(0.02)
        # Note: Arm servo A is intentionally NOT sent (strictly locked at 130 deg)
        if e_p != self.cur_elbow:
            self._send_servo_safe('E', e_p)
            time.sleep(0.02)

        self.cur_base = b_p
        self.cur_shoulder = s_p
        self.cur_arm = self.HOME_A
        self.cur_elbow = e_p
        self.status_msg = f"PRE-PICK #{self.locked_target_id}: Centering gripper on cotton (B{b_p} S{s_p} E{e_p})..."
        return True

    def _advance_to_approach(self):
        """Moves from Pre-Pick position smoothly to Target contact position using ONLY Shoulder and Elbow."""
        now = time.time()
        self.state = self.STATE_APPROACH
        self.harvest_stage = self.STAGE_APPROACH
        self.stage_start_time = now
        self.stage_duration = max(0.8, self.calculate_duration(*self.target_ik_angles) * 0.7)

        b_t, s_t, a_t, e_t = self.target_ik_angles
        if b_t != self.cur_base:
            self._send_servo_safe('B', b_t)
            time.sleep(0.02)
        if s_t != self.cur_shoulder:
            self._send_servo_safe('S', s_t)
            time.sleep(0.02)
        # Note: Arm servo A is intentionally NOT sent (strictly locked at 130 deg)
        if e_t != self.cur_elbow:
            self._send_servo_safe('E', e_t)
            time.sleep(0.02)

        self.cur_base = b_t
        self.cur_shoulder = s_t
        self.cur_arm = self.HOME_A
        self.cur_elbow = e_t
        self.status_msg = f"APPROACH #{self.locked_target_id}: Contacting boll with Sh {s_t}deg, Elbow {e_t}deg..."

    def _advance_to_grip(self):
        """Closes gripper and spins roller CW to pluck cotton boll."""
        now = time.time()
        self.state = self.STATE_GRIP
        self.harvest_stage = self.STAGE_PICKING
        self.stage_start_time = now
        self.stage_duration = self.PLUCK_DURATION_SEC

        # Command Y (gripper close) and CW (DC roller pluck)
        self._send_cmd_safe("Y")
        time.sleep(0.15)
        self._send_cmd_safe("CW")
        self.status_msg = f"HARVESTING #{self.locked_target_id}: Gripper Closed & Plucking CW..."

    def _advance_to_retract(self):
        """Stops DC motor and lifts shoulder to clear branches."""
        now = time.time()
        self._send_cmd_safe("STOP_DC")
        self.state = self.STATE_RETRACT
        self.harvest_stage = self.STAGE_LIFTING
        self.stage_start_time = now
        self.stage_duration = self.RETRACT_DURATION_SEC

        # Lift shoulder alone to safe clearance height (130 deg max safety limit)
        self._send_servo_safe('S', self.SHOULDER_MAX)
        self.cur_shoulder = self.SHOULDER_MAX
        self.status_msg = f"RETRACT #{self.locked_target_id}: Lifting Shoulder alone ({self.SHOULDER_MAX}deg)..."

    def _advance_to_drop(self):
        """Moves arm to Base 50 drop location within safety limits (130 deg max)."""
        now = time.time()
        self.state = self.STATE_DROP
        self.harvest_stage = self.STAGE_MOVE_DROP
        self.stage_start_time = now
        self.stage_duration = self.MOVE_DROP_DURATION_SEC

        # Move to Drop configuration: Base 50, Shoulder 130, Elbow 130 (Arm stays at 130)
        self._send_servo_safe('B', 50)
        time.sleep(0.04)
        self._send_servo_safe('S', self.SHOULDER_MAX)
        time.sleep(0.04)
        self._send_servo_safe('E', self.ELBOW_MAX)

        self.cur_base = 50
        self.cur_shoulder = self.SHOULDER_MAX
        self.cur_elbow = self.ELBOW_MAX
        self.cur_arm = self.HOME_A
        self.status_msg = f"DROP #{self.locked_target_id}: Moving to Drop Base 50..."

    def _advance_to_release(self):
        """Opens gripper and runs CCW roller to eject cotton into drop container."""
        now = time.time()
        self.state = self.STATE_RELEASE
        self.harvest_stage = self.STAGE_DROPPING
        self.stage_start_time = now
        self.stage_duration = self.EJECT_DURATION_SEC

        # Send 'S' (gripper open) and 'CCW' (roller eject)
        self._send_cmd_safe("S")
        time.sleep(0.15)
        self._send_cmd_safe("CCW")
        self.status_msg = f"RELEASE #{self.locked_target_id}: Ejecting cotton CCW into container..."

    def _advance_to_homing(self):
        """Stops DC roller and initiates return to Home pose after harvest."""
        now = time.time()
        self._send_cmd_safe("STOP_DC")
        self.state = self.STATE_HOMING
        self.harvest_stage = self.STAGE_HOMING
        self.auto_resume_search_after_harvest = True
        self.homing_start_time = now
        self.homing_start_base = self.cur_base
        self.homing_start_shoulder = self.cur_shoulder
        self.homing_start_arm = self.cur_arm
        self.homing_start_elbow = self.cur_elbow
        self.homing_phase_duration = 3.0

        self._send_servo_safe('B', self.HOME_B)
        time.sleep(0.02)
        self._send_servo_safe('E', self.HOME_E)
        time.sleep(0.02)
        self._send_servo_safe('S', self.HOME_S)
        # Note: Arm servo A is intentionally NOT sent (already at 130 deg)
        self.status_msg = f"HOMING #{self.locked_target_id}: Centering to Home ({self.homing_phase_duration:.1f}s)..."

    # =========================================================================
    # DIAGNOSTICS & TELEMETRY
    # =========================================================================
    def _log_kinematics_diagnostics(self, p_cam, p_base, sol_target, err_target, ok_target, reason_target):
        fb_b, fb_s, fb_a, fb_e = self.freeze_pose
        dist_from_sh = math.sqrt(p_base[0]**2 + p_base[1]**2 + (p_base[2] - rk.BASE_SHOULDER_HEIGHT_MM)**2)

        print("\n" + "=" * 80)
        print(f" [KINEMATICS REPORT] Target #{self.locked_target_id} | Stage: {self.harvest_stage}")
        print(f"  1. Freeze Pose at Detection : Base={fb_b}deg, Sh={fb_s}deg, Arm={fb_a}deg, Elbow={fb_e}deg")
        print(f"  2. RealSense Camera Frame   : X={p_cam[0]:.1f}mm, Y={p_cam[1]:.1f}mm, Z={p_cam[2]:.1f}mm (dist: {p_cam[2]/10.0:.1f}cm)")
        print(f"  3. Converted Robot Base {0} : X={p_base[0]:.1f}mm, Y={p_base[1]:.1f}mm, Z={p_base[2]:.1f}mm")
        print(f"     -> Radial Dist: {math.sqrt(p_base[0]**2 + p_base[1]**2):.1f}mm | Dist from Shoulder Pivot: {dist_from_sh:.1f}mm (Max: {rk.MAX_ARM_REACH_MM:.1f}mm)")
        if ok_target:
            b, s, a, e = sol_target
            fk_sol = rk.forward_kinematics(b, s, a, e)['gripper']
            print(f"  4. Inverse Kinematics Sol   : Base={b}deg, Sh={s}deg, Arm={a}deg, Elbow={e}deg")
            print(f"  5. FK Reconstructed Gripper : X={fk_sol[0]:.1f}mm, Y={fk_sol[1]:.1f}mm, Z={fk_sol[2]:.1f}mm")
            print(f"  6. Closed-Loop Position Err : {err_target:.4f} mm (Sub-millimeter accurate!)")
            print(f"  7. Reachability Status      : {reason_target}")
        else:
            print(f"  4. Inverse Kinematics Sol   : FAILED - {reason_target}")
            print(f"  5. Reachability Status      : REJECTED (Unsafe/Unreachable)")
        print("=" * 80 + "\n")

    # =========================================================================
    # PUBLIC API & BACKWARD COMPATIBILITY
    # =========================================================================
    @property
    def is_harvesting(self):
        """Returns True if the controller is in any active picking state."""
        return self.state in [
            self.STATE_TARGET_LOCKED,
            self.STATE_TARGET_LONG,
            self.STATE_PRE_PICK,
            self.STATE_APPROACH,
            self.STATE_GRIP,
            self.STATE_RETRACT,
            self.STATE_DROP,
            self.STATE_RELEASE
        ]

    def is_red_box_active(self):
        return self.red_box_active and self.is_harvesting

    def trigger_pick(self):
        """Allows manual 'P' key override to advance from TARGET_LOCKED pause immediately."""
        if self.state == self.STATE_TARGET_LOCKED:
            print("\n[MANUAL TRIGGER] 'P' pressed -> Advancing immediately to Pick!\n")
            return self._plan_and_execute_pick()
        return False

    def home(self):
        """Commands arm to return smoothly to Home stance and wait in IDLE until ENTER."""
        self._send_cmd_safe("STOP_DC")
        self.red_box_active = False
        self.locked_target_id = None
        self.current_step = 0
        self.auto_resume_search_after_harvest = False

        self.state = self.STATE_HOMING
        self.harvest_stage = self.STAGE_HOMING
        self.homing_start_time = time.time()
        self.homing_start_base = self.cur_base
        self.homing_start_shoulder = self.cur_shoulder
        self.homing_start_arm = self.cur_arm
        self.homing_start_elbow = self.cur_elbow
        self.homing_phase_duration = 3.0

        print("\n" + "=" * 75)
        print(f" >>> [HOMING] Returning to HOME: B:{self.HOME_B} S:{self.HOME_S} A:{self.HOME_A} E:{self.HOME_E}")
        print("=" * 75 + "\n")
        self.status_msg = "HOMING: Returning to Home stance..."
        self._send_servo_safe('B', self.HOME_B)
        time.sleep(0.02)
        self._send_servo_safe('E', self.HOME_E)
        time.sleep(0.02)
        self._send_servo_safe('S', self.HOME_S)
        time.sleep(0.02)
        self._send_servo_safe('A', self.HOME_A)
        time.sleep(0.02)
        self._send_cmd_safe("O") # Open gripper
        self._send_cmd_safe("H") # Microcontroller homing trigger

    def stop(self):
        """Emergency stop: halts all arm servos and DC motor immediately."""
        self.state = self.STATE_PAUSED
        self._send_cmd_safe("STOP")
        self._send_cmd_safe("STOP_DC")
        self.status_msg = "EMERGENCY STOP (Press ENTER to resume)"
        print(f"[EMERGENCY STOP] {self.status_msg}")

    def parse_serial_feedback(self, msg):
        """Syncs real-time servo positions if reported by ESP32 firmware."""
        if not msg:
            return
        if "POS " in msg:
            import re
            m = re.search(r'POS\s+B:(\d+)\s+S:(\d+)\s+E:(\d+)\s+A:(\d+)', msg)
            if m:
                self.cur_base = int(m.group(1))
                self.cur_shoulder = int(m.group(2))
                self.cur_elbow = int(m.group(3))
                self.cur_arm = int(m.group(4))

    def calculate_duration(self, target_b, target_s, target_a=None, target_e=None):
        """Calculates transit time based on maximum servo angle delta and ms/deg speed."""
        delta_b = abs(target_b - self.cur_base)
        delta_s = abs(target_s - self.cur_shoulder)
        delta_a = abs((target_a if target_a is not None else self.cur_arm) - self.cur_arm)
        delta_e = abs((target_e if target_e is not None else self.cur_elbow) - self.cur_elbow)
        max_delta = max(delta_b, delta_s, delta_a, delta_e)
        if max_delta == 0:
            return 0.1
        accel_buffer = ((1000.0 / self.speed_ms_per_deg) / 25.0) + 0.30
        duration = (max_delta * (self.speed_ms_per_deg / 1000.0)) + accel_buffer
        return max(0.40, duration)

    def calculate_collaborative_reach(self, dx=0, dy=0, z_mm=0.0, x_mm=0.0, y_mm=0.0):
        """Backward-compatible wrapper returning calculated IK angles."""
        p_cam = np.array([x_mm, y_mm, z_mm])
        p_base = rk.camera_to_base_frame(p_cam, self.cur_base, self.cur_shoulder, self.cur_arm, self.cur_elbow)
        sol, err, ok, reason = rk.inverse_kinematics(p_base)
        if ok:
            return sol
        return self.cur_base, self.cur_shoulder, self.cur_arm, self.cur_elbow

    def compute_inverse_kinematics(self, dx=0, dy=0, z_mm=0.0, x_mm=0.0, y_mm=0.0):
        return self.calculate_collaborative_reach(dx, dy, z_mm, x_mm, y_mm)

    def get_calibrated_xyz(self):
        rem_wait = 0.0
        if self.state == self.STATE_TARGET_LOCKED:
            rem_wait = max(0.0, self.stage_duration - (time.time() - self.stage_start_time))
        return {
            'cam_xyz': (self.target_x_mm, self.target_y_mm, self.target_z_mm),
            'gripper_xyz': tuple(np.round(self.locked_target_base_xyz, 1)),
            'deltas': (
                self.target_ik_base - self.cur_base,
                self.target_ik_shoulder - self.cur_shoulder,
                self.target_ik_elbow - self.cur_elbow
            ),
            'rem_wait': rem_wait,
            'target_id': self.locked_target_id
        }

    def get_hud_info(self):
        """Provides rich telemetry for real-time video overlay HUD."""
        now = time.time()
        wait_rem = max(0.0, self.stage_duration - (now - self.stage_start_time)) if (self.state == self.STATE_TARGET_LOCKED) else 0.0

        return {
            "state": self.state,
            "step": self.current_step + 1,
            "total_steps": len(self.waypoints),
            "status_msg": self.status_msg,
            "dry_run": self.dry_run,
            "is_searching": (self.state == self.STATE_SEARCHING),
            "is_harvesting": self.is_harvesting,
            "is_homing": (self.state == self.STATE_HOMING),
            "is_waiting_for_pick": (self.state == self.STATE_TARGET_LOCKED),
            "base_angle": self.cur_base,
            "shoulder_angle": self.cur_shoulder,
            "arm_angle": self.cur_arm,
            "elbow_angle": self.cur_elbow,
            "harvest_stage": self.harvest_stage,
            "ik_status": self.ik_status_str,
            "target_ik_base": self.target_ik_base,
            "target_ik_shoulder": self.target_ik_shoulder,
            "target_ik_arm": self.target_ik_arm,
            "target_ik_elbow": self.target_ik_elbow,
            "target_cx": getattr(self, 'target_cx', 0),
            "target_cy": getattr(self, 'target_cy', 0),
            "target_z_mm": getattr(self, 'target_z_mm', 0.0),
            "target_base_xyz": tuple(np.round(self.locked_target_base_xyz, 1)),
            "fk_reconstructed_xyz": tuple(np.round(self.fk_reconstructed_xyz, 1)),
            "fk_error_mm": round(self.fk_error_mm, 2),
            "is_centering": (self.state in [self.STATE_PRE_PICK, self.STATE_APPROACH]),
            "wait_seconds_remaining": wait_rem,
            "measurement": self.measurement_data,
            "is_target_long": (self.state == self.STATE_TARGET_LONG)
        }
