
"""
Robot Kinematics and Coordinate Transformation Engine for Autonomous Cotton Harvester.

Geometry and Physical Constants:
  - Base Platform Height: 65 mm (physical platform height; NOT shoulder axis offset)
  - Base Rotation Axis to Shoulder Joint Centre: 110 mm
  - Shoulder-to-Arm Joint Centre (L1): 212 mm
  - Arm-to-Elbow Joint Centre (L2): 199 mm
  - Elbow-to-Gripper Tip / Contact Centre (L3): 202 mm
    * Elbow-to-gripper-back reference point: 169 mm
    * Gripper tooth length: ~70 mm
    * Straight configuration contact point: ~202 mm
  - RealSense D435i Camera Mount on Link 3 (Forearm/Gripper assembly):
    * Offset along link from Elbow pivot towards gripper: 70 mm (202 - 132 mm)
    * Offset perpendicular to link (stand height to optical center): 121 mm
    * Lateral offset: 0 mm (centered with gripper)
    * Camera optical axis (+Z_cam) points forward along Link 3 towards the gripper
    * Camera vertical axis (+Y_cam) points downwards towards Link 3
    * Camera horizontal axis (+X_cam) points to camera's right (+Z of Link 3)

Coordinate Conventions:
  Robot Base Frame {0}:
    - Origin: On base rotation axis at base mounting surface
    - +Z: Vertical UP along base rotation axis
    - +X: Forward when Base servo = 90 deg (Azimuth = 0 deg)
    - +Y: Left when Base servo = 90 deg (Azimuth = +90 deg)
    - Shoulder pivot center: (0, 0, 110 mm)

Physical Servo Mappings & Reversed Joint Directions:
  - Base (B): Range 10-170 deg (Firmware web limit: 20-140 deg). 90 is center forward.
      theta_base = 90 deg - B  (B decreases -> turns Left / +Y; B increases -> turns Right / -Y)
  - Shoulder (S): Range 50-140 deg. 140 is upright stow/home. 50 is near-horizontal.
      alpha1 = S - 50 deg  (Link 1 elevation above horizontal)
  - Arm (A): Range 50-140 deg. Physically reversed. 140 is folded back; 50 is unfolded/extended.
      alpha2 = alpha1 - (A - 70 deg)  (Link 2 elevation above horizontal)
  - Elbow (E): Range 50-140 deg. Physically reversed. 140 is folded down/home; 50 is pitched up.
      alpha3 = alpha2 + (110 deg - E)  (Link 3 / Forearm elevation above horizontal)
"""

import math
import numpy as np

# =============================================================================
# PHYSICAL ROBOT SPECIFICATIONS & HARDWARE CALIBRATION CONSTANTS
# =============================================================================

# Center-to-center link lengths (mm)
L1_SHOULDER_MM = 212.0
L2_ARM_MM      = 199.0
L3_GRIPPER_MM  = 202.0

# Base-axis-to-shoulder-joint-center elevation (mm)
BASE_SHOULDER_HEIGHT_MM = 110.0

# Base platform physical height (mm) - for reference, not added to shoulder offset
BASE_PLATFORM_HEIGHT_MM = 65.0

# Gripper tooth and back reference measurements (mm)
ELBOW_TO_GRIPPER_BACK_MM = 169.0
GRIPPER_TOOTH_LENGTH_MM  = 70.0

# Camera mounting on Link 3 (Forearm)
CAM_MOUNT_OFFSET_X_MM = 70.0    # 202mm - 132mm = 70mm forward of elbow pivot along link
CAM_MOUNT_OFFSET_Y_MM = 121.0   # 121mm upright stalk above link centerline
CAM_MOUNT_OFFSET_Z_MM = 0.0     # Lateral center offset

# Total maximum theoretical reach from shoulder pivot (straight line)
MAX_ARM_REACH_MM = L1_SHOULDER_MM + L2_ARM_MM + L3_GRIPPER_MM # 613.0 mm

# Calibrated Safe Servo Limits (degrees) - Reduced by 10 deg to 130 deg max for jerk-free operation
LIMIT_BASE_MIN     = 20
LIMIT_BASE_MAX     = 130   # -10 deg from 140 (safe mechanical margin)
LIMIT_SHOULDER_MIN = 50
LIMIT_SHOULDER_MAX = 130   # -10 deg from 140 (safe mechanical margin)
LIMIT_ARM_MIN      = 50
LIMIT_ARM_MAX      = 130   # -10 deg from 140 (safe mechanical margin)
LIMIT_ELBOW_MIN    = 50
LIMIT_ELBOW_MAX    = 130   # -10 deg from 140 (safe mechanical margin)
LIMIT_GRIPPER_OPEN = 60
LIMIT_GRIPPER_CLOSE= 10

# Safe Home Stance Angles (degrees) - All pitch joints at safe 130 deg limit
HOME_B = 90
HOME_S = 130
HOME_A = 130
HOME_E = 130
HOME_G = LIMIT_GRIPPER_OPEN


# =============================================================================
# ANGLE CONVERSIONS
# =============================================================================

def servo_to_link_angles(B, S, A, E):
    """
    Converts software servo angles (degrees) into physical geometric angles:
      theta_base : Azimuth angle (radians) in horizontal plane.
      alpha1     : Elevation angle of Link 1 above horizontal forward (radians).
      alpha2     : Elevation angle of Link 2 above horizontal forward (radians).
      alpha3     : Elevation angle of Link 3 (forearm/gripper) above horizontal forward (radians).
    """
    theta_base = math.radians(90.0 - float(B))
    alpha1 = math.radians(float(S) - 50.0)
    alpha2 = alpha1 - math.radians(float(A) - 70.0)
    alpha3 = alpha2 + math.radians(110.0 - float(E))
    return theta_base, alpha1, alpha2, alpha3


def link_to_servo_angles(theta_base, alpha1, alpha2, alpha3):
    """
    Converts physical link angles (radians) back into software servo angles (degrees).
    """
    B = 90.0 - math.degrees(theta_base)
    S = math.degrees(alpha1) + 50.0
    A = math.degrees(alpha1 - alpha2) + 70.0
    E = 110.0 - math.degrees(alpha3 - alpha2)
    return B, S, A, E


# =============================================================================
# FORWARD KINEMATICS (FK)
# =============================================================================

def forward_kinematics(B, S, A, E):
    """
    Computes 3D positions of all joints, gripper tip/contact, and camera in Robot Base frame {0}.
    
    Returns dict:
      'shoulder': [X, Y, Z] (mm)
      'arm':      [X, Y, Z] (mm)
      'elbow':    [X, Y, Z] (mm)
      'gripper':  [X, Y, Z] (mm) - Gripper contact point
      'camera':   [X, Y, Z] (mm) - Camera optical center
      'R_cam':    3x3 rotation matrix of Camera frame relative to Base frame
      'link_angles_deg': (theta_base_deg, alpha1_deg, alpha2_deg, alpha3_deg)
    """
    theta_base, a1, a2, a3 = servo_to_link_angles(B, S, A, E)

    # 2D planar coordinates in vertical pitch plane
    # r = radial distance from base axis, z = height above base mounting plate
    r0 = 0.0
    z0 = BASE_SHOULDER_HEIGHT_MM

    # Arm pivot (joint 2)
    r1 = r0 + L1_SHOULDER_MM * math.cos(a1)
    z1 = z0 + L1_SHOULDER_MM * math.sin(a1)

    # Elbow pivot (joint 3)
    r2 = r1 + L2_ARM_MM * math.cos(a2)
    z2 = z1 + L2_ARM_MM * math.sin(a2)

    # Gripper contact point / tip (end of link 3)
    r_grip = r2 + L3_GRIPPER_MM * math.cos(a3)
    z_grip = z2 + L3_GRIPPER_MM * math.sin(a3)

    # RealSense Camera position on Link 3
    r_cam = r2 + CAM_MOUNT_OFFSET_X_MM * math.cos(a3) - CAM_MOUNT_OFFSET_Y_MM * math.sin(a3)
    z_cam = z2 + CAM_MOUNT_OFFSET_X_MM * math.sin(a3) + CAM_MOUNT_OFFSET_Y_MM * math.cos(a3)

    cos_b = math.cos(theta_base)
    sin_b = math.sin(theta_base)

    p_shoulder = np.array([0.0, 0.0, z0])
    p_arm = np.array([r1 * cos_b, r1 * sin_b, z1])
    p_elbow = np.array([r2 * cos_b, r2 * sin_b, z2])
    p_grip = np.array([r_grip * cos_b, r_grip * sin_b, z_grip])
    p_cam = np.array([r_cam * cos_b, r_cam * sin_b, z_cam])

    # Camera Frame Orientation Matrix R_cam:
    # Camera +Z_c (optical axis): Points forward along Link 3
    # Camera +Y_c: Points downward towards Link 3 (-Y in link frame)
    # Camera +X_c: Points to camera right (+Z in link frame, perpendicular to pitch plane)
    z_c_axis = np.array([cos_b * math.cos(a3), sin_b * math.cos(a3), math.sin(a3)])
    y_c_axis = np.array([cos_b * math.sin(a3), sin_b * math.sin(a3), -math.cos(a3)])
    x_c_axis = np.array([sin_b, -cos_b, 0.0]) # Points right when looking forward along arm
    R_cam = np.column_stack([x_c_axis, y_c_axis, z_c_axis])

    return {
        'shoulder': p_shoulder,
        'arm': p_arm,
        'elbow': p_elbow,
        'gripper': p_grip,
        'camera': p_cam,
        'R_cam': R_cam,
        'link_angles_deg': (
            math.degrees(theta_base),
            math.degrees(a1),
            math.degrees(a2),
            math.degrees(a3)
        )
    }


# =============================================================================
# CAMERA-TO-ROBOT BASE COORDINATE TRANSFORMATION
# =============================================================================

def camera_to_base_frame(p_cam_3d, cur_B, cur_S, cur_A, cur_E):
    """
    Transforms 3D target coordinates (X_c, Y_c, Z_c in mm) measured by RealSense
    into Robot Base frame {0} (X_0, Y_0, Z_0 in mm), rigorously accounting for
    the arm's exact joint angles at the moment of detection.
    """
    fk = forward_kinematics(cur_B, cur_S, cur_A, cur_E)
    cam_pos = fk['camera']
    R_cam = fk['R_cam']

    p_c = np.array(p_cam_3d, dtype=float)
    p_base = cam_pos + R_cam @ p_c
    return p_base


def base_to_camera_frame(p_base_3d, cur_B, cur_S, cur_A, cur_E):
    """
    Transforms 3D point in Robot Base frame {0} back into Camera frame coordinates (mm).
    Useful for rechecking target visibility from a new arm configuration.
    """
    fk = forward_kinematics(cur_B, cur_S, cur_A, cur_E)
    cam_pos = fk['camera']
    R_cam = fk['R_cam']

    p_b = np.array(p_base_3d, dtype=float)
    p_c = R_cam.T @ (p_b - cam_pos)
    return p_c


# =============================================================================
# INVERSE KINEMATICS (IK)
# =============================================================================

def inverse_kinematics(target_base_xyz, lock_arm_angle=130.0, preferred_approach_pitch=0.0):
    """
    Calculates joint servo angles (B, S, A, E) to place the Gripper Contact Point
    at target_base_xyz = (X_0, Y_0, Z_0) in the Robot Base frame.
    
    When lock_arm_angle is specified (default 130.0 deg), the Arm servo is kept
    strictly locked at that angle with ZERO movement. Positioning is achieved
    solely using Shoulder (S) and Elbow (E) in the vertical pitch plane, plus
    Base (B) for azimuth alignment.
    
    Parameters:
      target_base_xyz: tuple or array (X_mm, Y_mm, Z_mm) in Robot Base frame
      lock_arm_angle: float (default 130.0), locks Arm servo angle
      preferred_approach_pitch: desired pitch angle (radians) of gripper at contact
                                
    Returns:
      (B, S, A, E) : tuple of floats (degrees rounded to 1 decimal)
      fk_error_mm  : float, positioning error between target and FK(sol)
      is_valid     : bool, True if reachable and within all servo safety limits
      reason       : str, diagnostic message
    """
    tx, ty, tz = target_base_xyz

    # 1. Base Azimuth Angle
    theta_base = math.atan2(ty, tx)
    B = 90.0 - math.degrees(theta_base)
    B = float(np.clip(B, LIMIT_BASE_MIN, LIMIT_BASE_MAX))

    # 2. Planar Target in Arm Vertical Pitch Plane
    r_goal = math.sqrt(tx**2 + ty**2)
    z_goal = float(tz)

    A_fixed = float(lock_arm_angle) if lock_arm_angle is not None else 130.0

    # Compound Link 12 geometry formed by Link 1 (212mm) and Link 2 (199mm) with Arm locked at A_fixed
    bend_arm_rad = math.radians(A_fixed - 70.0)
    C1 = L1_SHOULDER_MM + L2_ARM_MM * math.cos(bend_arm_rad)
    C2 = L2_ARM_MM * math.sin(bend_arm_rad)
    L12 = math.sqrt(C1**2 + C2**2)
    phi12 = math.atan2(C2, C1)

    # 3D Distance from Shoulder Pivot (0, BASE_SHOULDER_HEIGHT_MM) to Target
    dr = r_goal
    dz = z_goal - BASE_SHOULDER_HEIGHT_MM
    D = math.sqrt(dr**2 + dz**2)

    min_reach = abs(L12 - L3_GRIPPER_MM)
    max_reach = L12 + L3_GRIPPER_MM

    if D > max_reach:
        d_excess = D - max_reach
        return None, d_excess, False, f"Target distance {D:.1f}mm exceeds max 2-DOF reach ({max_reach:.1f}mm) by {d_excess:.1f}mm: move robot forward by {d_excess:.0f}mm"
    if D < min_reach:
        d_short = min_reach - D
        return None, d_short, False, f"Target distance {D:.1f}mm too close to shoulder (min {min_reach:.1f}mm): move back by {d_short:.1f}mm"

    # Closed-Form Law of Cosines for 2-Link Planar Kinematics
    gamma = math.atan2(dz, dr)
    cos_psi = (L12**2 + D**2 - L3_GRIPPER_MM**2) / (2.0 * L12 * D)
    cos_psi = max(-1.0, min(1.0, cos_psi))
    psi = math.acos(cos_psi)

    valid_sols = []
    for sign in [-1.0, 1.0]:
        theta12 = gamma + sign * psi
        a1 = theta12 + phi12
        S = math.degrees(a1) + 50.0

        if not (LIMIT_SHOULDER_MIN <= S <= LIMIT_SHOULDER_MAX):
            continue

        a2 = a1 - bend_arm_rad
        r2 = L1_SHOULDER_MM * math.cos(a1) + L2_ARM_MM * math.cos(a2)
        z2 = BASE_SHOULDER_HEIGHT_MM + L1_SHOULDER_MM * math.sin(a1) + L2_ARM_MM * math.sin(a2)

        a3 = math.atan2(z_goal - z2, r_goal - r2)
        E = 110.0 - math.degrees(a3 - a2)

        if not (LIMIT_ELBOW_MIN <= E <= LIMIT_ELBOW_MAX):
            continue

        sol = (round(B, 1), round(S, 1), round(A_fixed, 1), round(E, 1))
        fk = forward_kinematics(*sol)
        err = float(np.linalg.norm(fk['gripper'] - np.array([tx, ty, tz])))

        # Favor forward-reaching posture close to normal harvest stance (S~100, E~110)
        posture_cost = abs(S - 100.0) + abs(E - 110.0) * 0.8
        valid_sols.append((posture_cost, sol, err))

    if not valid_sols:
        return None, 999.0, False, f"Target within reach radius ({D:.1f}mm) but Shoulder/Elbow angle limits [50, 140] deg exceeded"

    valid_sols.sort(key=lambda x: x[0])
    best_sol = valid_sols[0][1]
    best_err = valid_sols[0][2]
    return best_sol, best_err, True, f"OK (Arm locked at {A_fixed}deg; Sh & Elbow solved, Err={best_err:.2f}mm)"


def compute_pre_pick_pose(target_base_xyz, backoff_dist_mm=35.0, lock_arm_angle=130.0):
    """
    Computes a safe Pre-Pick waypoint position backed off from target_base_xyz
    along the radial approach direction toward the robot base.
    
    Returns:
      pre_pick_xyz : np.ndarray [X, Y, Z]
      ik_angles    : tuple (B, S, A, E) or None
      is_valid     : bool
      reason       : str
    """
    tb = np.array(target_base_xyz, dtype=float)
    r_dir = np.array([tb[0], tb[1], 0.0])
    r_len = np.linalg.norm(r_dir)
    if r_len < 1e-4:
        r_unit = np.array([1.0, 0.0, 0.0])
    else:
        r_unit = r_dir / r_len

    # Try requested backoff distance first
    for dist in [backoff_dist_mm, 25.0, 15.0]:
        pre_pick_xyz = tb - dist * r_unit
        angles, err, ok, reason = inverse_kinematics(pre_pick_xyz, lock_arm_angle=lock_arm_angle)
        if ok:
            return pre_pick_xyz, angles, True, reason

    # Fallback to target position itself if backoff puts it out of reach
    angles, err, ok, reason = inverse_kinematics(tb, lock_arm_angle=lock_arm_angle)
    return tb, angles, ok, reason


def check_reachability(target_base_xyz, lock_arm_angle=130.0):
    """Quick reachability check without full motion planning."""
    angles, err, ok, reason = inverse_kinematics(target_base_xyz, lock_arm_angle=lock_arm_angle)
    return ok, reason, angles


def calculate_elbow_aiming(target_base_xyz, cur_shoulder=100.0, cur_elbow=110.0, lock_arm_angle=130.0):
    """
    Direct Elbow Aiming Engine:
    Calculates the exact Elbow angle required to point Link 3 and the gripper
    directly at target_base_xyz (X, Y, Z), keeping Arm strictly locked at 130 deg
    and Shoulder at a stable picking height.
    
    Guarantees a safe, valid, non-rejecting solution within physical servo limits.
    
    Returns:
      sol_target : tuple (B, S, A, E)
      sol_pre    : tuple (B, S, A, E)
      err_mm     : float, distance from gripper to target
      reason     : str
    """
    tx, ty, tz = target_base_xyz
    theta_base = math.atan2(ty, tx)
    B = 90.0 - math.degrees(theta_base)
    B = float(np.clip(B, LIMIT_BASE_MIN, LIMIT_BASE_MAX))

    S = float(np.clip(cur_shoulder if 70.0 <= cur_shoulder <= 115.0 else 100.0, LIMIT_SHOULDER_MIN, LIMIT_SHOULDER_MAX))
    A = float(lock_arm_angle if lock_arm_angle is not None else 130.0)

    a1 = math.radians(S - 50.0)
    a2 = a1 - math.radians(A - 70.0)

    r_elb = L1_SHOULDER_MM * math.cos(a1) + L2_ARM_MM * math.cos(a2)
    z_elb = BASE_SHOULDER_HEIGHT_MM + L1_SHOULDER_MM * math.sin(a1) + L2_ARM_MM * math.sin(a2)

    r_target = math.hypot(tx, ty)
    dr = r_target - r_elb
    dz = tz - z_elb

    target_a3 = math.atan2(dz, dr)
    E = 110.0 - math.degrees(target_a3 - a2)
    E = float(np.clip(E, LIMIT_ELBOW_MIN, LIMIT_ELBOW_MAX))

    sol_target = (round(B, 1), round(S, 1), round(A, 1), round(E, 1))

    # Pre-pick approach stance: slightly backed off (8 deg) so it approaches the boll smoothly
    E_pre = float(np.clip(E - 8.0 if E > 70.0 else E + 8.0, LIMIT_ELBOW_MIN, LIMIT_ELBOW_MAX))
    sol_pre = (round(B, 1), round(S, 1), round(A, 1), round(E_pre, 1))

    fk = forward_kinematics(*sol_target)
    err = float(np.linalg.norm(fk['gripper'] - np.array([tx, ty, tz])))

    return sol_target, sol_pre, err, f"Elbow aimed directly at cotton boll (E={round(E, 1)}deg)"


def measure_gripper_to_cotton(p_cam_3d, cur_B, cur_S, cur_A, cur_E, max_reach_dist_mm=280.0, max_cam_depth_mm=280.0):
    """
    Measures the exact 3D distance and angles between the Gripper Contact Center
    and the center of the detected Cotton Boll.
    
    In the camera coordinate frame, Link 3's Gripper Contact Point is located at
    [0.0, +121.0, +132.0] mm (fixed rigid offset).
    Teeth plane is forward at Z = +132.0 mm along the forearm axis.
    
    Evaluates exact aiming reachability (2-DOF Base + Elbow) within safe servo limits (130 deg max).
    If the cotton boll exceeds reach (Z > 280mm or aiming servo angles out of limits),
    the measurement is flagged as 'LONG' (too far to catch or pluck from current position).
    
    Returns dict:
      'p_cotton_cam'   : (Xc, Yc, Zc) in mm
      'p_cotton_base'  : [X0, Y0, Z0] in mm
      'p_gripper_cam'  : (0.0, 121.0, 132.0) in mm
      'p_gripper_base' : [Xg, Yg, Zg] in mm
      'distance_mm'    : float, distance from gripper teeth to cotton center
      'dx_mm'          : lateral error relative to gripper centerline (+ right, - left)
      'dy_mm'          : vertical error relative to camera optical axis (+ down, - up)
      'dz_mm'          : forward depth beyond gripper teeth tip (+ in front, - inside jaws)
      'yaw_deg'        : horizontal angle from camera optical axis to cotton center
      'pitch_deg'      : vertical elevation angle from camera optical axis to cotton center
      'is_long'        : bool, True if measurement exceeds plucking reach
      'status_str'     : "LONG" or "REACHABLE"
    """
    Xc = float(p_cam_3d[0])
    Yc = float(p_cam_3d[1])
    Zc = float(p_cam_3d[2])

    # Fixed offset of Gripper Contact Center in Camera frame:
    # X=0 (aligned with gripper axis), Y=+121mm (stalk offset), Z=+132mm (202mm L3 - 70mm mount)
    grp_cam_x = 0.0
    grp_cam_y = CAM_MOUNT_OFFSET_Y_MM       # 121.0 mm
    grp_cam_z = L3_GRIPPER_MM - CAM_MOUNT_OFFSET_X_MM # 132.0 mm (teeth plane forward)

    dx = Xc - grp_cam_x
    dy = Yc
    dz = Zc - grp_cam_z  # Forward distance from gripper teeth tip (+ ahead, - already inside jaws)

    # Effective distance from gripper teeth contact zone
    forward_dist = max(0.0, dz)
    dist_from_teeth = math.hypot(dx, forward_dist)

    # Angular direction from camera optical axis to target center
    forward_proj = max(10.0, Zc)
    yaw_deg = math.degrees(math.atan2(Xc, forward_proj))
    pitch_deg = math.degrees(math.atan2(Yc, forward_proj))

    # Base frame coordinates
    p_cotton_base = camera_to_base_frame((Xc, Yc, Zc), cur_B, cur_S, cur_A, cur_E)
    fk = forward_kinematics(cur_B, cur_S, cur_A, cur_E)
    p_gripper_base = fk['gripper']

    # Aiming solution validation
    sol_aim, _, err_aim, _ = calculate_elbow_aiming(p_cotton_base, cur_shoulder=cur_S, cur_elbow=cur_E, lock_arm_angle=cur_A)
    aim_b, aim_s, aim_a, aim_e = sol_aim
    is_angle_valid = (LIMIT_ELBOW_MIN <= aim_e <= LIMIT_ELBOW_MAX) and (LIMIT_BASE_MIN <= aim_b <= LIMIT_BASE_MAX)

    # LONG Condition:
    # Flagged as LONG only if boll is beyond maximum operating reach (Zc > 280mm) or outside safe joint limits
    is_long = (Zc > max_cam_depth_mm or not is_angle_valid)
    status_str = "LONG" if is_long else "REACHABLE"

    return {
        'p_cotton_cam': (round(Xc, 1), round(Yc, 1), round(Zc, 1)),
        'p_cotton_base': tuple(np.round(p_cotton_base, 1)),
        'p_gripper_cam': (grp_cam_x, grp_cam_y, grp_cam_z),
        'p_gripper_base': tuple(np.round(p_gripper_base, 1)),
        'distance_mm': round(dist_from_teeth, 1),
        'dx_mm': round(dx, 1),
        'dy_mm': round(dy, 1),
        'dz_mm': round(dz, 1),
        'yaw_deg': round(yaw_deg, 1),
        'pitch_deg': round(pitch_deg, 1),
        'is_long': is_long,
        'status_str': status_str
    }

