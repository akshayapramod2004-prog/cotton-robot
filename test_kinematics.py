"""
Kinematics verification and test script for Cotton Harvester Robot Arm.
Tests:
  1. Forward Kinematics (FK)
  2. Camera-to-Robot Base Coordinate Transformation
  3. Inverse Kinematics (IK)
  4. Closed-loop Round-trip Validation: FK(IK(P)) == P
  5. Joint Limit & Reachability Verification
"""

import math
import numpy as np

# Confirmed Physical Robot Dimensions (mm)
L1_SHOULDER_MM = 212.0   # Shoulder-to-Arm pivot
L2_ARM_MM      = 199.0   # Arm-to-Elbow pivot
L3_ELBOW_MM    = 202.0   # Elbow pivot to Gripper contact point / tip
BASE_HEIGHT_MM = 110.0   # Base-axis to Shoulder pivot height (plate height 65mm not added)

# Camera mounting on Link 3 (Forearm/Gripper assembly)
# Located 70mm forward along link from Elbow pivot (202 - 132 = 70mm)
# Located 121mm above link centerline (including stand)
CAM_OFFSET_LINK_X = 70.0    # Along link towards gripper
CAM_OFFSET_LINK_Y = 121.0   # Perpendicular to link (normal, up)
CAM_OFFSET_LINK_Z = 0.0     # Lateral offset (0 if centered, or +left/-right)

# Joint Limits (degrees)
LIMITS = {
    'B': (10, 170),
    'S': (50, 140),
    'A': (50, 140),
    'E': (50, 140),
    'G': (10, 60)
}

HOME_ANGLES = {
    'B': 90,
    'S': 140,
    'A': 130,
    'E': 140,
    'G': 60
}

def servo_to_link_angles(B, S, A, E):
    """
    Converts software servo angles (degrees) to physical geometric angles:
      theta_base: Azimuth angle (radians) in world horizontal plane (+Y is left, +X is forward)
      alpha1: Elevation of Link 1 above horizontal forward (radians)
      alpha2: Elevation of Link 2 above horizontal forward (radians)
      alpha3: Elevation of Link 3 above horizontal forward (radians)
    """
    # Base: 90 is forward (0 rad), 60 is left (+30 deg), 120 is right (-30 deg)
    theta_base = math.radians(90.0 - B)

    # Shoulder: 140 is upright (~90 deg), 50 is near horizontal (~0 deg)
    alpha1 = math.radians(S - 50.0)

    # Arm: physically reversed. At A=130, bent down ~60 deg from Link 1.
    alpha2 = alpha1 - math.radians(A - 70.0)

    # Elbow: physically reversed. At E=100, Link 3 is parallel to ground when alpha2=-10 deg.
    # At E=130, tilts down by 30 deg.
    alpha3 = alpha2 + math.radians(110.0 - E)

    return theta_base, alpha1, alpha2, alpha3

def link_to_servo_angles(theta_base, alpha1, alpha2, alpha3):
    """Converts physical link angles back to software servo angles (degrees)."""
    B = 90.0 - math.degrees(theta_base)
    S = math.degrees(alpha1) + 50.0
    A = math.degrees(alpha1 - alpha2) + 70.0
    E = 110.0 - math.degrees(alpha3 - alpha2)
    return B, S, A, E

def forward_kinematics(B, S, A, E):
    """
    Computes 3D positions of all joints, gripper tip, and camera in Robot Base frame {0}.
    Origin (0,0,0) is at base rotation axis on the mounting plate.
    Z points UP, X points forward at B=90, Y points left at B=90.
    """
    theta_base, a1, a2, a3 = servo_to_link_angles(B, S, A, E)

    # 2D planar coordinates in vertical arm plane (r = radial distance, z = vertical height)
    # Shoulder pivot
    r0, z0 = 0.0, BASE_HEIGHT_MM

    # Arm pivot
    r1 = r0 + L1_SHOULDER_MM * math.cos(a1)
    z1 = z0 + L1_SHOULDER_MM * math.sin(a1)

    # Elbow pivot
    r2 = r1 + L2_ARM_MM * math.cos(a2)
    z2 = z1 + L2_ARM_MM * math.sin(a2)

    # Gripper contact point / tip (at L3 along Link 3)
    r_grip = r2 + L3_ELBOW_MM * math.cos(a3)
    z_grip = z2 + L3_ELBOW_MM * math.sin(a3)

    # Camera origin on Link 3
    r_cam = r2 + CAM_OFFSET_LINK_X * math.cos(a3) - CAM_OFFSET_LINK_Y * math.sin(a3)
    z_cam = z2 + CAM_OFFSET_LINK_X * math.sin(a3) + CAM_OFFSET_LINK_Y * math.cos(a3)

    # Convert planar (r, lateral, z) to 3D base frame using base azimuth
    cos_b = math.cos(theta_base)
    sin_b = math.sin(theta_base)

    p_shoulder = np.array([0.0, 0.0, z0])
    p_arm = np.array([r1 * cos_b, r1 * sin_b, z1])
    p_elbow = np.array([r2 * cos_b, r2 * sin_b, z2])
    p_grip = np.array([r_grip * cos_b, r_grip * sin_b, z_grip])
    p_cam = np.array([r_cam * cos_b, r_cam * sin_b, z_cam])

    # Rotation matrix of Camera relative to Base frame:
    # Camera +Z_c is along Link 3 direction (cos_b*cos_a3, sin_b*cos_a3, sin_a3)
    # Camera +Y_c is DOWN towards Link 3 (-cos_b*(-sin_a3)... so pointing towards -Y_link3)
    # Camera +X_c is to the camera's RIGHT (+Z_link3, perpendicular to arm plane)
    z_c_axis = np.array([cos_b * math.cos(a3), sin_b * math.cos(a3), math.sin(a3)])
    y_c_axis = np.array([cos_b * math.sin(a3), sin_b * math.sin(a3), -math.cos(a3)])
    x_c_axis = np.array([sin_b, -cos_b, 0.0]) # points to right of arm plane
    R_cam = np.column_stack([x_c_axis, y_c_axis, z_c_axis])

    return {
        'shoulder': p_shoulder,
        'arm': p_arm,
        'elbow': p_elbow,
        'gripper': p_grip,
        'camera': p_cam,
        'R_cam': R_cam,
        'link_angles': (theta_base, a1, a2, a3)
    }

def camera_to_base_frame(p_cam_3d, cur_B, cur_S, cur_A, cur_E):
    """
    Transforms 3D target coordinates (X_c, Y_c, Z_c in mm) measured by RealSense
    into Robot Base frame {0} (X_0, Y_0, Z_0 in mm).
    """
    fk = forward_kinematics(cur_B, cur_S, cur_A, cur_E)
    cam_pos = fk['camera']
    R_cam = fk['R_cam']

    p_c = np.array(p_cam_3d, dtype=float)
    p_base = cam_pos + R_cam @ p_c
    return p_base

def inverse_kinematics(target_base_xyz, preferred_approach_pitch=0.0):
    """
    Calculates joint servo angles (B, S, A, E) to place the Gripper Contact Point
    at target_base_xyz = (X_0, Y_0, Z_0) in Robot Base frame.
    
    Returns:
      (B, S, A, E), fk_error_mm, is_valid, reason
    """
    tx, ty, tz = target_base_xyz

    # 1. Base Azimuth
    theta_base = math.atan2(ty, tx)
    B = 90.0 - math.degrees(theta_base)
    if not (LIMITS['B'][0] <= B <= LIMITS['B'][1]):
        return None, 999.0, False, f"Base angle {B:.1f} out of limits [{LIMITS['B'][0]}, {LIMITS['B'][1]}]"

    # 2. Target in vertical arm plane
    r_goal = math.sqrt(tx**2 + ty**2)
    z_goal = tz

    # Search for an approach pitch angle alpha3 that keeps all joints within limits
    # and satisfies IK. Preferred approach is near horizontal (alpha3 ~ 0 or pointing to boll)
    best_solution = None
    min_dist_to_preferred = 999.0

    pitch_candidates = [
        preferred_approach_pitch,
        0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.5
    ]
    # Also add dense sweep around preferred
    for p_deg in range(-35, 36, 5):
        p_rad = math.radians(p_deg)
        if p_rad not in pitch_candidates:
            pitch_candidates.append(p_rad)

    for a3 in pitch_candidates:
        # Elbow position for this gripper pitch
        r2 = r_goal - L3_ELBOW_MM * math.cos(a3)
        z2 = z_goal - L3_ELBOW_MM * math.sin(a3)

        # Vector from shoulder pivot (0, BASE_HEIGHT_MM) to elbow pivot (r2, z2)
        dr = r2
        dz = z2 - BASE_HEIGHT_MM
        d_sq = dr**2 + dz**2
        d = math.sqrt(d_sq)

        # Reach check for 2-link sub-arm (L1 and L2)
        if d > (L1_SHOULDER_MM + L2_ARM_MM) or d < abs(L1_SHOULDER_MM - L2_ARM_MM):
            continue

        # Law of Cosines for Arm joint (elbow up configuration)
        cos_elbow_joint = (d_sq - L1_SHOULDER_MM**2 - L2_ARM_MM**2) / (2.0 * L1_SHOULDER_MM * L2_ARM_MM)
        cos_elbow_joint = max(-1.0, min(1.0, cos_elbow_joint))
        # Relative bend angle between L1 and L2
        theta_bend = math.acos(cos_elbow_joint)

        # Angle of direct line from shoulder to elbow
        gamma = math.atan2(dz, dr)
        cos_beta = (d_sq + L1_SHOULDER_MM**2 - L2_ARM_MM**2) / (2.0 * L1_SHOULDER_MM * d)
        cos_beta = max(-1.0, min(1.0, cos_beta))
        beta = math.acos(cos_beta)

        # Elbow up solution: L1 is above the chord line
        a1 = gamma + beta
        a2 = a1 - theta_bend

        # Convert to servo angles
        test_B, test_S, test_A, test_E = link_to_servo_angles(theta_base, a1, a2, a3)

        # Check all servo limits
        if (LIMITS['S'][0] <= test_S <= LIMITS['S'][1] and
            LIMITS['A'][0] <= test_A <= LIMITS['A'][1] and
            LIMITS['E'][0] <= test_E <= LIMITS['E'][1]):
            
            # Verify FK
            fk_chk = forward_kinematics(test_B, test_S, test_A, test_E)
            err = np.linalg.norm(fk_chk['gripper'] - np.array([tx, ty, tz]))
            if err < 1.0: # Sub-millimeter accuracy
                dist_pref = abs(a3 - preferred_approach_pitch)
                if dist_pref < min_dist_to_preferred:
                    min_dist_to_preferred = dist_pref
                    best_solution = ((test_B, test_S, test_A, test_E), err, a3)

    if best_solution is not None:
        (b_ang, s_ang, a_ang, e_ang), err, chosen_pitch = best_solution
        return (round(b_ang, 1), round(s_ang, 1), round(a_ang, 1), round(e_ang, 1)), err, True, "OK"

    return None, 999.0, False, "Target outside reachable workspace within servo limits"


if __name__ == "__main__":
    print("=" * 60)
    print("COTTON HARVESTER ROBOT KINEMATICS VALIDATION")
    print("=" * 60)

    # 1. Test Home Configuration
    fk_home = forward_kinematics(HOME_ANGLES['B'], HOME_ANGLES['S'], HOME_ANGLES['A'], HOME_ANGLES['E'])
    print(f"\n1. Home Pose FK (B=90, S=140, A=130, E=140):")
    print(f"   Shoulder : {fk_home['shoulder']}")
    print(f"   Arm pivot: {fk_home['arm']}")
    print(f"   Elbow    : {fk_home['elbow']}")
    print(f"   Gripper  : {fk_home['gripper']}")
    print(f"   Camera   : {fk_home['camera']}")

    # 2. Test Sector 1 Search Pose
    fk_search = forward_kinematics(90, 100, 130, 100)
    print(f"\n2. Search Pose FK (B=90, S=100, A=130, E=100):")
    print(f"   Gripper Tip : {fk_search['gripper']}")
    print(f"   Camera Pos  : {fk_search['camera']}")
    cam_to_grp = np.linalg.norm(fk_search['gripper'] - fk_search['camera'])
    dx_grp = fk_search['gripper'][0] - fk_search['camera'][0]
    dz_grp = fk_search['gripper'][2] - fk_search['camera'][2]
    print(f"   Dist Cam to Gripper Tip : {cam_to_grp:.1f} mm (dx={dx_grp:.1f}mm, dz={dz_grp:.1f}mm)")

    # 3. Test Camera-to-Base Transformation with a detected cotton boll
    # Boll detected at Z=200mm in camera frame, centered (X=0, Y=0)
    boll_cam = (0.0, 0.0, 200.0)
    boll_base = camera_to_base_frame(boll_cam, 90, 100, 130, 100)
    print(f"\n3. Cotton Boll Detection Transform:")
    print(f"   Detected at Camera coords: {boll_cam} mm")
    print(f"   Transformed to Robot Base: {boll_base} mm")

    # 4. Inverse Kinematics to reach that Boll
    ik_angles, ik_err, ok, reason = inverse_kinematics(boll_base)
    print(f"\n4. Inverse Kinematics Solution to Reach Boll:")
    print(f"   Target Base Point : {boll_base}")
    print(f"   Calculated Servos : {ik_angles}")
    print(f"   IK Error          : {ik_err:.3f} mm")
    print(f"   Status            : {reason}")

    # Reconstruct with FK
    if ok:
        fk_sol = forward_kinematics(*ik_angles)
        print(f"   Reconstructed Gripper Contact: {fk_sol['gripper']}")
        diff = np.linalg.norm(fk_sol['gripper'] - boll_base)
        print(f"   Round-trip verification error: {diff:.4f} mm")
        assert diff < 1.0, f"Error {diff} exceeds 1mm!"
        print("   >>> SUCCESS: Round-trip FK(IK(P)) is exact! <<<")

    # 5. Test Multiple Reachable Workspace Points
    print("\n5. Testing Grid of Targets Across Workspace:")
    passed = 0
    total = 0
    for z_c in [120, 150, 180, 200, 230, 260]:
        for x_c in [-60, -30, 0, 30, 60]:
            for y_c in [-30, 0, 30]:
                total += 1
                b_pt = camera_to_base_frame((x_c, y_c, z_c), 90, 100, 130, 100)
                sol, err, is_ok, res = inverse_kinematics(b_pt)
                if is_ok:
                    passed += 1

    print(f"   Reachable points: {passed}/{total} ({passed/total*100:.1f}%) within strict servo safety limits!")
