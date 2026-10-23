"""
Interactive Calibration, Kinematics Testing, and Validation Tool
for Cotton Harvester Robotic Arm.

Allows safe testing of:
  1. Forward Kinematics (FK) against physical joint positions.
  2. Inverse Kinematics (IK) for arbitrary target points (X, Y, Z).
  3. Camera-to-Robot 3D Coordinate Conversion.
  4. Individual servo angle and direction verification (safe jogging).
  5. Pre-Pick and Approach trajectory planning.
  6. Dry-run / hardware simulation of complete pick-and-place cycle.
"""

import math
import sys
import time
import numpy as np

import robot_kinematics as rk
from serial_controller import SerialArmCommunicator


def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title.center(68)}")
    print("=" * 70)


def test_forward_kinematics():
    print_header("1. FORWARD KINEMATICS CALCULATOR")
    print("Enter servo angles in degrees (or press Enter for default):")
    try:
        b_str = input(f"  Base angle (10-170, default {rk.HOME_B}): ").strip()
        s_str = input(f"  Shoulder angle (50-140, default {rk.HOME_S}): ").strip()
        a_str = input(f"  Arm angle (50-140, default {rk.HOME_A}): ").strip()
        e_str = input(f"  Elbow angle (50-140, default {rk.HOME_E}): ").strip()

        b = float(b_str) if b_str else rk.HOME_B
        s = float(s_str) if s_str else rk.HOME_S
        a = float(a_str) if a_str else rk.HOME_A
        e = float(e_str) if e_str else rk.HOME_E
    except ValueError:
        print("[ERROR] Invalid numeric input.")
        return

    fk = rk.forward_kinematics(b, s, a, e)
    tb, a1, a2, a3 = fk['link_angles_deg']

    print("\n" + "-" * 70)
    print(f" SERVO ANGLES INPUT : B={b:.1f}deg, S={s:.1f}deg, A={a:.1f}deg, E={e:.1f}deg")
    print(f" GEOMETRIC ANGLES   : Base Azimuth={tb:.1f}deg, Link1(Sh)={a1:.1f}deg, Link2(Arm)={a2:.1f}deg, Forearm={a3:.1f}deg")
    print("-" * 70)
    print(f" Joint 3D Positions in Robot Base Frame {{0}} (Origin at base axis on mounting plate):")
    print(f"  * Shoulder Pivot (0,0,H)   : X={fk['shoulder'][0]:7.1f}mm, Y={fk['shoulder'][1]:7.1f}mm, Z={fk['shoulder'][2]:7.1f}mm")
    print(f"  * Arm Pivot                : X={fk['arm'][0]:7.1f}mm, Y={fk['arm'][1]:7.1f}mm, Z={fk['arm'][2]:7.1f}mm")
    print(f"  * Elbow Pivot              : X={fk['elbow'][0]:7.1f}mm, Y={fk['elbow'][1]:7.1f}mm, Z={fk['elbow'][2]:7.1f}mm")
    print(f"  * Gripper Contact Point    : X={fk['gripper'][0]:7.1f}mm, Y={fk['gripper'][1]:7.1f}mm, Z={fk['gripper'][2]:7.1f}mm")
    print(f"  * RealSense Optical Center : X={fk['camera'][0]:7.1f}mm, Y={fk['camera'][1]:7.1f}mm, Z={fk['camera'][2]:7.1f}mm")
    print("-" * 70)
    cam_to_grp = np.linalg.norm(fk['gripper'] - fk['camera'])
    print(f"  Distance Camera to Gripper Contact: {cam_to_grp:.1f}mm (dx={fk['gripper'][0]-fk['camera'][0]:.1f}mm, dz={fk['gripper'][2]-fk['camera'][2]:.1f}mm)")


def test_inverse_kinematics():
    print_header("2. INVERSE KINEMATICS CALCULATOR")
    print("Enter desired 3D coordinates (mm) in Robot Base Frame {0}:")
    try:
        x_str = input("  Target X (mm, forward): ").strip()
        y_str = input("  Target Y (mm, left + / right -): ").strip()
        z_str = input("  Target Z (mm, height above base plate): ").strip()

        tx = float(x_str) if x_str else 540.0
        ty = float(y_str) if y_str else 0.0
        tz = float(z_str) if z_str else 250.0
    except ValueError:
        print("[ERROR] Invalid numeric input.")
        return

    target = (tx, ty, tz)
    sol, err, ok, reason = rk.inverse_kinematics(target)

    print("\n" + "-" * 70)
    print(f" REQUESTED TARGET : X={tx:.1f}mm, Y={ty:.1f}mm, Z={tz:.1f}mm")
    dist_sh = math.sqrt(tx**2 + ty**2 + (tz - rk.BASE_SHOULDER_HEIGHT_MM)**2)
    print(f" Distance from Shoulder Pivot: {dist_sh:.1f}mm (Max arm reach: {rk.MAX_ARM_REACH_MM:.1f}mm)")
    print("-" * 70)

    if ok:
        b, s, a, e = sol
        fk = rk.forward_kinematics(b, s, a, e)
        print(f" >>> IK SOLUTION FOUND:")
        print(f"     Base Servo (B)     : {b:.1f} deg  (Limits: [{rk.LIMIT_BASE_MIN}, {rk.LIMIT_BASE_MAX}])")
        print(f"     Shoulder Servo (S) : {s:.1f} deg  (Limits: [{rk.LIMIT_SHOULDER_MIN}, {rk.LIMIT_SHOULDER_MAX}])")
        print(f"     Arm Servo (A)      : {a:.1f} deg  (Limits: [{rk.LIMIT_ARM_MIN}, {rk.LIMIT_ARM_MAX}])")
        print(f"     Elbow Servo (E)    : {e:.1f} deg  (Limits: [{rk.LIMIT_ELBOW_MIN}, {rk.LIMIT_ELBOW_MAX}])")
        print(f"     Reconstructed FK   : X={fk['gripper'][0]:.2f}mm, Y={fk['gripper'][1]:.2f}mm, Z={fk['gripper'][2]:.2f}mm")
        print(f"     Positioning Error  : {err:.4f} mm")
        print(f"     Status             : {reason}")
    else:
        print(f" >>> TARGET UNREACHABLE:")
        print(f"     Reason: {reason}")


def test_camera_coordinate_conversion():
    print_header("3. CAMERA-TO-ROBOT COORDINATE CONVERSION")
    print("Enter RealSense 3D measurement in Camera Frame (mm):")
    try:
        zc_str = input("  Camera Depth Z_c (mm, e.g. 150): ").strip()
        xc_str = input("  Camera X_c (mm, right +, left -, default 0): ").strip()
        yc_str = input("  Camera Y_c (mm, down +, up -, default 110): ").strip()

        zc = float(zc_str) if zc_str else 150.0
        xc = float(xc_str) if xc_str else 0.0
        yc = float(yc_str) if yc_str else 110.0

        print("\nEnter arm joint configuration at moment of detection:")
        b_str = input(f"  Current Base angle (default 90): ").strip()
        s_str = input(f"  Current Shoulder angle (default 100): ").strip()
        a_str = input(f"  Current Arm angle (default 130): ").strip()
        e_str = input(f"  Current Elbow angle (default 100): ").strip()

        b = float(b_str) if b_str else 90.0
        s = float(s_str) if s_str else 100.0
        a = float(a_str) if a_str else 130.0
        e = float(e_str) if e_str else 100.0
    except ValueError:
        print("[ERROR] Invalid numeric input.")
        return

    p_cam = (xc, yc, zc)
    p_base = rk.camera_to_base_frame(p_cam, b, s, a, e)
    sol, err, ok, reason = rk.inverse_kinematics(p_base)

    print("\n" + "-" * 70)
    print(f" RealSense Measurement : X_c={xc:.1f}mm, Y_c={yc:.1f}mm, Z_c={zc:.1f}mm ({zc/10.0:.1f}cm)")
    print(f" Arm Detection Pose    : B={b:.1f}deg, S={s:.1f}deg, A={a:.1f}deg, E={e:.1f}deg")
    print("-" * 70)
    print(f" Converted Robot Base  : X={p_base[0]:.1f}mm, Y={p_base[1]:.1f}mm, Z={p_base[2]:.1f}mm")
    if ok:
        print(f" IK Required Servos    : Base={sol[0]}deg, Sh={sol[1]}deg, Arm={sol[2]}deg, Elbow={sol[3]}deg")
        print(f" Round-Trip Accuracy   : {err:.4f} mm")
        print(f" Status                : {reason}")
    else:
        print(f" Reachability          : REJECTED - {reason}")


def test_safe_servo_jogging():
    print_header("4. SAFE SERVO JOGGING & LIMITS CHECK")
    serial_comm = SerialArmCommunicator(port="COM7", baudrate=115200)
    is_live = serial_comm.is_connected
    mode_str = "HARDWARE ACTIVE on " + str(serial_comm.port) if is_live else "VIRTUAL / SIMULATED MODE"
    print(f"Serial Status: {mode_str}\n")

    while True:
        print("-" * 50)
        print("Select Joint to Jog:")
        print("  1. Base (B)      - Limits: 20-140 deg")
        print("  2. Shoulder (S)  - Limits: 50-140 deg")
        print("  3. Arm (A)       - Limits: 50-140 deg")
        print("  4. Elbow (E)     - Limits: 50-140 deg")
        print("  5. Gripper (G)   - 10 (Close) to 60 (Open)")
        print("  6. Home All (H)")
        print("  0. Back to Main Menu")

        ch = input("\nChoice [0-6]: ").strip()
        if ch == '0':
            break
        elif ch == '6':
            print("Sending Home ('H')...")
            serial_comm.send_cmd("H")
            continue

        axis_map = {'1': ('B', 20, 140), '2': ('S', 50, 140), '3': ('A', 50, 140), '4': ('E', 50, 140), '5': ('G', 10, 60)}
        if ch not in axis_map:
            print("Invalid choice.")
            continue

        axis_char, min_a, max_a = axis_map[ch]
        ang_str = input(f"Enter target angle for Joint {axis_char} [{min_a}-{max_a} deg]: ").strip()
        try:
            ang = int(ang_str)
            if not (min_a <= ang <= max_a):
                print(f"[REJECTED] Angle {ang} is outside safe limits [{min_a}, {max_a}]!")
                continue
            if axis_char == 'G':
                cmd = "C" if ang <= 20 else "O"
                print(f"Commanding Gripper -> '{cmd}' (Angle {ang}deg)")
                serial_comm.send_cmd(cmd)
            else:
                print(f"Commanding Joint {axis_char} -> {ang} deg")
                serial_comm.send_servo(axis_char, ang)
        except ValueError:
            print("Invalid number.")


def test_dry_run_full_sequence():
    print_header("5. FULL PICK-AND-PLACE TRAJECTORY SIMULATION")
    from cotton_search_controller import CottonSearchController
    mock_comm = SerialArmCommunicator(port=None)
    controller = CottonSearchController(mock_comm)
    controller.set_dry_run(True)

    print("Simulating detection of stationary cotton boll at camera Z=140mm, X=0, Y=110mm:")
    controller.on_cotton_locked(target_id=1, cx=640, cy=360, z_mm=140.0, x_mm=0.0, y_mm=110.0)

    # Step through stages
    stages = [
        ("STABILIZING 3D", 1.0),
        ("PRE-PICK MOVE", 1.5),
        ("APPROACH MOVE", 1.2),
        ("GRIP & PLUCK CW", 4.5),
        ("RETRACT LIFT", 2.0),
        ("DROP MOVE (BASE 50)", 3.5),
        ("RELEASE CCW", 4.0),
        ("HOMING", 3.0)
    ]

    for name, dur in stages:
        print(f"\n--- Advancing Stage: {name} (Duration: {dur}s) ---")
        controller.stage_start_time = time.time() - (dur + 0.1)
        controller.homing_start_time = time.time() - (dur + 0.1)
        controller.update()

    print("\n>>> FULL DRY-RUN COMPLETE: Arm returned to Home and resumed search! <<<")


def main():
    while True:
        print_header("COTTON ROBOT ARM CALIBRATION & TEST TOOL")
        print("  1. Forward Kinematics (FK) - Physical joint positions")
        print("  2. Inverse Kinematics (IK) - Solve target (X, Y, Z)")
        print("  3. Camera-to-Robot 3D Coordinate Conversion")
        print("  4. Safe Individual Servo Jogging (Hardware / Simulated)")
        print("  5. Dry-Run Full Pick-and-Place Trajectory")
        print("  0. Exit")

        choice = input("\nSelect test [0-5]: ").strip()
        if choice == '1':
            test_forward_kinematics()
        elif choice == '2':
            test_inverse_kinematics()
        elif choice == '3':
            test_camera_coordinate_conversion()
        elif choice == '4':
            test_safe_servo_jogging()
        elif choice == '5':
            test_dry_run_full_sequence()
        elif choice == '0':
            print("\nExiting calibration tool. Goodbye!\n")
            sys.exit(0)
        else:
            print("[WARN] Invalid option.")


if __name__ == "__main__":
    main()
