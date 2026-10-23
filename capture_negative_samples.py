import os
import sys
import time
import cv2
import argparse
from pathlib import Path

TRAIN_IMG = Path(r"C:\cotton_robot\dataset\cotton\combined_training_v4\train\images")
TRAIN_LBL = Path(r"C:\cotton_robot\dataset\cotton\combined_training_v4\train\labels")

def run_collector(cam_index=None):
    TRAIN_IMG.mkdir(parents=True, exist_ok=True)
    TRAIN_LBL.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" HARD NEGATIVE BACKGROUND COLLECTOR FOR RETRAINING")
    print("=" * 75)
    print("Use this tool to capture:")
    print("  1. Dark room & dim ambient lighting (removes dark room false triggers)")
    print("  2. Faces, skin, hands (teaches YOLO never to detect human skin/faces)")
    print("  3. Direct sunlight / harsh window glare (suppresses sunlight false alarms)")
    print("  4. Household white objects (white cups, paper, walls, clothing, cables)")
    print("-" * 75)
    print("CONTROLS:")
    print("  [SPACE] : Snap 1 Negative Background Image")
    print("  [B]     : Burst Capture (20 rapid frames while you pan the camera)")
    print("  [X]     : Switch camera (Cam 0 <-> Cam 1 <-> Cam 2)")
    print("  [Q/ESC] : Exit and save")
    print("=" * 75)

    current_idx = 0 if cam_index is None else cam_index
    cap = cv2.VideoCapture(current_idx)
    if not cap.isOpened():
        print(f"[WARN] Camera index {current_idx} failed, trying 1...")
        cap = cv2.VideoCapture(1)
        current_idx = 1 if cap.isOpened() else 0

    if not cap.isOpened():
        print("[ERROR] Could not open any camera. Please check USB connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    captured_this_session = 0
    burst_mode = False
    burst_remaining = 0
    last_burst_time = 0.0

    window_name = "Negative Background Collector - Press SPACE to capture, B for burst, Q to exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 680)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        h, w = frame.shape[:2]
        display_frame = frame.copy()

        # Handle burst capture
        if burst_remaining > 0:
            now = time.time()
            if now - last_burst_time >= 0.15:  # 150ms per frame
                timestamp = int(time.time() * 1000)
                img_name = f"neg_real_burst_{timestamp}.jpg"
                lbl_name = f"neg_real_burst_{timestamp}.txt"
                cv2.imwrite(str(TRAIN_IMG / img_name), frame)
                (TRAIN_LBL / lbl_name).write_text("", encoding="utf-8")
                captured_this_session += 1
                burst_remaining -= 1
                last_burst_time = now

        # Draw Center Object Framing Guide
        box_sz = int(min(h, w) * 0.50)
        bx1 = (w - box_sz) // 2
        by1 = (h - box_sz) // 2
        bx2 = bx1 + box_sz
        by2 = by1 + box_sz
        guide_color = (0, 0, 255) if burst_remaining > 0 else (0, 255, 255)
        cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), guide_color, 2)
        cv2.putText(display_frame, "HOLD WHITE OBJECT HERE", (bx1 + 10, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, guide_color, 2, cv2.LINE_AA)

        # Draw HUD Overlay banner
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (15, 15, 15), -1)
        cv2.rectangle(overlay, (0, h - 60), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.80, display_frame, 0.20, 0, display_frame)

        # Large High-Visibility Counter & Target Goal (100)
        target_goal = 100
        count_color = (0, 255, 120) if captured_this_session >= target_goal else (0, 220, 255)
        count_str = f"CAPTURED: {captured_this_session} / {target_goal} IMAGES"
        if burst_remaining > 0:
            count_str += f"  [BURST ACTIVE: {burst_remaining} LEFT]"
            count_color = (0, 80, 255)

        cv2.putText(display_frame, count_str, (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, count_color, 3, cv2.LINE_AA)

        # Progress bar
        bar_w = int(w * 0.70)
        bar_fill = int(bar_w * min(1.0, captured_this_session / float(target_goal)))
        cv2.rectangle(display_frame, (25, 65), (25 + bar_w, 82), (60, 60, 60), -1)
        cv2.rectangle(display_frame, (25, 65), (25 + bar_fill, 82), (0, 255, 120), -1)
        cv2.rectangle(display_frame, (25, 65), (25 + bar_w, 82), (200, 200, 200), 1)

        # Subtitle instructions
        help_str = "Press [SPACE] for 1 snap  |  Press [B] for 20 rapid bursts  |  Press [Q] when done"
        cv2.putText(display_frame, help_str, (25, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key == 32:  # SPACE
            timestamp = int(time.time() * 1000)
            img_name = f"neg_real_snap_{timestamp}.jpg"
            lbl_name = f"neg_real_snap_{timestamp}.txt"
            cv2.imwrite(str(TRAIN_IMG / img_name), frame)
            (TRAIN_LBL / lbl_name).write_text("", encoding="utf-8")
            captured_this_session += 1
            print(f"[CAPTURED # {captured_this_session}] Saved: {img_name} (with 0-byte negative label)")
        elif key in (ord('b'), ord('B')):
            burst_remaining = 20
            last_burst_time = time.time()
            print("[BURST MODE] Capturing 20 rapid frames... Pan camera around scene!")
        elif key in (ord('x'), ord('X')):
            cap.release()
            current_idx = (current_idx + 1) % 3
            print(f"Switching to camera index {current_idx}...")
            cap = cv2.VideoCapture(current_idx)
            if not cap.isOpened():
                print(f"Cam {current_idx} not found, reverting to 0...")
                current_idx = 0
                cap = cv2.VideoCapture(0)

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 75)
    print(f" CAPTURE SESSION COMPLETE: {captured_this_session} new negative background frames added!")
    print(f" Destination Images: {TRAIN_IMG}")
    print(f" Destination Labels: {TRAIN_LBL} (0-byte empty files)")
    print(" YOLO will now use these frames to penalize false detections on faces, dark noise, and white objects.")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture hard negative background frames directly from webcam")
    parser.add_argument("--cam", type=int, default=None, help="Camera index (0=internal, 1=external, etc.)")
    args = parser.parse_args()
    run_collector(args.cam)
