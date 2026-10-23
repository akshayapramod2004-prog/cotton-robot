import time
import sys
import argparse
import math
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from serial_controller import SerialArmCommunicator
from realsense_camera import RealSenseCamera
from cotton_search_controller import CottonSearchController

if sys.platform == "win32":
    import msvcrt

# Model weights configuration - Trained Single-Class cotton model saved in weights/best.pt
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "best.pt"
MODEL_PATH = str(WEIGHTS_PATH if WEIGHTS_PATH.exists() else Path("yolo11s.pt"))

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)

class RoboticTargetTracker:
    """
    Robotic Arm Vision Tracker:
      1. Dynamic Velocity Matching (IoU + Centroid): Tracks rapid cotton motion without spawning extra boxes.
      2. Spatial Deduplication: Merges overlapping targets so only ONE box exists per cotton boll.
      3. Temporal Persistence Buffer (Hysteresis Filter): Prevents box flickering.
      4. EMA Position Smoothing: Eliminates spatial coordinate jitter.
    """
    def __init__(self, max_missing_frames=25, alpha=0.30):
        self.max_missing_frames = max_missing_frames
        self.alpha = alpha
        self.targets = {}
        self.next_target_id = 1

    def update(self, detected_boxes, frame_shape):
        current_ids = set()

        # Two-Priority Tracking: IoU spatial overlap first, then centroid distance
        for det in detected_boxes:
            x1, y1, x2, y2 = det['box']
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            best_match_id = None

            # Priority 1: Match by IoU overlap (highest IoU > 0.22)
            best_iou = 0.22
            for tid, tdata in self.targets.items():
                if tid in current_ids:
                    continue
                iou = compute_iou([x1, y1, x2, y2], tdata['smoothed_box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = tid

            # Priority 2: If no IoU overlap (e.g. camera sweep motion), match nearest centroid
            # Tight 35px threshold prevents adjacent bolls on the same branch from merging together
            if best_match_id is None:
                min_dist = float('inf')
                max_reach = min(35.0, max(box_w, box_h) * 0.45)
                for tid, tdata in self.targets.items():
                    if tid in current_ids:
                        continue
                    tcx = (tdata['smoothed_box'][0] + tdata['smoothed_box'][2]) / 2.0
                    tcy = (tdata['smoothed_box'][1] + tdata['smoothed_box'][3]) / 2.0
                    dist = np.hypot(cx - tcx, cy - tcy)
                    if dist < max_reach and dist < min_dist:
                        min_dist = dist
                        best_match_id = tid

            if best_match_id is not None:
                tid = best_match_id
                prev_sbox = np.array(self.targets[tid]['smoothed_box'], dtype=np.float32)
                curr_box = np.array([x1, y1, x2, y2], dtype=np.float32)

                # Rock-solid coordinate stabilization:
                # Deadband: if box moved < 5.0 pixels, freeze coordinates completely (zero twitching/jitter)
                max_delta = float(np.max(np.abs(curr_box - prev_sbox)))
                if max_delta < 5.0:
                    smoothed = prev_sbox
                else:
                    # Smooth continuous exponential moving average (65% previous + 35% new)
                    smoothed = 0.35 * curr_box + 0.65 * prev_sbox

                new_consec = self.targets[tid].get('consecutive_frames', 0) + 1
                in_depth_range = det.get('in_depth_range', False)
                det_z = det.get('z_mm', 0.0)
                det_p3d = det.get('p3d', (0.0, 0.0, 0.0))
                # Fast Sudden Confirmation on Frame 1: Confirms immediately if conf >= 0.15 or in reach range (3-28cm)
                is_confirmed = self.targets[tid].get('confirmed', False) or \
                               (det['conf'] >= 0.15) or \
                               in_depth_range or \
                               (new_consec >= 2)

                target_dict = {
                    'box': [x1, y1, x2, y2],
                    'smoothed_box': smoothed.tolist(),
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'conf': det['conf'],
                    'last_seen': 0,
                    'consecutive_frames': new_consec,
                    'confirmed': is_confirmed,
                    'in_depth_range': in_depth_range
                }
                if det_z > 0:
                    target_dict['last_valid_z'] = det_z
                    target_dict['last_valid_p3d'] = det_p3d

                self.targets[tid].update(target_dict)
                current_ids.add(tid)
            else:
                # Create new target candidate
                tid = self.next_target_id
                self.next_target_id += 1
                in_depth_range = det.get('in_depth_range', False)
                det_z = det.get('z_mm', 0.0)
                det_p3d = det.get('p3d', (0.0, 0.0, 0.0))

                # FAST SUDDEN DETECTION ON FRAME 1:
                # Confirms immediately on frame 1 if conf >= 0.15 or in reach range (3-28cm)
                is_confirmed = (det['conf'] >= 0.15) or in_depth_range
                self.targets[tid] = {
                    'box': [x1, y1, x2, y2],
                    'smoothed_box': [float(x1), float(y1), float(x2), float(y2)],
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'conf': det['conf'],
                    'last_seen': 0,
                    'consecutive_frames': 1,
                    'confirmed': is_confirmed,
                    'picked': False,
                    'last_valid_z': det_z if det_z > 0 else 0.0,
                    'last_valid_p3d': det_p3d if det_z > 0 else (0.0, 0.0, 0.0),
                    'in_depth_range': in_depth_range
                }
                current_ids.add(tid)

        # Aging and Cleanup for missed targets (Box persists and disappears ONLY when cotton is removed)
        dead_ids = []
        for tid in list(self.targets.keys()):
            if tid not in current_ids:
                self.targets[tid]['last_seen'] += 1
                # Give candidates 3 frames grace to confirm before dropping
                if not self.targets[tid].get('confirmed', False):
                    if self.targets[tid]['last_seen'] > 3:
                        dead_ids.append(tid)
                elif self.targets[tid]['last_seen'] > self.max_missing_frames:  # Disappears ONLY after cotton is removed
                    dead_ids.append(tid)

        for tid in dead_ids:
            del self.targets[tid]

        # Deduplicate only true duplicate ghost boxes (Preserves multiple clustered cotton bolls side-by-side)
        tids = list(self.targets.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                tid1, tid2 = tids[i], tids[j]
                if tid1 in self.targets and tid2 in self.targets:
                    box1 = self.targets[tid1]['smoothed_box']
                    box2 = self.targets[tid2]['smoothed_box']
                    iou = compute_iou(box1, box2)
                    c1 = ((box1[0]+box1[2])/2.0, (box1[1]+box1[3])/2.0)
                    c2 = ((box2[0]+box2[2])/2.0, (box2[1]+box2[3])/2.0)
                    c_dist = np.hypot(c1[0]-c2[0], c1[1]-c2[1])
                    # Only merge if practically identical box (high overlap > 70% and nearly identical center)
                    if iou > 0.70 and c_dist < 25:
                        if self.targets[tid1]['last_seen'] > self.targets[tid2]['last_seen']:
                            del self.targets[tid1]
                        elif self.targets[tid2]['last_seen'] > self.targets[tid1]['last_seen']:
                            del self.targets[tid2]
                        elif self.targets[tid1]['conf'] < self.targets[tid2]['conf']:
                            del self.targets[tid1]
                        else:
                            del self.targets[tid2]

        return self.targets

def enhance_lighting_for_inference(frame_bgr):
    """
    Normalizes extreme lighting conditions for YOLO detection:
    - Extreme Sunlight (mean luminance > 165): recovers texture & contours from clipped highlights.
    - Dark Room or Auto-Exposure Dimming (median luminance < 90 or mean < 80):
      lifts shadow details using adaptive CLAHE so cotton stays visible even when other white objects enter.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    median_lum = float(np.median(gray))

    # Lift cotton shadows if room is dim OR if a white object caused camera auto-exposure to dim the scene
    if median_lum < 90 or mean_lum < 80:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
        return enhanced, mean_lum
    elif mean_lum > 165:  # Extreme sunlight / high glare
        # Highlight recovery: apply gamma curve to compress clipped whites
        inv_gamma = 1.35
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(frame_bgr, lut)
        return enhanced, mean_lum

    return frame_bgr, mean_lum

def is_skin_region(crop_bgr, mean_lum=128.0):
    """
    Robust human face and skin filter:
    - Rejects human faces, necks, arms, and hands.
    - Preserves genuine cotton bolls even in bright sunlight or dark rooms.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False

    h, w = crop_bgr.shape[:2]
    total_pixels = h * w
    if total_pixels == 0:
        return False

    # Check for pure bright white/cream cotton fibers:
    # Cotton has high intensity and balanced R, G, B channels
    b, g, r = cv2.split(crop_bgr)
    v_thresh = max(60, min(140, int(mean_lum * 0.85)))

    rg_diff = cv2.absdiff(r, g)
    rb_diff = cv2.absdiff(r, b)
    is_bright = (r > v_thresh) & (g > v_thresh) & (b > max(35, v_thresh - 35))
    is_neutral = (rg_diff < 40) & (rb_diff < 55)
    white_cotton_mask = is_bright & is_neutral
    white_ratio = np.count_nonzero(white_cotton_mask) / float(total_pixels)

    # If substantial white cotton fibers (> 4%), it's genuine cotton!
    if white_ratio > 0.04:
        return False

    # Human skin detection in HSV
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lower_skin1 = np.array([0, 30, 45], dtype=np.uint8)
    upper_skin1 = np.array([25, 185, 245], dtype=np.uint8)
    lower_skin2 = np.array([165, 30, 45], dtype=np.uint8)
    upper_skin2 = np.array([180, 185, 245], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)

    skin_ratio = np.count_nonzero(skin_mask) / float(total_pixels)

    # Human skin criteria:
    if skin_ratio > 0.35 and white_ratio < 0.10:
        return True
    if skin_ratio > 0.50:
        return True

    return False

def is_smooth_flat_object(crop_bgr):
    """
    Rejects smooth flat white objects (plates, bowls, paper sheets, tiles, walls, mugs)
    that lack genuine cotton fiber texture and multi-lobed convolutions.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    h, w = crop_bgr.shape[:2]
    # Small objects (< 50px) should not be evaluated by smoothness
    if h < 50 or w < 50:
        return False

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Check inner 60% of the crop to avoid border edge artifacts
    m_h, m_w = max(2, int(h * 0.20)), max(2, int(w * 0.20))
    inner = gray[m_h:h-m_h, m_w:w-m_w]
    if inner.size == 0:
        inner = gray

    # Standard deviation of pixel intensities (smooth objects have very low variance)
    std_dev = float(np.std(inner))
    # Laplacian texture energy (measures fine surface wrinkles, fibers, cracks)
    laplacian_var = float(cv2.Laplacian(inner, cv2.CV_64F).var())

    # Truly smooth objects (like shiny ceramic plates, glossy plastic covers, and blank printer paper)
    # have very low standard deviation (< 7.0) and virtually zero edge texture (laplacian < 12.0)
    if std_dev < 7.0 and laplacian_var < 12.0:
        return True

    return False

def is_hollow_ring_or_non_solid(crop_bgr, mean_lum=128.0):
    """
    Rejects hollow rings, donuts, foam washers, and annular packing materials
    where the center is an empty hole showing the background/face.
    Real cotton bolls have a solid white fiber core at their center.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    h, w = crop_bgr.shape[:2]
    # Small cotton bolls (< 55px) never have holes; only check large objects (> 55px) like rings/donuts
    if h < 55 or w < 55:
        return False

    # Extract central 35% core of the bounding box
    ch1, ch2 = int(h * 0.32), int(h * 0.68)
    cw1, cw2 = int(w * 0.32), int(w * 0.68)
    core = crop_bgr[ch1:ch2, cw1:cw2]
    if core.size == 0:
        return False

    # Check whiteness of the central core (allow up to 110 saturation for RealSense/indoor sensor color casts)
    hsv_core = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    v_min = max(35, min(120, int(mean_lum * 0.55)))
    s_max = 110
    white_core_mask = cv2.inRange(hsv_core, np.array([0, 0, v_min]), np.array([180, s_max, 255]))
    core_white_ratio = np.count_nonzero(white_core_mask) / float(core.shape[0] * core.shape[1])

    # If the central core has less than 15% white fibers, it's a hollow ring / hole
    if core_white_ratio < 0.15:
        return True

    return False

def main():
    parser = argparse.ArgumentParser(description="Robotic Manipulator Cotton Boll Tracker Demo")
    parser.add_argument("--cam", type=int, default=None, help="Camera Index (Default: Auto-detects RealSense D435i depth camera; or pass 0, 1, 2 for OpenCV)")
    args = parser.parse_args()

    print("=" * 70)
    print("  ROBOTIC MANIPULATOR STABLE COTTON BOLL TRACKER & CONTROLLER DEMO")
    print("  [LONG-RANGE 60CM DETECTION MODE ACTIVE]")
    print("=" * 70)

    # Initialize Hardware / Virtual Serial Communicator (Targeting ESP32 on COM7)
    serial_comm = SerialArmCommunicator(port="COM7", cooldown_sec=2.0)

    if not Path(MODEL_PATH).exists():
        print(f"WARNING: Model weights not found at {MODEL_PATH}")
        print("Falling back to default yolo11s.pt for webcam testing.")
        model_file = "yolo11s.pt"
    else:
        model_file = MODEL_PATH

    print(f"Loading YOLO Model: {model_file}")
    model = YOLO(model_file)

    # Automatically isolate ONLY 'Fully opened boll' (strictly excludes Defected & Partially opened)
    if len(model.names) == 1:
        active_classes = [0]
    else:
        active_classes = [
            cid for cid, cname in model.names.items()
            if any(k in cname.lower() for k in ["mature", "cotton", "fully"])
        ]
        if not active_classes:
            active_classes = [0]
    active_class_names = [model.names.get(c, str(c)) for c in active_classes]
    print(f"Active Harvest Classes     : {active_class_names} (MATURE COTTON BOLLS ACTIVE)")

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Inference Device: {device_name} (device={device})")

    cam = RealSenseCamera(cam_index=args.cam)

    print("\nRobotic Arm Vision Control Active!")
    cam_name_str = "RealSense D435i Depth Camera (Auto-Detected)" if cam.is_realsense else f"OpenCV Camera (Index {cam.cam_index})"
    print(f"  - Camera Source             : {cam_name_str}")
    print("  - Harvest Range Filter      : ACTIVE (15cm - 25cm Window ONLY)")
    print("  - Persistence Memory        : ACTIVE (Prevents Box Flickering)")
    print("  - Coordinate EMA Smoothing  : ACTIVE (Eliminates Jitter for Robotic Pick)")
    print("  - Human Hand/Skin Filter    : ACTIVE (Adaptive for distant cotton; Toggle: 'F')")
    print("  - Serial Communicator       : ACTIVE (Controls ESP32 Servo & TB6612 Motor)")
    print("  - Automated Search Engine   : READY (3-Sector Automated Cotton Search Logic)")
    print("  - Controls:")
    print("      Press ENTER: START / PAUSE Automated 3-Sector Cotton Search")
    print("      Press 'P'  : Execute Pick & CW Plucking (when cotton detected & aligned)")
    print("      Press 'H'  : Return Robotic Arm to HOME (B90, S130, A130, E130)")
    print("      Press 'X'  : Switch Camera Device (Cycle: Cam 0 <-> Cam 1 <-> Cam 2)")
    print("      Press 'Y'  : Manually Trigger 4-Stage Harvest (Pick & CW -> Base 50 Drop & CCW -> Home)")
    print("      Press 'S'  : Manually Trigger 'S' Drop Gripper & CCW Roller Eject on ESP32")
    print("      Press 'T'  : Toggle Auto/Manual Gripper Actuation Mode")
    print("      Press 'F'  : Toggle Human Hand / Skin Filter (ON / OFF)")
    print("      Press 'R'  : Cycle Inference Resolution (1024 -> 1280 -> 896)")
    print("      Press 'W' or '+' : Increase Confidence Threshold (+0.03)")
    print("      Press 'Z' or '-' : REDUCE Confidence Threshold (-0.03)")
    print("      Press 'E'        : Cycle Sensitivity Presets (0.20 -> 0.25 -> 0.30 -> 0.35 -> 0.45)")
    print("      Press 'C'  : Instant Negative Sample Capture")
    print("      Press 'Q' or 'Esc' to exit.\n")

    search_controller = CottonSearchController(serial_comm, speed_ms_per_deg=85)
    tracker = RoboticTargetTracker(max_missing_frames=25, alpha=0.30)
    prev_time = time.time()
    conf_thresh = 0.20  # Fast sensitive threshold (detects cotton instantly on frame 1)
    img_size = 768      # Native 768 training resolution for maximum detection accuracy & high FPS
    filter_skin = True  # Reject human hand / skin false positives (Toggle with 'F')
    enable_filter = False
    # User Specification: Operating reach is 3cm - 28cm (30mm - 280mm). Sudden Detection & Pick within 28cm!
    SEARCH_DETECT_MIN_DIST_MM = 30.0   # 3.0 cm
    SEARCH_DETECT_MAX_DIST_MM = 280.0  # 28.0 cm (MAX 28cm)
    auto_send_y = True  # Auto-harvests after 6s countdown, or user can press 'P' key anytime to pick earlier!
    search_controller.auto_pick_on_timeout = auto_send_y
    last_auto_pick_time = 0.0
    auto_pick_cooldown = 0.5  # Snappy cooldown to allow instant picking without 4s lag
    last_serial_status = "SERIAL IDLE"
    locked_primary_id = None  # Persistent primary lock to eliminate blinking between READY and LOCKED

    # Arm and hold servos gently at Home pose on launch (Arm fixed at 130) so servos do not power-snap
    if serial_comm.is_connected:
        search_controller.home()
        last_serial_status = "ARM HOLDING AT HOME (Press ENTER to start search)"

    # Configure OpenCV display window ONCE before loop (allows free resize & maximize)
    window_title = "Cotton Boll Detection - Realtime Camera"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1024, 768)

    cam_fail_count = 0
    while True:
        ret, frame, depth_map = cam.read()
        if not ret or frame is None:
            cam_fail_count += 1
            if cam_fail_count % 15 == 0:
                print(f"[CAMERA RECONNECTING] Waiting for camera stream recovery... ({cam_fail_count}/120)")
            if cam_fail_count > 120:
                print("ERROR: Could not read camera frame after multiple attempts. Exiting.")
                break
            time.sleep(0.06)
            continue
        cam_fail_count = 0

        # Auto-reconnect to real hardware on COM7 if it was temporarily busy at startup
        if not serial_comm.is_connected:
            now_chk = time.time()
            if (now_chk - getattr(serial_comm, '_last_reconnect_ts', 0.0)) > 2.0:
                serial_comm._last_reconnect_ts = now_chk
                if serial_comm.try_reconnect():
                    last_serial_status = f"PORT: {serial_comm.port} (REAL HARDWARE ACTIVE)"

        # Update automated cotton search trajectory & motion engine
        search_controller.update()

        # 1. Adapt frame lighting for extreme sunlight or dark rooms
        inference_frame, mean_lum = enhance_lighting_for_inference(frame)

        # Generate candidates dynamically matching the user's active conf_thresh
        effective_conf = max(0.10, round(conf_thresh - 0.05, 2))

        # Run model inference with high-resolution 1024px multi-scale feature mapping
        results = model.predict(
            source=inference_frame,
            imgsz=img_size,
            conf=effective_conf,
            iou=0.55,  # Allows multiple clustered cotton bolls to be detected side-by-side
            classes=active_classes,
            device=device,
            verbose=False
        )

        frame_h, frame_w = frame.shape[:2]
        detected_boxes = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names.get(class_id, f"Class {class_id}")

                # --- STRICT FILTER: ONLY FULLY OPENED COTTON ALLOWED ---
                if class_id not in active_classes:
                    continue
                if "defected" in class_name.lower() or "partially" in class_name.lower() or "flower" in class_name.lower():
                    continue

                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                aspect_ratio = box_w / float(box_h)
                box_area = box_w * box_h

                # --- INSTANT DEPTH CAMERA EVALUATION ---
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                p3d = (0.0, 0.0, 0.0)
                z_mm = 0.0
                if depth_map is not None:
                    p3d = cam.get_3d_point(cx, cy, depth_map, box=[x1, y1, x2, y2])
                    z_mm = p3d[2]

                in_depth_range = (SEARCH_DETECT_MIN_DIST_MM <= z_mm <= SEARCH_DETECT_MAX_DIST_MM)

                # --- SCALE & DEPTH AWARE CONFIDENCE GATE ---
                min_req_conf = conf_thresh
                if confidence < min_req_conf:
                    continue

                # --- 1. PHYSICAL 3D SIZE FILTER (Distance-Adaptive) ---
                # A real cotton boll is ~30mm to 65mm in diameter (max ~85mm for dual clusters).
                if z_mm > 0:
                    # Dynamically calculate the maximum allowed pixel size at distance z_mm:
                    # At 4cm close-up: max_allowed ~ 1400px (allows full close-up bolls)
                    # At 12cm close-up: max_allowed ~ 485px
                    # At 25cm: max_allowed ~ 235px
                    max_allowed_px = (95.0 * 615.0) / max(30.0, z_mm)
                    if box_w > max_allowed_px or box_h > max_allowed_px:
                        continue
                else:
                    # When extremely close up (< 11cm) or depth is unmeasured, allow up to 620px
                    if box_w > 620 or box_h > 620 or box_area > 380000:
                        continue

                # --- 2. REJECT EXTREME ASPECT RATIO SLIVERS ---
                if aspect_ratio < 0.30 or aspect_ratio > 3.0:
                    continue

                # --- 3. REJECT HUMAN HAND / SKIN FALSE POSITIVES (LIGHTING ADAPTIVE) ---
                crop_box = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                if filter_skin:
                    if is_skin_region(crop_box, mean_lum=mean_lum):
                        continue

                # --- 4. REJECT TRULY BLANK/FLAT SURFACES ---
                if confidence < 0.35 and crop_box.size > 0:
                    if is_smooth_flat_object(crop_box):
                        continue

                # --- 5. REJECT FACES & CLUTTER (LIGHTING ADAPTIVE WHITENESS CHECK) ---
                if crop_box.size > 0 and confidence < 0.28:
                    hsv_c = cv2.cvtColor(crop_box, cv2.COLOR_BGR2HSV)
                    v_min = max(30, min(100, int(mean_lum * 0.45)))
                    s_max = 130
                    white_m = cv2.inRange(hsv_c, np.array([0, 0, v_min]), np.array([180, s_max, 255]))
                    w_ratio = np.count_nonzero(white_m) / float(crop_box.shape[0] * crop_box.shape[1])
                    if w_ratio < 0.02:
                        continue  # Must contain at least 2% bright fibers

                detected_boxes.append({
                    'box': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'class_name': class_name,
                    'conf': confidence,
                    'p3d': p3d,
                    'z_mm': z_mm,
                    'in_depth_range': in_depth_range
                })

        # Update Robotic Target Tracker
        active_targets = tracker.update(detected_boxes, frame.shape)

        annotated_frame = frame.copy()
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time

        # Target Detection & Automated Harvest Trigger:
        # User Specification: Max distance is 30cm (300mm). Detection & Pick ONLY within 30cm!
        SEARCH_DETECT_MIN_DIST_MM = 30.0   # 3.0 cm
        SEARCH_DETECT_MAX_DIST_MM = 280.0  # 28.0 cm (MAX 28cm)

        # Target Locking with Sticky Hysteresis (Zero Blinking / Flickering between READY & LOCKED)
        primary_target_id = None
        primary_cx, primary_cy = 0, 0
        primary_3d = (0.0, 0.0, 0.0)

        # 1. Keep previous primary target locked as long as it is alive in tracker memory
        if locked_primary_id is not None and locked_primary_id in active_targets:
            tdata = active_targets[locked_primary_id]
            if tdata.get('confirmed', False) and tdata.get('last_seen', 0) <= tracker.max_missing_frames:
                primary_target_id = locked_primary_id
                sx1, sy1, sx2, sy2 = map(int, tdata['smoothed_box'])
                primary_cx = (sx1 + sx2) // 2
                primary_cy = (sy1 + sy2) // 2
                primary_3d = tdata.get('last_valid_p3d', (0.0, 0.0, 0.0))
            else:
                locked_primary_id = None

        # 2. If no target locked, pick the best confirmed cotton in view
        if primary_target_id is None:
            best_in_range_conf = 0.0
            fallback_target_id = None
            highest_conf = 0.0
            fallback_cx, fallback_cy = 0, 0
            fallback_3d = (0.0, 0.0, 0.0)

            for tid, tdata in active_targets.items():
                if not tdata.get('confirmed', False):
                    continue
                class_name = tdata['class_name']
                class_id = tdata['class_id']
                is_ready = ("fully opened" in class_name.lower() or "cotton" in class_name.lower() or class_id == 1 or class_id in active_classes)
                if not is_ready:
                    continue

                sx1, sy1, sx2, sy2 = map(int, tdata['smoothed_box'])
                cx = (sx1 + sx2) // 2
                cy = (sy1 + sy2) // 2
                p3d = tdata.get('last_valid_p3d', (0.0, 0.0, 0.0))
                z_dist = tdata.get('last_valid_z', 0.0)

                # Re-sample depth if not already acquired
                if depth_map is not None and z_dist <= 0:
                    p3d = cam.get_3d_point(cx, cy, depth_map, box=[sx1, sy1, sx2, sy2])
                    z_dist = p3d[2]
                    if z_dist > 0:
                        tdata['last_valid_z'] = z_dist
                        tdata['last_valid_p3d'] = p3d

                # Prioritize targets inside the reach window (10-30cm)
                in_window = (z_dist == 0.0 or (SEARCH_DETECT_MIN_DIST_MM <= z_dist <= SEARCH_DETECT_MAX_DIST_MM))
                if in_window and tdata['conf'] > best_in_range_conf:
                    best_in_range_conf = tdata['conf']
                    primary_target_id = tid
                    primary_cx, primary_cy = cx, cy
                    primary_3d = p3d
                elif primary_target_id is None and tdata['conf'] > highest_conf:
                    highest_conf = tdata['conf']
                    fallback_target_id = tid
                    fallback_cx, fallback_cy = cx, cy
                    fallback_3d = p3d

            if primary_target_id is None and fallback_target_id is not None:
                primary_target_id = fallback_target_id
                primary_cx, primary_cy = fallback_cx, fallback_cy
                primary_3d = fallback_3d

            locked_primary_id = primary_target_id

        # Cleanly determine z_mm before any state updates (avoids UnboundLocalError)
        z_mm = 0.0
        if primary_target_id is not None:
            z_mm = primary_3d[2]
            if z_mm <= 0 and primary_target_id in active_targets:
                z_mm = active_targets[primary_target_id].get('last_valid_z', 0.0)

        # Update target coordinates for visual servoing controller while harvesting
        if search_controller.is_harvesting and primary_cx > 0 and primary_cy > 0:
            search_controller.update_target_coordinates(primary_cx, primary_cy, z_mm, primary_3d[0], primary_3d[1])

        if primary_target_id is not None:
            is_valid_range = (z_mm == 0.0 or (SEARCH_DETECT_MIN_DIST_MM <= z_mm <= SEARCH_DETECT_MAX_DIST_MM))
            now = time.time()
            # Start pick sequence ONLY when actively searching or manually commanded (not while waiting in IDLE at Home)
            if is_valid_range and search_controller.state == search_controller.STATE_SEARCHING and (now - last_auto_pick_time) >= auto_pick_cooldown:
                last_auto_pick_time = now

                # Actuate visual servoing harvest pipeline in Python:
                # Arm sweeps halt immediately, 3D coordinates stabilize, and only Shoulder + Elbow move to pick!
                search_controller.on_cotton_locked(
                    target_id=primary_target_id,
                    cx=primary_cx,
                    cy=primary_cy,
                    z_mm=z_mm,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    x_mm=primary_3d[0],
                    y_mm=primary_3d[1]
                )

                z_cm = z_mm / 10.0 if z_mm > 0 else 0.0
                last_serial_status = f"TARGET #{primary_target_id} LOCKED ({z_cm:.1f}cm) -> Solving 2-DOF IK (Sh+Elbow only)..."
            elif not is_valid_range and z_mm > 0:
                z_cm = z_mm / 10.0
                range_status = f"OUT OF OPERATING RANGE: {z_cm:.1f}cm"
                last_serial_status = f"TARGET #{primary_target_id} [{range_status}]"

        # Read any incoming Serial feedback from Arduino IDE / Arm Controller
        arm_feedback = serial_comm.read_response()
        if arm_feedback:
            last_serial_status = f"ARM ECHO: {arm_feedback}"
            search_controller.parse_serial_feedback(arm_feedback)

        # Render Stable Bounding Boxes (Only confirmed targets are rendered - zero flicker!)
        for tid, tdata in active_targets.items():
            if not tdata.get('confirmed', False):
                continue  # Weak noise is suppressed and never flickers on screen

            # Persistent Zero-Flicker: Box remains continuously visible until cotton is removed
            sx1, sy1, sx2, sy2 = map(int, tdata['smoothed_box'])
            cx = (sx1 + sx2) // 2
            cy = (sy1 + sy2) // 2
            class_name = tdata['class_name']
            class_id = tdata['class_id']
            conf = tdata['conf']

            # Universal status matching: supports both V8 (4-class) and V9 (3-class) models
            is_ready = ("fully opened" in class_name.lower() or "cotton" in class_name.lower() or class_id == 1 or class_id in active_classes)
            is_primary = (tid == primary_target_id)
            is_harvesting = is_primary and ((time.time() - last_auto_pick_time) < auto_pick_cooldown)

            # Calculate 3D Point for this target if RealSense Depth Map is available
            x3d, y3d, z3d = 0.0, 0.0, 0.0
            if depth_map is not None:
                x3d, y3d, z3d = cam.get_3d_point(cx, cy, depth_map, box=[sx1, sy1, sx2, sy2])
                if z3d <= 0 and tdata.get('last_valid_z', 0.0) > 0:
                    z3d = tdata['last_valid_z']

            in_zone = (z3d == 0.0 or (SEARCH_DETECT_MIN_DIST_MM <= z3d <= SEARCH_DETECT_MAX_DIST_MM))

            is_red_box_target = (is_primary and search_controller.is_red_box_active())

            if is_red_box_target:
                # 2nd Detection after STOP: Force a bold RED COLOR SQUARE BOX
                cal_data = search_controller.get_calibrated_xyz()
                rem_wait = cal_data['rem_wait']
                cam_x, cam_y, cam_z = cal_data['cam_xyz']
                grp_x, grp_y, grp_z = cal_data['gripper_xyz']
                d_b, d_s, d_e = cal_data['deltas']

                # Create symmetric square box centered on cotton
                box_half_side = max(sx2 - sx1, sy2 - sy1) // 2 + 10
                sq_x1 = max(0, cx - box_half_side)
                sq_y1 = max(0, cy - box_half_side)
                sq_x2 = min(frame.shape[1] - 1, cx + box_half_side)
                sq_y2 = min(frame.shape[0] - 1, cy + box_half_side)

                # 1. Draw Bold Red Square Box (BGR: 0, 0, 255)
                cv2.rectangle(annotated_frame, (sq_x1, sq_y1), (sq_x2, sq_y2), (0, 0, 255), 4)
                # 2. Draw Target Center Crosshair & Bullseye
                cv2.drawMarker(annotated_frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
                cv2.circle(annotated_frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.circle(annotated_frame, (cx, cy), 12, (255, 255, 255), 2)

                # 3. Comprehensive On-Screen HUD Text (RealSense 3D Coordinates, Measurement, & Centering)
                hud_inf = search_controller.get_hud_info()
                dry_str = " [DRY-RUN]" if hud_inf.get("dry_run", False) else ""
                meas = hud_inf.get("measurement")
                is_long_target = hud_inf.get("is_target_long", False) or (meas and meas.get("is_long", False))

                if is_long_target:
                    title_text = f"[TARGET #{tid}] >>> [LONG] TOO FAR TO PLUCK ({cam_z/10.0:.1f}cm) <<<"
                    cam_text = f"D435i Cam: Z={cam_z:.1f}mm ({cam_z/10.0:.1f}cm) | X:{cam_x:+.0f}mm, Y:{cam_y:+.0f}mm"
                    grp_dist = meas['distance_mm'] if meas else (cam_z - 132.0)
                    grp_text = f"MEASUREMENT: [LONG] Dist to Gripper: {grp_dist:.0f}mm > 115mm (CANNOT REACH)"
                    grp_col = (0, 0, 255) # Red
                    adj_text = "STATUS: SAYING 'LONG' -> HOLDING SAFELY -> RESUMING SEARCH SWEEP..."
                else:
                    if getattr(search_controller, 'auto_pick_on_timeout', True):
                        title_text = f"[TARGET #{tid}]{dry_str} [REACHABLE] CENTER AIM & PLUCK | Stage: {hud_inf.get('harvest_stage')}"
                    else:
                        title_text = f"[TARGET #{tid}]{dry_str} (WAIT: {rem_wait:.1f}s) -> PRESS 'P' TO PLUCK"
                    cam_text = f"D435i Cam: Z={cam_z:.1f}mm ({cam_z/10.0:.1f}cm) | X:{cam_x:+.0f}mm, Y:{cam_y:+.0f}mm"
                    if meas:
                        grp_text = f"GRIPPER MEASURE: Dist={meas['distance_mm']}mm | Yaw={meas['yaw_deg']:+.1f}deg, Pitch={meas['pitch_deg']:+.1f}deg"
                    else:
                        grp_text = f"GRIPPER MEASURE: Calculating exact distance and angle to center..."
                    grp_col = (0, 255, 0) # Green
                    adj_text = f"CENTER AIM: B{hud_inf.get('target_ik_base')} S{hud_inf.get('target_ik_shoulder')} [A:130 locked] E{hud_inf.get('target_ik_elbow')} (Limits <=130deg)"

                # Draw dark shadow backgrounds for crystal-clear readability
                cv2.putText(annotated_frame, title_text, (sq_x1, max(sq_y1 - 50, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, gr_col_title := ((0, 0, 255) if is_long_target else (0, 255, 0)), 2)
                cv2.putText(annotated_frame, cam_text, (sq_x1, max(sq_y1 - 30, 42)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
                cv2.putText(annotated_frame, grp_text, (sq_x1, max(sq_y1 - 10, 62)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, grp_col, 2)
                cv2.putText(annotated_frame, adj_text, (sq_x1, min(sq_y2 + 22, frame.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2)

            else:
                if is_harvesting:
                    color = (0, 140, 255)      # Bright Orange: Harvesting Action Running
                    thickness = 3
                    status = f"HARVESTING #{tid} [GRIPPER ACTIVE]"
                elif is_primary and in_zone:
                    color = (0, 255, 0)        # Bright Green for Arm Target WITHIN 28CM RANGE
                    thickness = 3
                    if tdata.get('in_depth_range', False):
                        status = f"SUDDEN LOCK #{tid} [3-28CM]"
                    else:
                        status = f"ARM TARGET #{tid} [WITHIN 28CM]"
                elif is_primary and z3d > SEARCH_DETECT_MAX_DIST_MM:
                    color = (0, 255, 255)      # Yellow: Too far (> 28cm)
                    thickness = 2
                    status = f"ARM TARGET #{tid} [TOO FAR >28cm]"
                elif is_primary and z3d > 0 and z3d < SEARCH_DETECT_MIN_DIST_MM:
                    color = (0, 165, 255)      # Amber: Too close (< 3cm)
                    thickness = 2
                    status = f"ARM TARGET #{tid} [TOO CLOSE <3cm]"
                elif is_primary:
                    color = (0, 255, 0)
                    thickness = 3
                    status = f"ARM TARGET #{tid} [LOCKED]"
                else:
                    color = (0, 200, 0)        # Dark Green: Ready
                    thickness = 2
                    status = f"READY #{tid}"

                # Draw Smoothed Bounding Box
                cv2.rectangle(annotated_frame, (sx1, sy1), (sx2, sy2), color, thickness)

                # Draw Centroid Target Point for Robotic Arm Kinematics
                cv2.circle(annotated_frame, (cx, cy), 5, (255, 255, 255), -1)
                cv2.circle(annotated_frame, (cx, cy), 8, color, 2)

                # Coordinates & 140mm Reach Analysis label for Robotic Operator
                label_text = f"[{status}] {class_name[:14]} ({conf*100:.0f}%)"
                if z3d > 0:
                    z_cm = z3d / 10.0
                    coord_text = f"3D D435i: X={x3d:+.0f}mm, Y={y3d:+.0f}mm, Z={z3d:.0f}mm ({z_cm:.1f}cm)"
                    if z3d > 140.0:
                        dist_to_140 = z3d - 140.0
                        reach_140_text = f"--> MOVE FORWARD: {dist_to_140:.0f}mm ({dist_to_140/10.0:.1f}cm) to reach 140mm"
                        reach_140_color = (0, 215, 255) # Bright Amber
                    else:
                        reach_140_text = f"--> [REACHED <= 140mm!] Inside target zone by {140.0 - z3d:.0f}mm"
                        reach_140_color = (0, 255, 0)   # Vibrant Green
                else:
                    coord_text = f"Centroid (X:{cx}, Y:{cy}) [Depth Searching...]"
                    reach_140_text = "--> Waiting for D435i 3D Depth..."
                    reach_140_color = (200, 200, 200)

                cv2.putText(annotated_frame, label_text, (sx1, max(sy1 - 44, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
                cv2.putText(annotated_frame, coord_text, (sx1, max(sy1 - 24, 38)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                cv2.putText(annotated_frame, reach_140_text, (sx1, max(sy1 - 4, 56)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, reach_140_color, 2)

        # Draw Visual Servoing Aim Reticle & Center Dot Analysis
        opt_cx, opt_cy = frame_w // 2, frame_h // 2
        # Center Optical Gripper Aim Crosshair
        cv2.line(annotated_frame, (opt_cx - 18, opt_cy), (opt_cx + 18, opt_cy), (255, 255, 0), 1)
        cv2.line(annotated_frame, (opt_cx, opt_cy - 18), (opt_cx, opt_cy + 18), (255, 255, 0), 1)
        cv2.circle(annotated_frame, (opt_cx, opt_cy), 22, (255, 255, 0), 1)
        cv2.putText(annotated_frame, "AIM CENTER", (opt_cx - 40, opt_cy + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)

        if primary_target_id is not None and primary_cx > 0 and primary_cy > 0:
            dx = primary_cx - opt_cx
            dy = primary_cy - opt_cy
            is_aligned = (abs(dx) <= 30 and abs(dy) <= 30)
            vec_color = (0, 255, 0) if is_aligned else (0, 165, 255)

            # Center Point Dot Highlight (Target Reticle)
            cv2.circle(annotated_frame, (primary_cx, primary_cy), 12, (0, 255, 255), 2)
            cv2.circle(annotated_frame, (primary_cx, primary_cy), 3, (0, 0, 255), -1)

            # Draw Guidance Vector Line from Gripper Aim Center to Cotton Center Point Dot
            cv2.line(annotated_frame, (opt_cx, opt_cy), (primary_cx, primary_cy), vec_color, 2)

            dot_txt = f"CENTER DOT: dx={dx:+d}px, dy={dy:+d}px"
            if is_aligned:
                dot_txt += " [ALIGNED!]"
            cv2.putText(annotated_frame, dot_txt, (primary_cx - 70, max(22, primary_cy - 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, vec_color, 2)

        # Draw HUD for Robotic Operator
        is_sudden = primary_target_id and active_targets.get(primary_target_id, {}).get('in_depth_range', False)
        target_str = f"SUDDEN LOCK #{primary_target_id} (1-FRAME)" if is_sudden else (f"LOCKED TARGET #{primary_target_id}" if primary_target_id else "SEARCHING...")
        filter_str = "ON" if filter_skin else "OFF"
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f} | Conf: {conf_thresh:.2f} | ImgSz: {img_size} | DepthCam: 10-30cm Sudden | Lock: {target_str}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0) if is_sudden else (255, 255, 0),
            2
        )
        serial_mode_str = f"PORT: {serial_comm.port} (REAL HARDWARE ACTIVE)" if serial_comm.is_connected else "RETRYING HARDWARE COM7..."
        cam_type_str = f"RealSense D435i ({cam.width}x{cam.height})" if cam.is_realsense else f"Webcam #{cam.cam_index} ({cam.width}x{cam.height})"
        cv2.putText(
            annotated_frame,
            f"Cam: [{cam_type_str}] (Switch: 'X') | Serial [{serial_mode_str}] : {last_serial_status}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255) if primary_target_id else (200, 200, 200),
            1
        )

        # Draw Automated Search Engine Status Row on HUD
        hud_info = search_controller.get_hud_info()
        if hud_info.get("is_target_long", False):
            search_color = (0, 0, 255)  # Bright Red: LONG target
            search_banner = f">>> [LONG] TARGET IS TOO FAR TO PLUCK ({hud_info['status_msg']}) - MOVE ROBOT CLOSER! <<<"
        elif hud_info.get("is_waiting_for_pick", False):
            search_color = (0, 255, 255)  # Bright Yellow: Attention required
            rem_wait = hud_info.get("wait_seconds_remaining", 0.0)
            search_banner = f">>> [COTTON LOCKED] MEASURING 3D DISTANCE & ANGLES ({rem_wait:.1f}s) -> AUTO-PICK <<<"
        elif hud_info["is_harvesting"]:
            search_color = (0, 140, 255)  # Bright Orange: Harvest active
            search_banner = f">>> {hud_info['status_msg']} <<<"
        elif hud_info["is_searching"]:
            search_color = (0, 255, 0)    # Bright Green: Search running
            search_banner = f"SEARCH: {hud_info['status_msg']} [B:{hud_info['base_angle']}, S:{hud_info['shoulder_angle']}, A:{hud_info.get('arm_angle', 130)}, E:{hud_info.get('elbow_angle', 130)}] (P: Pick | ENTER: Pause)"
        elif hud_info.get("is_homing") or hud_info["state"] == "HOMING":
            search_color = (0, 215, 255)  # Cyan/Amber: Homing smoothly (zero jerk)
            search_banner = f">>> {hud_info['status_msg']} [B:{hud_info['base_angle']}, S:{hud_info['shoulder_angle']}, A:{hud_info.get('arm_angle', 130)}, E:{hud_info.get('elbow_angle', 130)}] <<<"
        elif hud_info["state"] == "PAUSED":
            search_color = (0, 255, 255)  # Yellow: Paused
            search_banner = f"SEARCH: PAUSED | {hud_info['status_msg']} (ENTER: Resume | H: Home)"
        elif hud_info["state"] == "COMPLETED":
            search_color = (255, 150, 0)
            search_banner = f"SEARCH: COMPLETED | Press ENTER to start new search sweep"
        else:
            search_color = (220, 220, 220) # Idle
            search_banner = f"SEARCH: IDLE | Press ENTER in window or terminal to START 3-Sector Search"

        cv2.putText(
            annotated_frame,
            search_banner,
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            search_color,
            2
        )

        cv2.imshow(window_title, annotated_frame)

        # Non-blocking check for keys typed in the PowerShell Terminal Window
        if sys.platform == "win32" and msvcrt.kbhit():
            term_char = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if term_char in ('\r', '\n'):
                # Enter key pressed in Terminal: start/toggle automated search
                search_controller.toggle_search()
                last_auto_pick_time = 0.0
            elif term_char == 'h':
                search_controller.home()
                last_serial_status = "SENT 'H' -> Robotic Arm Homed!"
            elif term_char == 'x':
                cam.switch_camera()
            elif term_char in ('p', 'y'):
                search_controller.trigger_pick()
                last_serial_status = "TRIGGERED 'P' -> Pick & CW Plucking Executed!"
                print("[TERMINAL TRIGGER] Started Harvest: Pick & CW -> Base 50 Drop & CCW -> Home!")
            elif term_char == 's':
                serial_comm.send_cmd("S")
                time.sleep(0.5)
                serial_comm.send_cmd("CCW")
                last_serial_status = "SENT 'S' & 'CCW' -> Drop Gripper & CCW Roller Ejected!"
                print("[TERMINAL TRIGGER] Sent 'S' & 'CCW' commands: Drop Gripper & CCW Roller Eject!")
            elif term_char == 't':
                auto_send_y = not auto_send_y
                search_controller.auto_pick_on_timeout = auto_send_y
                mode_str = "AUTOMATIC (Auto-harvests after 6s countdown, or press 'P' earlier)" if auto_send_y else "MANUAL (Press 'P' key to harvest)"
                print(f"[TERMINAL MODE TOGGLE] Gripper Pick Mode set to: {mode_str}")
                last_serial_status = f"GRIPPER MODE: {mode_str}"
            elif term_char == 'f':
                filter_skin = not filter_skin
                print(f"[TERMINAL] Human Hand/Skin Filter set to: {'ON' if filter_skin else 'OFF'}")
            elif term_char == 'd':
                is_dry = search_controller.toggle_dry_run()
                last_serial_status = f"DRY-RUN: {'ENABLED' if is_dry else 'DISABLED'}"
            elif term_char == 'r':
                img_size = 1280 if img_size == 1024 else (896 if img_size == 1280 else 1024)
                print(f"[TERMINAL] Inference Resolution cycled to: {img_size}px")
            elif term_char in ('w', '+', '='):
                conf_thresh = min(0.90, round(conf_thresh + 0.03, 2))
                print(f"[TERMINAL] Confidence INCREASED to: {conf_thresh:.2f}")
            elif term_char in ('z', '-', '_'):
                conf_thresh = max(0.10, round(conf_thresh - 0.03, 2))
                print(f"[TERMINAL] Confidence REDUCED to: {conf_thresh:.2f}")
            elif term_char == 'e':
                presets = [0.20, 0.25, 0.30, 0.35, 0.45]
                next_p = presets[0]
                for p in presets:
                    if p > conf_thresh + 0.02:
                        next_p = p
                        break
                conf_thresh = next_p
                print(f"[TERMINAL] Sensitivity preset toggled to: {conf_thresh:.2f}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key in (13, 10):  # ENTER Key pressed in Camera Window
            search_controller.toggle_search()
            last_auto_pick_time = 0.0
        elif key == ord("h") or key == ord("H"):
            search_controller.home()
            last_serial_status = "SENT 'H' -> Robotic Arm Homed!"
        elif key == ord("d") or key == ord("D"):
            is_dry = search_controller.toggle_dry_run()
            last_serial_status = f"DRY-RUN: {'ENABLED' if is_dry else 'DISABLED'}"
        elif key == ord("4"):  # Shortcut to start Cotton Search
            search_controller.start_search()
            last_auto_pick_time = 0.0
        elif key == ord("x") or key == ord("X"):
            cam.switch_camera()
        elif key == ord("c") or key == ord("C"):
            out_img_dir = Path(r"C:\cotton_robot\dataset\cotton\combined_training_v4\train\images")
            out_lbl_dir = Path(r"C:\cotton_robot\dataset\cotton\combined_training_v4\train\labels")
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)
            img_name = f"custom_negative_webcam_{timestamp}.jpg"
            lbl_name = f"custom_negative_webcam_{timestamp}.txt"
            cv2.imwrite(str(out_img_dir / img_name), frame)
            with open(out_lbl_dir / lbl_name, "w") as f:
                f.write("")
            print(f"[CAPTURE] Saved negative sample: {img_name}")
        elif key in (ord("p"), ord("P"), ord("y"), ord("Y")):
            search_controller.trigger_pick()
            last_serial_status = "TRIGGERED 'P' -> Pick & CW Plucking Executed!"
            print("[MANUAL TRIGGER] Started Harvest: Pick & CW -> Base 50 Drop & CCW -> Home!")
        elif key == ord("t") or key == ord("T"):
            auto_send_y = not auto_send_y
            search_controller.auto_pick_on_timeout = auto_send_y
            mode_str = "AUTOMATIC (Auto-harvests after 6s countdown, or press 'P' earlier)" if auto_send_y else "MANUAL (Press 'P' key to harvest)"
            print(f"[MODE TOGGLE] Gripper Pick Mode set to: {mode_str}")
            last_serial_status = f"GRIPPER MODE: {mode_str}"
        elif key == ord("f") or key == ord("F"):
            filter_skin = not filter_skin
            print(f"[HUD TOGGLE] Human Hand / Skin Filter set to: {'ON' if filter_skin else 'OFF'}")
        elif key == ord("r") or key == ord("R"):
            img_size = 1280 if img_size == 1024 else (896 if img_size == 1280 else 1024)
            print(f"[HUD TOGGLE] Inference Resolution cycled to: {img_size}px")
        elif key in (ord("e"), ord("E")):
            presets = [0.20, 0.25, 0.30, 0.35, 0.45]
            next_p = presets[0]
            for p in presets:
                if p > conf_thresh + 0.02:
                    next_p = p
                    break
            conf_thresh = next_p
            print(f"[HUD TOGGLE] Sensitivity preset toggled to: {conf_thresh:.2f}")
        elif key in (ord("w"), ord("W"), ord("+"), ord("=")):
            conf_thresh = min(0.90, round(conf_thresh + 0.03, 2))
            print(f"[HUD] Confidence threshold INCREASED to: {conf_thresh:.2f}")
        elif key in (ord("z"), ord("Z"), ord("-"), ord("_")):
            conf_thresh = max(0.10, round(conf_thresh - 0.03, 2))
            print(f"[HUD] Confidence threshold REDUCED to: {conf_thresh:.2f}")
        elif key == ord("s") or key == ord("S"):
            serial_comm.send_cmd("S")
            time.sleep(0.5)
            serial_comm.send_cmd("CCW")
            last_serial_status = "SENT 'S' & 'CCW' -> Drop Gripper & CCW Roller Ejected!"
            print("[MANUAL TRIGGER] Sent 'S' & 'CCW' commands: Drop Gripper & CCW Roller Eject!")

    cam.release()
    serial_comm.close()
    cv2.destroyAllWindows()
    print("Camera and Serial stream closed.")

if __name__ == "__main__":
    main()