import time
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

class RealSenseCamera:
    """
    Intel RealSense D435i Camera Wrapper:
    - Provides synchronized Color (RGB) and Depth streams aligned pixel-for-pixel.
    - Calculates 3D Spatial Coordinates (X_mm, Y_mm, Z_mm) for Robotic Arm Inverse Kinematics.
    - Gracefully falls back to standard OpenCV VideoCapture if RealSense is not available.
    """
    def __init__(self, width=1280, height=720, fps=30, cam_index=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam_index = cam_index
        self.is_realsense = False
        self.pipeline = None
        self.align = None
        self.depth_intrinsics = None
        self.cap = None

        self._initialize_camera()

    def _initialize_camera(self):
        # Check for connected Intel RealSense depth camera first
        if HAS_REALSENSE:
            try:
                ctx = rs.context()
                devices = ctx.query_devices()
                # If RealSense is connected and user didn't explicitly force index 0 (laptop webcam)
                if len(devices) > 0 and self.cam_index != 0:
                    device_name = devices[0].get_info(rs.camera_info.name)
                    serial_no = devices[0].get_info(rs.camera_info.serial_number)
                    usb_type = devices[0].get_info(rs.camera_info.usb_type_descriptor) if hasattr(rs.camera_info, 'usb_type_descriptor') else "3.x"
                    print(f"[REALSENSE OK] Connected to {device_name} (S/N: {serial_no}) on USB {usb_type}")

                    # On USB 2.1 connections, prioritize 640x480 for maximum reliability
                    if "2" in str(usb_type):
                        profiles_to_try = [
                            (640, 480, 30),
                            (424, 240, 30),
                            (848, 480, 30),
                            (1280, 720, 30)
                        ]
                    else:
                        profiles_to_try = [
                            (1280, 720, 30),
                            (848, 480, 30),
                            (640, 480, 30)
                        ]

                    started = False
                    for w, h, fps in profiles_to_try:
                        try:
                            self.pipeline = rs.pipeline()
                            config = rs.config()
                            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

                            profile = self.pipeline.start(config)
                            self.align = rs.align(rs.stream.color)

                            depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                            self.depth_intrinsics = depth_stream.get_intrinsics()

                            self.width, self.height, self.fps = w, h, fps
                            self.is_realsense = True
                            started = True
                            print(f"[REALSENSE MODE] Active Profile: {w}x{h} @ {fps}fps | 3D Depth Sensing ACTIVE!")
                            break
                        except Exception as pe:
                            if self.pipeline:
                                try:
                                    self.pipeline.stop()
                                except Exception:
                                    pass
                                self.pipeline = None

                    if started:
                        return
            except Exception as e:
                print(f"[REALSENSE WARN] Failed to start RealSense pipeline: {e}")

        # If user explicitly requested camera index 0 (or RealSense wasn't found), use OpenCV VideoCapture
        chosen_index = 0 if self.cam_index is None else self.cam_index
        cam_desc = "Built-in Laptop Webcam" if chosen_index == 0 else f"Camera Index {chosen_index}"
        print(f"[CAMERA INFO] Starting OpenCV Video Capture on {cam_desc} (Index {chosen_index})...")
        self.cap = cv2.VideoCapture(chosen_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(chosen_index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def switch_camera(self, target_index=None):
        """Switches between camera indices dynamically at runtime."""
        if self.is_realsense:
            print("[CAMERA] RealSense is currently active. Cannot switch OpenCV index.")
            return

        if target_index is None:
            # Cycle between 0, 1, and 2
            next_idx = 1 if self.cam_index == 0 else (2 if self.cam_index == 1 else 0)
        else:
            next_idx = target_index

        print(f"[CAMERA] Attempting switch to Camera Index {next_idx}...")
        new_cap = cv2.VideoCapture(next_idx)
        if new_cap is not None and new_cap.isOpened():
            ret, _ = new_cap.read()
            if ret:
                if self.cap:
                    self.cap.release()
                self.cap = new_cap
                self.cam_index = next_idx
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                print(f"[CAMERA OK] Successfully switched to Camera Index {next_idx}!")
                return
            new_cap.release()
        print(f"[CAMERA WARN] Camera Index {next_idx} not responding. Remaining on Index {self.cam_index}.")

    def read(self):
        """
        Returns:
            ret (bool): Success flag
            color_frame (np.ndarray): BGR image frame
            depth_frame (np.ndarray or None): Aligned depth frame (in mm), or None if standard webcam
        """
        if self.is_realsense and self.pipeline:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned_frames = self.align.process(frames)
                color_frame_rs = aligned_frames.get_color_frame()
                depth_frame_rs = aligned_frames.get_depth_frame()

                if not color_frame_rs or not depth_frame_rs:
                    return False, None, None

                color_image = np.asanyarray(color_frame_rs.get_data())
                depth_image = np.asanyarray(depth_frame_rs.get_data()) # Depth in millimeters

                return True, color_image, depth_image
            except Exception as e:
                print(f"[REALSENSE ERROR] Read frame error: {e}")
                try:
                    if self.pipeline:
                        self.pipeline.stop()
                except Exception:
                    pass
                self.pipeline = None
                time.sleep(0.5)
                # Attempt to reconnect RealSense up to 3 times before fallback
                for attempt in range(3):
                    print(f"[REALSENSE RECONNECT] Attempt {attempt+1}/3 to re-establish RealSense pipeline...")
                    self._initialize_camera()
                    if self.is_realsense:
                        print("[REALSENSE RECONNECT OK] Successfully reconnected RealSense!")
                        break
                    time.sleep(0.6)
                return False, None, None
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return ret, frame, None
        return False, None, None

    def get_3d_point(self, cx, cy, depth_image, box=None):
        """
        Calculates 3D coordinates (X_mm, Y_mm, Z_depth_mm) relative to camera lens.
        Includes robust bounding box aggregation and adaptive multi-radius hole filling
        to overcome IR speckle dropouts and motion blur on diffuse cotton fibers.
        """
        if depth_image is None or not self.is_realsense or self.depth_intrinsics is None:
            return 0.0, 0.0, 0.0

        h, w = depth_image.shape
        cx_int = int(np.clip(cx, 0, w - 1))
        cy_int = int(np.clip(cy, 0, h - 1))

        valid_depths = np.array([])

        # Priority 1: If bounding box provided, sample inner 70% of cotton boll
        if box is not None:
            bx1, by1, bx2, by2 = map(int, box)
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            # Sample central 70% of the boll to avoid background edges
            ix1 = max(0, int(bx1 + 0.15 * bw))
            ix2 = min(w, int(bx2 - 0.15 * bw))
            iy1 = max(0, int(by1 + 0.15 * bh))
            iy2 = min(h, int(by2 - 0.15 * bh))
            if ix2 > ix1 and iy2 > iy1:
                box_patch = depth_image[iy1:iy2, ix1:ix2]
                box_candidates = box_patch[box_patch > 0]
                if len(box_candidates) >= 5:
                    valid_depths = box_candidates

        # Priority 2: Adaptive search radius (2px -> 6px -> 14px -> 24px) around centroid
        if len(valid_depths) == 0:
            for radius in [2, 6, 14, 24]:
                ymin = max(0, cy_int - radius)
                ymax = min(h, cy_int + radius + 1)
                xmin = max(0, cx_int - radius)
                xmax = min(w, cx_int + radius + 1)
                patch = depth_image[ymin:ymax, xmin:xmax]
                candidate_depths = patch[patch > 0]
                if len(candidate_depths) >= 3:
                    valid_depths = candidate_depths
                    break

        if len(valid_depths) == 0:
            # OPTICAL PROXIMITY DEPTH FALLBACK FOR SHORT DISTANCE (< 11cm):
            # Below ~10.5cm, RealSense IR stereo disparity cannot match (blind spot).
            # We estimate 3D distance from known physical cotton boll scale (~55mm)
            # using pinhole camera projection: Z = (fx * 55mm) / max(w, h)
            if box is not None:
                bx1, by1, bx2, by2 = map(int, box)
                bw = max(1, bx2 - bx1)
                bh = max(1, by2 - by1)
                dim_px = max(bw, bh)
                if dim_px >= 140:  # Cotton boll is large close-up (within ~15cm)
                    fx = getattr(self.depth_intrinsics, 'fx', 615.0) if self.depth_intrinsics else 615.0
                    fy = getattr(self.depth_intrinsics, 'fy', 615.0) if self.depth_intrinsics else 615.0
                    ppx = getattr(self.depth_intrinsics, 'ppx', 320.0) if self.depth_intrinsics else 320.0
                    ppy = getattr(self.depth_intrinsics, 'ppy', 240.0) if self.depth_intrinsics else 240.0

                    z_est_mm = float(np.clip((fx * 55.0) / dim_px, 35.0, 115.0))
                    x_mm = round(((cx_int - ppx) * z_est_mm) / fx, 1)
                    y_mm = round(((cy_int - ppy) * z_est_mm) / fy, 1)
                    z_mm = round(z_est_mm, 1)
                    return x_mm, y_mm, z_mm

            return 0.0, 0.0, 0.0

        depth_mm = float(np.median(valid_depths))
        depth_m = depth_mm / 1000.0

        # Deproject 2D pixel (cx, cy) + depth -> 3D Point (X_m, Y_m, Z_m)
        point_3d_m = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [cx_int, cy_int], depth_m)
        x_mm = round(point_3d_m[0] * 1000.0, 1)
        y_mm = round(point_3d_m[1] * 1000.0, 1)
        z_mm = round(depth_mm, 1)

        return x_mm, y_mm, z_mm

    def release(self):
        if self.is_realsense and self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            print("[REALSENSE] Pipeline stopped cleanly.")
        elif self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            print("[CAMERA] Video capture released.")
