# Autonomous Cotton Harvester Robot System

An autonomous vision and robotic manipulation system designed for detecting fully opened cotton bolls using computer vision (YOLO11s with Texture-Based Augmentations) and actuating an ESP32-controlled 7-servo robotic arm with an Intel RealSense D435i depth camera.

---

## Hardware Architecture
* **Vision Sensor:** Intel RealSense D435i Depth Camera (640x480 @ 30 FPS)
* **Robotic Arm Controller:** ESP32 Microcontroller (USB Serial on COM7 @ 115200 Baud)
* **Servo Motors:**
  * 60kg Servo: GPIO 13 (Base rotation)
  * 30kg Servo: GPIO 12 (Shoulder)
  * 25kg Servo: GPIO 27 (Elbow)
  * 15kg Servo: GPIO 14 (Wrist pitch)
  * 3x MG90S Servos: GPIO 26, 25, 33 (3-finger cotton boll gripper)

---

## Project Structure
* `webcam_demo.py` — Main real-time computer vision, depth distance measurement, and robotic arm actuation controller.
* `realsense_camera.py` — Intel RealSense D435i pipeline and 3D spatial coordinate projection ($X, Y, Z\text{ mm}$).
* `serial_controller.py` — Serial communication bridge to ESP32 with automatic fallback to Virtual/Simulated mode.
* `weights/best.pt` — Production-grade V9 Texture-Trained YOLO11s model (87.8% mAP50).
* `esp32_cotton_arm_controller/` — Arduino C++ firmware for the ESP32 7-servo robotic arm.
* `train.py` & `resume_training.py` — Training pipeline with Albumentations texture augmentations (CLAHE, unsharp masking, grayscale de-biasing).
* `requirements.txt` — Python dependencies.

---

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Upload ESP32 Firmware
1. Open `esp32_cotton_arm_controller/esp32_cotton_arm_controller.ino` in Arduino IDE.
2. Ensure the `ESP32Servo` library is installed via the Arduino Library Manager.
3. Select board: **ESP32 Dev Module**, port: **COM7**, and click **Upload**.
4. **Close the Arduino IDE Serial Monitor tab** before running Python.

### 3. Launch Vision & Robot Arm Controller
```bash
python webcam_demo.py
```

### Keyboard Shortcuts in Demo:
* **`Y`** : Manually send pick command (`'Y'`) to ESP32 gripper.
* **`T`** : Toggle between Auto-Pick and Manual mode.
* **`W`** : Increase confidence threshold (+0.05).
* **`S`** : Decrease confidence threshold (-0.05).
* **`Q`** or **`Esc`** : Exit cleanly.
