# Cotton Harvester AI Dataset Guide

This document explains the dataset structure, class definitions, and annotation standards for integrating new real-world images (webcam / depth camera) into the cotton harvesting model.

---

## 1. Active Target Classes

The dataset uses **3 target classes** for the robotic harvester:

| Class ID | Class Name | Description & Visual Features | Action |
| :---: | :--- | :--- | :---: |
| **`0`** | **`Defected boll - Don't pick`** | Discolored, diseased, rotting, brownish, withered, or insect-damaged cotton bolls. | **REJECT** |
| **`1`** | **`Fully opened boll - Ready to pick`** | Mature, white, fluffy, cloud-like cotton bolls with visible open lobes. | **PICK** |
| **`2`** | **`Partially opened - Not ready to pick`** | Immature green cotton bolls, tightly closed carpels, or bolls just beginning to split. | **REJECT** |

### Background Negatives (False Positive Suppression)
* **What are they?** Images of human hands, arms, leaves, soil, white plates, paper, cups, clothing, walls, and flowers.
* **How are they labeled?** Leave the corresponding `.txt` file **completely empty (0 bytes)**.
* **Why?** This explicitly teaches YOLO that non-cotton objects are background and must NOT be detected.

---

## 2. Directory Structure

```
combined_training_v4/
│
├── data.yaml              <-- Dataset configuration file
│
├── train/
│   ├── images/            <-- Training images (.jpg, .png)
│   └── labels/            <-- YOLO bounding box labels (.txt)
│
├── valid/
│   ├── images/            <-- Validation images
│   └── labels/            <-- Validation labels
│
└── test/
    ├── images/            <-- Test images
    └── labels/            <-- Test labels
```

---

## 3. YOLO Annotation Format

Every image in `images/` (e.g. `webcam_frame_001.jpg`) has a matching text file in `labels/` with the same name (`webcam_frame_001.txt`).

Each line in the `.txt` file represents **one bounding box**:
```text
<class_id> <x_center> <y_center> <width> <height>
```
* All coordinates are **normalized between 0.0 and 1.0** relative to the image width and height:
  * `class_id` : Integer (`0`, `1`, or `2`)
  * `x_center` : Bounding box center X coordinate / image_width
  * `y_center` : Bounding box center Y coordinate / image_height
  * `width`    : Bounding box width / image_width
  * `height`   : Bounding box height / image_height

### Example (`train/labels/cotton_sample_01.txt`):
```text
1 0.482310 0.512400 0.185200 0.221400
0 0.721400 0.389200 0.114200 0.134500
```
*(Line 1 is a Fully opened boll; Line 2 is a Defected boll in the same image)*

---

## 4. Recommended Annotation Tools

You can use any standard YOLO-compatible annotation tool:
1. **AnyLabeling / LabelImg (Desktop - Recommended):**
   * Download AnyLabeling: https://github.com/vietanhdev/anylabeling
   * Select format: **YOLO**
   * Pre-define labels: `0: Defected boll - Don't pick`, `1: Fully opened boll - Ready to pick`, `2: Partially opened - Not ready to pick`.
2. **Roboflow (Web):**
   * Upload images $\rightarrow$ Annotate $\rightarrow$ Export in **YOLOv8 / YOLO11 format**.
3. **CVAT (Computer Vision Annotation Tool):**
   * Export format: **YOLO 1.1**.

---

## 5. Integrating New Webcam Images with Python

Use the included helper script:
```bash
python import_webcam_images.py --source /path/to/new_webcam_folder
```
This automatically copies the images into `train/images/`, validates image dimensions, creates template label files, and checks dataset integrity.
