import os
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data import augment
try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

# ---------------------------------------------------------------------------
# HARDENED TEXTURE & LIGHTING AUGMENTATION PIPELINE
# Specially tuned for:
# 1. Dark room ambient noise rejection (GaussNoise + ISONoise)
# 2. Extreme sunlight & glare rejection (RandomGamma + RandomBrightnessContrast)
# 3. Discrimination of white objects (CLAHE + Sharpen + ToGray forces true cotton fiber texture)
# ---------------------------------------------------------------------------
def setup_robust_albumentations():
    if not HAS_ALBUMENTATIONS:
        print("[AUGMENTATION INFO] Albumentations not installed. Using native Ultralytics HSV & geometry augmentations.")
        return

    orig_init = augment.Albumentations.__init__

    def patched_init(self, p=1.0, transforms=None, flip_idx=None):
        if transforms is None:
            transforms = [
                # 1. Micro-fiber extraction & contour sharpening
                A.CLAHE(clip_limit=(2.0, 4.0), tile_grid_size=(8, 8), p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.4),

                # 2. 35% Grayscale: Prevents model from using "plain white" as a trigger!
                # Forces model to learn physical fiber convolutions and lobe contours!
                A.ToGray(p=0.35),

                # 3. Extreme Illumination Invariance:
                # Wide dynamic range handles both dark rooms and washed-out direct sunlight
                A.RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.45, p=0.7),
                A.RandomGamma(gamma_limit=(40, 220), p=0.6),

                # 4. Sensor Gain & Low-Light Noise:
                # Simulates webcam high-gain ISO noise in dark rooms, preventing noise false positives
                A.GaussNoise(var_limit=(20.0, 70.0), p=0.4),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.2, 0.6), p=0.4),

                # 5. Defocus blur and spatial robustness
                A.Blur(blur_limit=3, p=0.2),
            ]
        orig_init(self, p=p, transforms=transforms, flip_idx=flip_idx)

    augment.Albumentations.__init__ = patched_init
    print("[TEXTURE & LIGHTING AUGMENTATION] Active: CLAHE, Sharpen, ToGray, RandomGamma, ISONoise")

if __name__ == "__main__":
    setup_robust_albumentations()

    # 1. Path to Single-Class dataset data.yaml
    DATA_YAML = r"C:\cotton_robot\dataset\cotton\combined_training_v4\data.yaml"

    # 2. Base weights to fine-tune from
    # Using clean yolo11s.pt to establish clean 1-class detection head
    BASE_WEIGHTS = Path(r"C:\cotton_robot\yolo11s.pt")

    if BASE_WEIGHTS.exists():
        print(f"Loading base YOLO11s weights from: {BASE_WEIGHTS}")
        model = YOLO(str(BASE_WEIGHTS))
    else:
        print("Downloading default YOLO11s weights...")
        model = YOLO("yolo11s.pt")

    print("=" * 75)
    print(" STARTING SINGLE-CLASS COTTON BOLL TRAINING (V1 SINGLE-CLASS)")
    print("=" * 75)
    print(f"Data config   : {DATA_YAML}")
    print(f"Target Class  : ONLY 1 Class -> 'Mature_cotton_balls'")
    print(f"Dataset Scope : Defected & partially opened bolls PURGED. ONLY fully opened cotton & clean scenes!")
    print(f"Robustness    : Hardened against Dark Rooms, Direct Sunlight, Faces & White Objects")
    print(f"Device        : CUDA GPU (device=0)" if torch.cuda.is_available() else "Device: CPU")
    print(f"Image size    : 768")
    print(f"Epochs        : 60")
    print("=" * 75)

    # 3. Model Training Run (Optimized for Small & Distant Cotton Detection)
    results = model.train(
        data=DATA_YAML,
        epochs=60,
        imgsz=768,
        batch=4,
        scale=0.7,         # Aggressive multi-scale zoom (trains small, distant bolls up to close-up macros)
        mosaic=1.0,        # 4-image stitching provides abundant small-object context
        close_mosaic=10,   # Disables mosaic for final 10 epochs for crisp, realistic small-boll boundary detection
        mixup=0.15,        # Blend scenes to reduce background bias
        copy_paste=0.25,   # Blends small cotton instances into diverse backgrounds
        box=8.5,           # Boosted box regression weight for pinpoint small-boll boundary accuracy
        cls=0.5,           # Balanced classification loss
        hsv_h=0.015,
        hsv_s=0.7,         # Broad saturation variation
        hsv_v=0.6,         # High brightness variation (dark rooms to blazing sunlight)
        workers=0,
        device=0 if torch.cuda.is_available() else "cpu",
        cos_lr=True,
        plots=True,
        project=r"C:\cotton_robot\runs\detect",
        name="cotton_single_class_v1",
        exist_ok=True
    )

    # 4. Automatically deploy newly trained best.pt to weights/best.pt
    trained_best = Path(r"C:\cotton_robot\runs\detect\cotton_single_class_v1\weights\best.pt")
    target_best = Path(r"C:\cotton_robot\weights\best.pt")

    if trained_best.exists():
        print("\n" + "=" * 75)
        print(" TRAINING COMPLETE! DEPLOYING NEW WEIGHTS...")
        print("=" * 75)
        if target_best.exists():
            backup_path = Path(r"C:\cotton_robot\weights\best_3class_backup.pt")
            shutil.copy2(target_best, backup_path)
            print(f"Backed up previous weights to: {backup_path}")

        shutil.copy2(trained_best, target_best)
        print(f"Successfully deployed: {trained_best} -> {target_best}")
        print("The robotic arm vision system (webcam_demo.py) is now ready to use the new 1-class model!")
        print("=" * 75)
