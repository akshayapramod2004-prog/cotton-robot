import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import train

if __name__ == "__main__":
    print("=" * 70)
    print(" RESUMING SINGLE-CLASS COTTON BOLL TRAINING FROM CHECKPOINT")
    print("=" * 70)

    # 1. Hook texture & lighting augmentations
    train.setup_robust_albumentations()

    # 2. Checkpoint path for latest training run
    LAST_WEIGHTS = Path(r"C:\cotton_robot\runs\detect\cotton_single_class_v1\weights\last.pt")

    if not LAST_WEIGHTS.exists():
        print(f"ERROR: Checkpoint not found at {LAST_WEIGHTS}")
        exit(1)

    print(f"Loading checkpoint state from: {LAST_WEIGHTS}")
    model = YOLO(str(LAST_WEIGHTS))

    # 3. Resume training seamlessly from the exact epoch where it was paused
    print("Resuming balance epochs (up to 60 total epochs)...")
    results = model.train(resume=True)

    # 4. Automatically deploy newly trained best.pt to weights/best.pt
    trained_best = Path(r"C:\cotton_robot\runs\detect\cotton_single_class_v1\weights\best.pt")
    target_best = Path(r"C:\cotton_robot\weights\best.pt")

    if trained_best.exists():
        print("\n" + "=" * 75)
        print(" TRAINING COMPLETE! DEPLOYING NEW WEIGHTS...")
        print("=" * 75)
        if target_best.exists():
            backup_path = Path(r"C:\cotton_robot\weights\best_backup.pt")
            shutil.copy2(target_best, backup_path)
            print(f"Backed up previous weights to: {backup_path}")

        shutil.copy2(trained_best, target_best)
        print(f"Successfully deployed: {trained_best} -> {target_best}")
        print("=" * 75)
