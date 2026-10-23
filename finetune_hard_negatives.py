import os
import shutil
import time
import random
import torch
from pathlib import Path
from ultralytics import YOLO
import train

if __name__ == "__main__":
    print("=" * 75)
    print(" HARD NEGATIVE RAPID FINE-TUNING FOR REAL-ROOM WHITE OBJECT REJECTION")
    print("=" * 75)

    # 1. Dataset roots
    DATASET_DIR = Path(r"C:\cotton_robot\dataset\cotton\combined_training_v4")
    TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
    TRAIN_LBL_DIR = DATASET_DIR / "train" / "labels"

    # 2. Check base weights
    BASE_WEIGHTS = Path(r"C:\cotton_robot\weights\best.pt")
    if not BASE_WEIGHTS.exists():
        BASE_WEIGHTS = Path(r"C:\cotton_robot\runs\detect\cotton_single_class_v1\weights\best.pt")

    print(f"Base Weights       : {BASE_WEIGHTS}")

    # 3. Collect ALL captured real room negatives
    real_neg_imgs = []
    for lbl in TRAIN_LBL_DIR.glob("neg_real_*.txt"):
        for ext in [".jpg", ".png", ".jpeg", ".webp"]:
            candidate = TRAIN_IMG_DIR / f"{lbl.stem}{ext}"
            if candidate.exists():
                real_neg_imgs.append(str(candidate).replace("\\", "/"))
                break

    for lbl in TRAIN_LBL_DIR.glob("custom_negative_*.txt"):
        for ext in [".jpg", ".png", ".jpeg", ".webp"]:
            candidate = TRAIN_IMG_DIR / f"{lbl.stem}{ext}"
            if candidate.exists():
                real_neg_imgs.append(str(candidate).replace("\\", "/"))
                break

    print(f"Real Room Negatives Found: {len(real_neg_imgs)} images")

    if len(real_neg_imgs) < 20:
        print(f"\n[BLOCKED] Only found {len(real_neg_imgs)} captured real negatives (need at least 20-100)!")
        print("Please run 'capture_negative_samples.py' first:")
        print("  .\\.venv312\\Scripts\\python.exe capture_negative_samples.py")
        exit(1)

    # 4. Assemble High-Impact Balanced Training Set (~960 images total)
    # 700 positive cotton + 150 paper negatives + 100% of newly captured real room negatives
    print("\nAssembling balanced high-impact training mix...")
    pos_imgs = []
    paper_neg_imgs = []
    for lbl in TRAIN_LBL_DIR.glob("*.txt"):
        if lbl.name.startswith("neg_real_") or lbl.name.startswith("custom_negative_"):
            continue
        for ext in [".jpg", ".png", ".jpeg", ".webp", ".JPG", ".PNG"]:
            img_path = TRAIN_IMG_DIR / f"{lbl.stem}{ext}"
            if img_path.exists():
                if lbl.stat().st_size > 0:
                    pos_imgs.append(str(img_path).replace("\\", "/"))
                else:
                    paper_neg_imgs.append(str(img_path).replace("\\", "/"))
                break

    random.seed(42)
    selected_pos = random.sample(pos_imgs, min(700, len(pos_imgs)))
    selected_paper = random.sample(paper_neg_imgs, min(150, len(paper_neg_imgs)))
    finetune_list = selected_pos + selected_paper + real_neg_imgs
    random.shuffle(finetune_list)

    print(f"  - Genuine Cotton Images   : {len(selected_pos)}")
    print(f"  - Real Room Hard Negatives: {len(real_neg_imgs)} (100% INCLUDED!)")
    print(f"  - Background Paper Images : {len(selected_paper)}")
    print(f"  - Total Fine-Tune Dataset : {len(finetune_list)} images (~120 batches)")

    txt_path = DATASET_DIR / "finetune_train.txt"
    txt_path.write_text("\n".join(finetune_list), encoding="utf-8")

    finetune_yaml = DATASET_DIR / "finetune_data.yaml"
    finetune_yaml.write_text(f"""path: {str(DATASET_DIR).replace('\\\\', '/')}
train: finetune_train.txt
val: valid/images
test: test/images
nc: 1
names:
  - Mature_cotton_balls
""", encoding="utf-8")

    # 5. Hook robust augmentations
    train.setup_robust_albumentations()

    # 6. Load model
    print(f"\nLoading model from: {BASE_WEIGHTS}")
    model = YOLO(str(BASE_WEIGHTS))

    # 7. Fast Fine-Tuning:
    # 6 epochs on ~960 images = ~15 minutes total!
    print("\nStarting Rapid Fine-Tuning (6 epochs, ~15 mins total)...")
    results = model.train(
        data=str(finetune_yaml),
        epochs=6,
        imgsz=768,
        batch=8 if torch.cuda.is_available() else 4,
        lr0=0.001,
        lrf=0.01,
        freeze=10,         # Freeze backbone (fast training, keeps core cotton features intact)
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0,
        project=r"C:\cotton_robot\runs\detect",
        name="cotton_finetune_negatives",
        exist_ok=True
    )

    # 8. Deploy newly fine-tuned weights
    trained_best = Path(r"C:\cotton_robot\runs\detect\cotton_finetune_negatives\weights\best.pt")
    target_best = Path(r"C:\cotton_robot\weights\best.pt")

    if trained_best.exists():
        print("\n" + "=" * 75)
        print(" FINE-TUNING COMPLETE! DEPLOYING NEW HARDENED WEIGHTS...")
        print("=" * 75)
        backup_path = Path(r"C:\cotton_robot\weights\best_pre_finetune_backup.pt")
        if target_best.exists():
            shutil.copy2(target_best, backup_path)
            print(f"Backed up previous weights to: {backup_path}")

        shutil.copy2(trained_best, target_best)
        print(f"Successfully deployed: {trained_best} -> {target_best}")
        print("webcam_demo.py is now updated with the newly fine-tuned weights!")
        print("=" * 75)
