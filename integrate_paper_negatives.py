import os
import shutil
import time
from pathlib import Path
from PIL import Image

SRC_DIR = Path(r"C:\Users\Akshaya_J\Downloads\paper.v1i.yolov11\train\images")
DEST_IMG_DIR = Path(r"C:\cotton_robot\dataset\cotton\combined_training_v4\train\images")
DEST_LBL_DIR = Path(r"C:\cotton_robot\dataset\cotton\combined_training_v4\train\labels")

def integrate_negatives():
    print("=" * 75)
    print(" INTEGRATING PAPER NEGATIVE BACKGROUND IMAGES INTO DATASET")
    print("=" * 75)
    print(f"Source Folder : {SRC_DIR}")
    print(f"Destination   : {DEST_IMG_DIR}")
    print("=" * 75)

    if not SRC_DIR.exists():
        print(f"[ERROR] Source folder does not exist: {SRC_DIR}")
        return

    DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DEST_LBL_DIR.mkdir(parents=True, exist_ok=True)

    images = list(SRC_DIR.glob("*.*"))
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in images if p.suffix.lower() in supported_exts]

    print(f"Found {len(images)} source images to integrate...")

    copied = 0
    skipped = 0
    corrupt = 0
    t0 = time.time()

    for idx, img_path in enumerate(images, 1):
        # Quick validation
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            print(f"[WARN] Skipping corrupt image: {img_path.name} ({e})")
            corrupt += 1
            continue

        target_img_name = f"neg_paper_{img_path.name}"
        target_img_path = DEST_IMG_DIR / target_img_name
        target_lbl_path = DEST_LBL_DIR / f"{target_img_path.stem}.txt"

        if target_img_path.exists() and target_lbl_path.exists():
            skipped += 1
            continue

        # Copy image file
        shutil.copy2(img_path, target_img_path)

        # Write 0-byte negative label file
        target_lbl_path.write_text("", encoding="utf-8")

        copied += 1
        if copied % 500 == 0 or idx == len(images):
            print(f"  Progress: {idx}/{len(images)} processed ({copied} copied, {skipped} skipped)...")

    elapsed = time.time() - t0
    print("=" * 75)
    print(f" INTEGRATION COMPLETE in {elapsed:.1f}s")
    print(f"  Successfully Copied : {copied}")
    print(f"  Skipped (Existing)  : {skipped}")
    print(f"  Corrupt/Invalid     : {corrupt}")
    print("=" * 75)

if __name__ == "__main__":
    integrate_negatives()
