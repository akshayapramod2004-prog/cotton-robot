import os
import sys
from pathlib import Path

DATASET_ROOT = Path(r"C:\cotton_robot\dataset\cotton\combined_training_v4")

def remove_synthetic():
    print("=" * 75)
    print(" REMOVING OPENCV SYNTHETIC BACKGROUND NEGATIVES")
    print("=" * 75)

    splits = ["train", "valid", "test"]
    total_imgs_removed = 0
    total_lbls_removed = 0

    for s in splits:
        img_dir = DATASET_ROOT / s / "images"
        lbl_dir = DATASET_ROOT / s / "labels"

        targets = [f for f in img_dir.glob("*.*") if "white_obj" in f.name.lower() or "hard_neg" in f.name.lower()]
        print(f"\nProcessing [{s.upper()}]: Found {len(targets)} synthetic files.")

        imgs_removed = 0
        lbls_removed = 0

        for img in targets:
            lbl = lbl_dir / f"{img.stem}.txt"
            
            # Remove image
            try:
                img.unlink()
                imgs_removed += 1
            except Exception as e:
                print(f"  [ERROR] Could not delete image {img.name}: {e}")

            # Remove label
            if lbl.exists():
                try:
                    lbl.unlink()
                    lbls_removed += 1
                except Exception as e:
                    print(f"  [ERROR] Could not delete label {lbl.name}: {e}")

        print(f"  [{s.upper()}] Done: Removed {imgs_removed} images, {lbls_removed} labels.")
        total_imgs_removed += imgs_removed
        total_lbls_removed += lbls_removed

    print("\n" + "=" * 75)
    print(f" SUMMARY: Successfully removed {total_imgs_removed} images and {total_lbls_removed} labels.")
    print("=" * 75)

if __name__ == "__main__":
    remove_synthetic()
