import os
import sys
import yaml
from pathlib import Path
from collections import Counter, defaultdict
# pyrefly: ignore [missing-import]
from PIL import Image

def verify_dataset(dataset_dir):
    dataset_path = Path(dataset_dir).resolve()
    print("=" * 75)
    print(f" DATASET INTEGRITY & VALIDATION REPORT")
    print(f" Target Directory: {dataset_path}")
    print("=" * 75)

    yaml_file = dataset_path / "data.yaml"
    if not yaml_file.exists():
        print(f"[ERROR] data.yaml not found at {yaml_file}")
        return

    with open(yaml_file, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    nc = data_cfg.get("nc", 0)
    names = data_cfg.get("names", [])
    print(f"Classes ({nc}): {names}")

    splits = ["train", "valid", "test"]
    split_stats = {}
    all_image_hashes = defaultdict(list)
    total_issues = 0

    for split in splits:
        print(f"\n--- Checking Split: [{split.upper()}] ---")
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"

        if not img_dir.exists():
            print(f"[WARN] Image folder for '{split}' does not exist: {img_dir}")
            continue

        images = list(img_dir.glob("*.*"))
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [p for p in images if p.suffix.lower() in image_exts]

        stats = {
            "total_images": len(images),
            "corrupt_images": 0,
            "missing_labels": 0,
            "bg_negative_images": 0,
            "positive_images": 0,
            "total_objects": 0,
            "class_counts": Counter(),
            "invalid_class_ids": 0,
            "out_of_bounds_bbox": 0,
            "zero_area_bbox": 0,
            "label_format_errors": 0
        }

        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        label_map = {lbl.stem: lbl for lbl in labels}

        # Check for orphan labels
        img_stems = {img.stem for img in images}
        orphan_labels = [lbl for lbl in labels if lbl.stem not in img_stems]
        if orphan_labels:
            print(f"  [WARN] {len(orphan_labels)} orphan label file(s) found without matching images.")
            total_issues += len(orphan_labels)

        for img_path in images:
            # 1. Image Readability Verification
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                stats["corrupt_images"] += 1
                print(f"  [ERROR] Corrupt image: {img_path.name} - {e}")
                total_issues += 1
                continue

            lbl_path = label_map.get(img_path.stem)
            if not lbl_path or not lbl_path.exists():
                stats["missing_labels"] += 1
                stats["bg_negative_images"] += 1
                continue

            # Read label file
            try:
                lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            except Exception as e:
                stats["label_format_errors"] += 1
                print(f"  [ERROR] Cannot read label file: {lbl_path.name} - {e}")
                total_issues += 1
                continue

            if not lines or all(not line.strip() for line in lines):
                stats["bg_negative_images"] += 1
                continue

            img_has_valid_obj = False
            for line_no, line in enumerate(lines, 1):
                parts = line.strip().split()
                if not parts:
                    continue

                if len(parts) < 5:
                    stats["label_format_errors"] += 1
                    print(f"  [ERROR] Invalid label line (expected >=5 elements): {lbl_path.name}:L{line_no} -> '{line}'")
                    total_issues += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                except ValueError:
                    stats["label_format_errors"] += 1
                    print(f"  [ERROR] Non-numeric label values: {lbl_path.name}:L{line_no}")
                    total_issues += 1
                    continue

                if cls_id < 0 or cls_id >= nc:
                    stats["invalid_class_ids"] += 1
                    print(f"  [ERROR] Class ID {cls_id} out of range [0, {nc-1}] in {lbl_path.name}:L{line_no}")
                    total_issues += 1

                if w <= 0 or h <= 0:
                    stats["zero_area_bbox"] += 1
                    print(f"  [ERROR] Zero/negative bbox size (w={w}, h={h}) in {lbl_path.name}:L{line_no}")
                    total_issues += 1

                x1, y1 = xc - w / 2, yc - h / 2
                x2, y2 = xc + w / 2, yc + h / 2
                if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
                    stats["out_of_bounds_bbox"] += 1

                stats["class_counts"][cls_id] += 1
                stats["total_objects"] += 1
                img_has_valid_obj = True

            if img_has_valid_obj:
                stats["positive_images"] += 1
            else:
                stats["bg_negative_images"] += 1

        split_stats[split] = stats

        print(f"  Total Images        : {stats['total_images']}")
        print(f"  Positive Images     : {stats['positive_images']}")
        print(f"  Negative (BG) Images: {stats['bg_negative_images']}")
        print(f"  Total Bounding Boxes: {stats['total_objects']}")

        print(f"  Class Breakdown:")
        for cid in range(nc):
            name = names[cid] if cid < len(names) else f"Class_{cid}"
            cnt = stats['class_counts'][cid]
            pct = (cnt / stats['total_objects'] * 100) if stats['total_objects'] > 0 else 0
            print(f"    - ID {cid} ({name}): {cnt} objects ({pct:.1f}%)")

        if stats['corrupt_images'] > 0 or stats['invalid_class_ids'] > 0 or stats['label_format_errors'] > 0:
            print(f"  [WARN] Issues found in {split}: Corrupt Images: {stats['corrupt_images']}, Invalid Class IDs: {stats['invalid_class_ids']}, BBox Errors: {stats['label_format_errors']}")

    print("\n" + "=" * 75)
    print(" SUMMARY & HEALTH CHECKS")
    print("=" * 75)
    if total_issues == 0:
        print("[OK] DATASET HEALTH STATUS: CLEAN & READY FOR TRAINING!")
    else:
        print(f"[WARN] DATASET HEALTH STATUS: {total_issues} ISSUE(S) DETECTED! Review details above.")
    print("=" * 75)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else r"C:\cotton_robot\dataset\cotton\combined_training_v4"
    verify_dataset(target)
