import os
import shutil
import hashlib
import time
from pathlib import Path
from PIL import Image

DEST_IMG_DIR = Path(r"C:\cotton_robot\dataset\cotton\combined_training_v4\train\images")
DEST_LBL_DIR = Path(r"C:\cotton_robot\dataset\cotton\combined_training_v4\train\labels")
DATASET_ROOT = Path(r"C:\cotton_robot\dataset\cotton\combined_training_v4")

SOURCES = [
    ("cd3", "Cotton-Detection.v3i", Path(r"C:\Users\Akshaya_J\Downloads\Cotton-Detection.v3i.yolov11\train")),
    ("c3", "cotton.v3i", Path(r"C:\Users\Akshaya_J\Downloads\cotton.v3i.yolov11\train")),
    ("c2r", "cotton.v2-release", Path(r"C:\Users\Akshaya_J\Downloads\cotton.v2-release.yolov11\train")),
    ("c1", "Cotton.v1i", Path(r"C:\Users\Akshaya_J\Downloads\Cotton.v1i.yolov11\train"))
]

def get_file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def integrate_cotton():
    print("=" * 75)
    print(" INTEGRATING NEW COTTON BOLL IMAGES (SINGLE CLASS: 0)")
    print("=" * 75)
    
    DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DEST_LBL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Index all existing images across all splits to prevent any duplicates
    print("Step 1: Indexing existing dataset checksums...")
    existing_md5s = set()
    for s in ["train", "valid", "test"]:
        s_img_dir = DATASET_ROOT / s / "images"
        if s_img_dir.exists():
            for img in s_img_dir.glob("*.*"):
                existing_md5s.add(get_file_md5(img))
    print(f"Indexed {len(existing_md5s)} existing images across the dataset.")

    # 2. Process each source
    print("\nStep 2: Processing and integrating downloaded sources...")
    seen_md5s = set(existing_md5s)

    stats = {
        "total_source_images": 0,
        "copied": 0,
        "skipped_existing_dataset": 0,
        "skipped_internal_dup": 0,
        "skipped_corrupt": 0,
        "total_boxes_remapped": 0
    }

    t0 = time.time()

    for prefix, s_name, s_path in SOURCES:
        s_img_dir = s_path / "images"
        s_lbl_dir = s_path / "labels"
        
        if not s_img_dir.exists():
            print(f"[WARN] Images folder not found: {s_img_dir}")
            continue

        images = [f for f in s_img_dir.glob("*.*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        stats["total_source_images"] += len(images)
        print(f"\n--- Source: {s_name} ({len(images)} images) ---")

        source_copied = 0
        source_skipped_dup = 0

        for img in images:
            # Check corrupt
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception as e:
                print(f"  [WARN] Corrupt image {img.name}: {e}")
                stats["skipped_corrupt"] += 1
                continue

            # Check duplicate by content hash
            img_md5 = get_file_md5(img)
            if img_md5 in existing_md5s:
                stats["skipped_existing_dataset"] += 1
                source_skipped_dup += 1
                continue
            if img_md5 in seen_md5s:
                stats["skipped_internal_dup"] += 1
                source_skipped_dup += 1
                continue

            seen_md5s.add(img_md5)

            # Target filenames
            target_img_name = f"{prefix}_{img.name}"
            target_img_path = DEST_IMG_DIR / target_img_name
            target_lbl_path = DEST_LBL_DIR / f"{target_img_path.stem}.txt"

            # Process label file: remap all classes to 0
            src_lbl = s_lbl_dir / f"{img.stem}.txt"
            new_lines = []

            if src_lbl.exists():
                src_lines = src_lbl.read_text(encoding="utf-8").strip().splitlines()
                for line in src_lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            # Clamp coordinates to [0.0, 1.0]
                            xc = max(0.0, min(1.0, float(parts[1])))
                            yc = max(0.0, min(1.0, float(parts[2])))
                            w = max(0.0, min(1.0, float(parts[3])))
                            h = max(0.0, min(1.0, float(parts[4])))
                            # Remap class ID to 0 (single class: fully_opened_cotton)
                            new_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                            stats["total_boxes_remapped"] += 1
                        except ValueError:
                            continue

            # Copy image
            shutil.copy2(img, target_img_path)

            # Write updated label
            target_lbl_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

            source_copied += 1
            stats["copied"] += 1

        print(f"  Result: Copied {source_copied}, Skipped Duplicates {source_skipped_dup}")

    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f" INTEGRATION COMPLETE in {elapsed:.1f}s")
    print(f"  Total Source Images Processed  : {stats['total_source_images']}")
    print(f"  Successfully Added to Train    : {stats['copied']}")
    print(f"  Skipped (Existing in Dataset) : {stats['skipped_existing_dataset']}")
    print(f"  Skipped (Internal Duplicate)  : {stats['skipped_internal_dup']}")
    print(f"  Total Bounding Boxes (Class 0) : {stats['total_boxes_remapped']}")
    print("=" * 75)

if __name__ == "__main__":
    integrate_cotton()
