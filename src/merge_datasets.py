"""
merge_datasets.py
Merges Ambulance + Emergency Vehicles datasets with class remap.

Final classes:
  0: ambulance
  1: non-ambulance

Drops: '-' (empty), 'Emergency-Vehicles' (mixed = confusing)
"""

import shutil
from pathlib import Path

# ============ CONFIG ============
PROJECT_ROOT = Path(__file__).parent.parent   # E:\Ambulance
SOURCE_DATA = PROJECT_ROOT / "source_data"
OUTPUT_DIR  = PROJECT_ROOT / "merged_dataset"

# Class remap: {source_class_id: target_class_id, None to drop}
AMBULANCE_CLASS_MAP = {
    0: 0,    # Ambulance -> ambulance
    1: 1,    # Normal Automobiles -> non-ambulance
}

EMERGENCY_CLASS_MAP = {
    0: None,  # '-' (empty) -> DROP
    1: None,  # Emergency-Vehicles (mixed) -> DROP
    2: 1,     # Non-emergency -> non-ambulance
}

# Split mapping: {source_split: target_split}
AMBULANCE_SPLIT_MAP = {
    'train': 'train',
    'valid': 'valid',
}

EMERGENCY_SPLIT_MAP = {
    'train': 'train',
    'test':  'valid',   # no valid folder in Emergency, use test
}

FINAL_CLASS_NAMES = ['ambulance', 'non-ambulance']
# ================================


def auto_detect_folder(base_dir, pattern):
    """Find a subfolder matching a glob pattern."""
    matches = list(base_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No folder matching '{pattern}' in {base_dir}")
    return matches[0]


def remap_label(src, dst, class_map):
    """Remap class IDs in a label file. Returns (kept, dropped)."""
    if not src.exists():
        dst.touch()
        return 0, 0

    with open(src) as f:
        lines = f.readlines()

    kept_lines, dropped = [], 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        src_cls = int(parts[0])
        tgt = class_map.get(src_cls)
        if tgt is None:
            dropped += 1
            continue
        parts[0] = str(tgt)
        kept_lines.append(' '.join(parts) + '\n')

    with open(dst, 'w') as f:
        f.writelines(kept_lines)
    return len(kept_lines), dropped


def copy_split(source_root, src_split, tgt_split, prefix, class_map, output_dir):
    """Copy images + remapped labels from one split."""
    src_img = source_root / src_split / 'images'
    src_lbl = source_root / src_split / 'labels'

    if not src_img.exists():
        print(f"  [SKIP] {source_root.name}/{src_split} - no images folder")
        return 0, 0, 0

    if not src_lbl.exists():
        print(f"  [WARN] {source_root.name}/{src_split} - no labels folder")

    dst_img = output_dir / tgt_split / 'images'
    dst_lbl = output_dir / tgt_split / 'labels'
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    imgs, kept_total, dropped_total = 0, 0, 0
    for img in src_img.iterdir():
        if img.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue

        # Prefix to avoid filename collisions
        new_img_name = f"{prefix}_{img.name}"
        new_lbl_name = f"{prefix}_{img.stem}.txt"

        shutil.copy2(img, dst_img / new_img_name)
        kept, dropped = remap_label(
            src_lbl / f"{img.stem}.txt",
            dst_lbl / new_lbl_name,
            class_map
        )
        imgs += 1
        kept_total += kept
        dropped_total += dropped

    print(f"  {prefix}/{src_split} -> {tgt_split}: "
          f"{imgs} imgs, {kept_total} kept, {dropped_total} dropped")
    return imgs, kept_total, dropped_total


def write_data_yaml(output_dir, class_names):
    yaml_path = output_dir / 'data.yaml'
    content = (
        f"path: {output_dir.absolute().as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f"\n[+] Generated: {yaml_path}")


def main():
    print(f"Merging into: {OUTPUT_DIR}\n")

    # Auto-detect source folders
    try:
        amb_dir = auto_detect_folder(SOURCE_DATA, "Ambulance*")
        eme_dir = auto_detect_folder(SOURCE_DATA, "Emergency*")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print(f"Ambulance source: {amb_dir.name}")
    print(f"Emergency source: {eme_dir.name}\n")

    # Clean output if exists
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        ans = input(f"{OUTPUT_DIR.name} mein content hai. Delete & recreate? (y/n): ")
        if ans.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(OUTPUT_DIR)

    # Stats accumulators
    stats = {
        'amb_train': (0, 0), 'amb_val': (0, 0),
        'eme_train': (0, 0), 'eme_val': (0, 0),
    }

    print("=== Ambulance dataset ===")
    for src, tgt in AMBULANCE_SPLIT_MAP.items():
        _, kept, dropped = copy_split(amb_dir, src, tgt, 'amb',
                                       AMBULANCE_CLASS_MAP, OUTPUT_DIR)
        key = f'amb_{"train" if tgt == "train" else "val"}'
        stats[key] = (stats[key][0] + kept, stats[key][1] + dropped)

    print("\n=== Emergency Vehicles dataset ===")
    for src, tgt in EMERGENCY_SPLIT_MAP.items():
        _, kept, dropped = copy_split(eme_dir, src, tgt, 'eme',
                                       EMERGENCY_CLASS_MAP, OUTPUT_DIR)
        key = f'eme_{"train" if tgt == "train" else "val"}'
        stats[key] = (stats[key][0] + kept, stats[key][1] + dropped)

    # Write final data.yaml
    write_data_yaml(OUTPUT_DIR, FINAL_CLASS_NAMES)

    # Summary
    train_kept = stats['amb_train'][0] + stats['eme_train'][0]
    train_drop = stats['amb_train'][1] + stats['eme_train'][1]
    val_kept   = stats['amb_val'][0] + stats['eme_val'][0]
    val_drop   = stats['amb_val'][1] + stats['eme_val'][1]

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"TRAIN: {train_kept} annotations kept, {train_drop} dropped")
    print(f"  From Ambulance: {stats['amb_train'][0]}")
    print(f"  From Emergency: {stats['eme_train'][0]}")
    print(f"\nVAL:   {val_kept} annotations kept, {val_drop} dropped")
    print(f"  From Ambulance: {stats['amb_val'][0]}")
    print(f"  From Emergency: {stats['eme_val'][0]}")
    print(f"\nFinal classes: {FINAL_CLASS_NAMES}")
    print(f"Output folder:  {OUTPUT_DIR}")
    print(f"\n[+] DONE. Next: write train.py and run training.")


if __name__ == "__main__":
    main()