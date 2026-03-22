"""
Script to generate label CSV files for MM-WLAuslan dataset.
Format: {class_id}_{view}_{modality}.mp4

Usage:
    python prepare_mm_wlauslan_labels.py \
        --video_dir /mnt/sda1/VSLR_Storage/MM-WLAuslan/videos \
        --output_dir ./data1
"""

import os
import csv
import argparse


def parse_split_arg(raw_value):
    """Parse comma-separated split names into a clean list."""
    return [item.strip() for item in raw_value.split(',') if item.strip()]


def extract_class_id(filename):
    """Extract class ID from filename like 00000_kf_rgb.mp4"""
    parts = filename.replace('.mp4', '').split('_')
    if len(parts) >= 1:
        try:
            return int(parts[0])
        except ValueError:
            return None
    return None


def scan_folder(folder_path):
    """Scan folder and return list of (filename, class_id) tuples"""
    samples = []
    
    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder not found: {folder_path}")
        return samples
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):
            class_id = extract_class_id(filename)
            if class_id is not None:
                samples.append((filename, class_id))
    
    return samples


def collect_samples(video_dir, split_names, group_name):
    """Collect samples from a list of split folders under video_dir."""
    samples = []
    print(f"[{group_name}] Using splits: {', '.join(split_names) if split_names else '(none)'}")
    for split in split_names:
        folder_path = os.path.join(video_dir, split)
        split_samples = scan_folder(folder_path)
        samples.extend(split_samples)
        print(f"  - {split}: {len(split_samples)} videos")
    return samples


def generate_csv(output_dir, train_samples, val_samples, test_samples):
    """Generate CSV files for train, val, and test splits"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for train/val/test features
    train_dir = os.path.join(output_dir, 'train_fused_features')
    val_dir = os.path.join(output_dir, 'val_fused_features')
    test_dir = os.path.join(output_dir, 'test_fused_features')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Write train_label.csv
    train_csv_path = os.path.join(train_dir, 'train_label.csv')
    with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file_name', 'label_id'])
        writer.writeheader()
        for filename, class_id in sorted(train_samples):
            writer.writerow({'file_name': filename, 'label_id': class_id})
    print(f"✓ Created: {train_csv_path} ({len(train_samples)} samples)")
    
    # Write val_label.csv
    val_csv_path = os.path.join(val_dir, 'val_label.csv')
    with open(val_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file_name', 'label_id'])
        writer.writeheader()
        for filename, class_id in sorted(val_samples):
            writer.writerow({'file_name': filename, 'label_id': class_id})
    print(f"✓ Created: {val_csv_path} ({len(val_samples)} samples)")
    
    # Write test_label.csv
    test_csv_path = os.path.join(test_dir, 'test_label.csv')
    with open(test_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file_name', 'label_id'])
        writer.writeheader()
        for filename, class_id in sorted(test_samples):
            writer.writerow({'file_name': filename, 'label_id': class_id})
    print(f"✓ Created: {test_csv_path} ({len(test_samples)} samples)")
    
    print(f"\nSummary:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")
    print(f"  Total: {len(train_samples) + len(val_samples) + len(test_samples)} samples")


def main(video_dir, output_dir, train_splits, val_splits, test_splits):
    """Main function to process MM-WLAuslan dataset"""
    
    print(f"Scanning MM-WLAuslan dataset...")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Show available folders for quick sanity-check.
    if os.path.exists(video_dir):
        available_dirs = sorted(
            [name for name in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, name))]
        )
        print(f"Available subfolders: {', '.join(available_dirs)}\n")

    train_samples = collect_samples(video_dir, train_splits, "TRAIN")
    val_samples = collect_samples(video_dir, val_splits, "VAL")
    test_samples = collect_samples(video_dir, test_splits, "TEST")

    print("\nFound:")
    print(f"  train (combined): {len(train_samples)} videos")
    print(f"  val (combined):   {len(val_samples)} videos")
    print(f"  test (combined):  {len(test_samples)} videos\n")
    
    # Generate CSV files
    generate_csv(output_dir, train_samples, val_samples, test_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate label CSV files for MM-WLAuslan")
    parser.add_argument("--video_dir", required=True, help="Path to MM-WLAuslan videos folder")
    parser.add_argument("--output_dir", default="./data1", help="Output directory for CSV files")
    parser.add_argument(
        "--train_splits",
        default="train",
        help="Comma-separated folder names used as train split (e.g. train or testSTU)",
    )
    parser.add_argument(
        "--val_splits",
        default="valid",
        help="Comma-separated folder names used as val split",
    )
    parser.add_argument(
        "--test_splits",
        default="testTW,testSTU,testSYN,testTED",
        help="Comma-separated folder names used as test split",
    )
    
    args = parser.parse_args()
    
    main(
        args.video_dir,
        args.output_dir,
        parse_split_arg(args.train_splits),
        parse_split_arg(args.val_splits),
        parse_split_arg(args.test_splits),
    )
