import pandas as pd
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import random
import math

current_dir = os.path.dirname(os.path.abspath(__file__)) # Đang ở fusion/
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from fusion.early_fusion import early_fusion
from fusion.interpolate import interpolate_missing
from fusion.kalman_filter import apply_kalman_filter
from fusion.normalize_pose import normalize_pose

# --- CẤU HÌNH MẶC ĐỊNH ---
# Có thể override bằng command-line args.
CONFIG = {
    # Folder keypoints .npy sau MediaPipe
    "RAW_KEYPOINTS_DIR": os.path.join(parent_dir, "data1", "keypoints"),

    # Root chứa các split train_fused_features/val_fused_features/test_fused_features
    "DATA_ROOT": os.path.join(parent_dir, "data1"),

    # Config Augmentation
    "NUM_AUG": 5,      # Train sẽ nhân 6 lần (1 gốc + 5 giả)
    "MAX_FRAMES": 64
}

# ---------------------------------------------------------
# HÀM HỖ TRỢ
# ---------------------------------------------------------
def aug_random_rotate(kpts):
    angle = random.uniform(-15, 15)
    rad = math.radians(angle)
    cos_val, sin_val = math.cos(rad), math.sin(rad)
    data = kpts.copy()
    for t in range(data.shape[0]):
        for v in range(data.shape[1]):
            x, z = data[t, v, 0], data[t, v, 2]
            data[t, v, 0] = x * cos_val - z * sin_val
            data[t, v, 2] = x * sin_val + z * cos_val
    return data

def process_one_sample(file_path, max_frames=64, augment=False):
    try:
        kpt = np.load(file_path) # (T, 46, 3)
        kpt = normalize_pose(kpt)
        kpt = interpolate_missing(kpt)
        if augment:
            kpt = aug_random_rotate(kpt)
        kpt = apply_kalman_filter(kpt)
        fused = early_fusion(kpt, max_frames=max_frames)
        return fused
    except Exception as e:
        print(f"[ERR] File {file_path}: {e}")
        return None


def resolve_source_path(raw_keypoints_dir, split_name, filename_mp4):
    """Find keypoints file for a video name in common layouts.

    Supported layouts:
    1) {raw_keypoints_dir}/{video}.npy
    2) {raw_keypoints_dir}/{split_name}/{video}.npy
    """
    base_name = os.path.splitext(str(filename_mp4).strip())[0]
    candidates = [
        os.path.join(raw_keypoints_dir, f"{base_name}.npy"),
        os.path.join(raw_keypoints_dir, split_name, f"{base_name}.npy"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path, f"{base_name}.npy", base_name

    return None, f"{base_name}.npy", base_name


def load_split_csv(csv_path):
    """Load CSV and normalize to columns: file_name, label_id."""
    if not os.path.exists(csv_path):
        print(f"[WARN] Không tìm thấy {csv_path}. Bỏ qua.")
        return None

    df = pd.read_csv(csv_path)

    # Chuẩn hóa theo format CE-GCN hiện tại
    rename_map = {}
    if "filename" in df.columns:
        rename_map["filename"] = "file_name"
    if "label" in df.columns:
        rename_map["label"] = "label_id"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Trường hợp CSV không có header thật sự -> đọc lại với header=None
    if "file_name" not in df.columns or "label_id" not in df.columns:
        df_no_header = pd.read_csv(csv_path, header=None)
        if df_no_header.shape[1] >= 2:
            df = df_no_header.iloc[:, :2].copy()
            df.columns = ["file_name", "label_id"]

    if "file_name" not in df.columns or "label_id" not in df.columns:
        raise ValueError(
            f"CSV {csv_path} phải có cột file_name,label_id (hoặc filename,label)."
        )

    return df[["file_name", "label_id"]]

# ---------------------------------------------------------
# LOGIC CHÍNH 
# ---------------------------------------------------------
def process_split(split_name, csv_path, save_dir, raw_keypoints_dir, is_train=False):
    print(f"\n>>> ĐANG XỬ LÝ: {split_name.upper()} (CSV: {csv_path})")

    df = load_split_csv(csv_path)
    if df is None:
        return

    os.makedirs(save_dir, exist_ok=True)

    new_csv_rows = []
    missing_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename_mp4 = row["file_name"]
        label = int(row["label_id"])

        src_path, filename_npy, base_name = resolve_source_path(raw_keypoints_dir, split_name, filename_mp4)
        if src_path is None:
            missing_count += 1
            continue

        # 1. Xử lý bản gốc
        feat = process_one_sample(src_path, CONFIG["MAX_FRAMES"], augment=False)
        if feat is not None:
            # Lưu file npy mới
            np.save(os.path.join(save_dir, filename_npy), feat)
            # Lưu vào CSV mới tên file .npy (để Feeder đọc được ngay)
            new_csv_rows.append([filename_npy, label])

        # 2. Augmentation (Chỉ Train)
        if is_train and CONFIG["NUM_AUG"] > 0:
            for i in range(CONFIG["NUM_AUG"]):
                aug_feat = process_one_sample(src_path, CONFIG["MAX_FRAMES"], augment=True)
                if aug_feat is not None:
                    aug_name = f"{base_name}_aug{i}.npy"
                    np.save(os.path.join(save_dir, aug_name), aug_feat)
                    new_csv_rows.append([aug_name, label])

    # Lưu CSV mới có header đúng theo VSLDataset
    new_csv_path = os.path.join(save_dir, f"{split_name}_label.csv")
    pd.DataFrame(new_csv_rows, columns=["file_name", "label_id"]).to_csv(
        new_csv_path, index=False
    )

    print(f"[DONE] Đã lưu {len(new_csv_rows)} mẫu vào {save_dir}")
    if missing_count > 0:
        print(f"[WARN] Có {missing_count} file trong CSV không tìm thấy file .npy tương ứng!")

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fused 9-channel features from keypoints")
    parser.add_argument("--raw_keypoints_dir", default=CONFIG["RAW_KEYPOINTS_DIR"], help="Directory containing keypoint .npy files")
    parser.add_argument("--data_root", default=CONFIG["DATA_ROOT"], help="Root directory containing train/val/test fused feature folders")
    parser.add_argument("--num_aug", type=int, default=CONFIG["NUM_AUG"], help="Number of augmented samples per train sample")
    parser.add_argument("--max_frames", type=int, default=CONFIG["MAX_FRAMES"], help="Max frames for early fusion")
    args = parser.parse_args()

    CONFIG["RAW_KEYPOINTS_DIR"] = os.path.abspath(args.raw_keypoints_dir)
    CONFIG["DATA_ROOT"] = os.path.abspath(args.data_root)
    CONFIG["NUM_AUG"] = args.num_aug
    CONFIG["MAX_FRAMES"] = args.max_frames

    train_dir = os.path.join(CONFIG["DATA_ROOT"], "train_fused_features")
    val_dir = os.path.join(CONFIG["DATA_ROOT"], "val_fused_features")
    test_dir = os.path.join(CONFIG["DATA_ROOT"], "test_fused_features")

    process_split(
        split_name="train",
        csv_path=os.path.join(train_dir, "train_label.csv"),
        save_dir=train_dir,
        raw_keypoints_dir=CONFIG["RAW_KEYPOINTS_DIR"],
        is_train=True,
    )
    process_split(
        split_name="val",
        csv_path=os.path.join(val_dir, "val_label.csv"),
        save_dir=val_dir,
        raw_keypoints_dir=CONFIG["RAW_KEYPOINTS_DIR"],
        is_train=False,
    )
    process_split(
        split_name="test",
        csv_path=os.path.join(test_dir, "test_label.csv"),
        save_dir=test_dir,
        raw_keypoints_dir=CONFIG["RAW_KEYPOINTS_DIR"],
        is_train=False,
    )