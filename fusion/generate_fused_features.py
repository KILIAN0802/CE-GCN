import pandas as pd
import numpy as np
import os
import sys
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

# --- CẤU HÌNH  ---
CONFIG = {
    # 1. Folder chứa 5899 file .npy gốc (đã trích xuất từ video)
    "RAW_KEYPOINTS_DIR": "/home/ibmelab/Documents/GG/VSLRecognition/CTRGCN/data/keypoints", 
    
    # 2. Folder chứa các file CSV (trong csv tên file là .mp4)
    "CSV_DIR": "/home/ibmelab/Documents/GG/VSLRecognition/CTRGCN/data/MultiVSL200/labelCenter",
    
    # 3. Folder đích xuất ra
    "OUTPUT_ROOT": "/home/ibmelab/Documents/GG/VSLRecognition/CTRGCN/data/fused_features1",
    
    # 4. Tên file CSV
    "TRAIN_CSV": "train_labels.csv",
    "VAL_CSV": "val_labels.csv",
    "TEST_CSV": "test_labels.csv",
    
    # 5. Config Augmentation
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

# ---------------------------------------------------------
# LOGIC CHÍNH 
# ---------------------------------------------------------
def process_split(split_name, csv_filename, is_train=False):
    print(f"\n>>> ĐANG XỬ LÝ: {split_name.upper()} (Map từ {csv_filename})")
    
    csv_path = os.path.join(CONFIG["CSV_DIR"], csv_filename)
    if not os.path.exists(csv_path):
        print(f"[WARN] Không tìm thấy {csv_path}. Bỏ qua.")
        return

    save_dir = os.path.join(CONFIG["OUTPUT_ROOT"], f"{split_name}_fused_features")
    os.makedirs(save_dir, exist_ok=True)
    
    # Đọc CSV (Cột 0: filename.mp4, Cột 1: label)
    df = pd.read_csv(csv_path, header=None, names=['filename', 'label'])
    
    new_csv_rows = []
    missing_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename_mp4 = row['filename']
        label = row['label']
        
        # --- FIX: CHUYỂN .mp4 SANG .npy ---
        # Lấy tên gốc (bỏ đuôi cũ) và thêm đuôi .npy
        base_name = os.path.splitext(filename_mp4)[0] # "video_001"
        filename_npy = base_name + ".npy"              # "video_001.npy"
        
        # Tìm file trong folder raw
        src_path = os.path.join(CONFIG["RAW_KEYPOINTS_DIR"], filename_npy)
        
        if not os.path.exists(src_path):
            # Thử tìm trường hợp file gốc không có đuôi .npy (đề phòng)
            src_path_alt = os.path.join(CONFIG["RAW_KEYPOINTS_DIR"], base_name)
            if os.path.exists(src_path_alt):
                src_path = src_path_alt
            else:
                # Nếu vẫn không thấy thì bỏ qua
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

    # Lưu CSV mới
    new_csv_path = os.path.join(save_dir, f"{split_name}_label.csv")
    pd.DataFrame(new_csv_rows).to_csv(new_csv_path, index=False, header=False)
    
    print(f"[DONE] Đã lưu {len(new_csv_rows)} mẫu vào {save_dir}")
    if missing_count > 0:
        print(f"[WARN] Có {missing_count} file trong CSV không tìm thấy file .npy tương ứng!")

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Train
    process_split("train", CONFIG["TRAIN_CSV"], is_train=True)
    # 2. Val
    process_split("val", CONFIG["VAL_CSV"], is_train=False)
    # 3. Test
    process_split("test", CONFIG["TEST_CSV"], is_train=False)