import os
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset

# ==========================================
# 1. HÀM CHUẨN HÓA
# ==========================================
def center_pose(data):
    """
    Đưa skeleton về gốc tọa độ (0,0) dựa trên khớp đầu tiên (thường là Hông/Spine)
    Input: (C, T, V)
    """
    C, T, V = data.shape
    # Chỉ xử lý 3 kênh đầu (Joint: x, y, z)
    if C >= 3:
        # Lấy tọa độ khớp số 0 tại frame đầu tiên làm gốc
        origin = data[0:3, :, 0:1].mean(axis=1, keepdims=True)  # (3, 1, 1)
        # Trừ đi gốc
        data[0:3, :, :] = data[0:3, :, :] - origin
    return data

# ==========================================
# 2. CÁC HÀM AUGMENTATION
# ==========================================
def random_shift(data):
    """Dịch chuyển toàn bộ skeleton sang trái/phải/lên/xuống ngẫu nhiên"""
    C, T, V = data.shape
    if C >= 3:
        offset = np.random.uniform(-0.1, 0.1, 3) # (3,)
        offset = offset[:, None, None] # (3, 1, 1)
        data[0:3, :, :] += offset
    return data

def random_move(data, angle_cand=[-10., -5., 0., 5., 10.]):
    """Xoay nhẹ skeleton"""
    return data # Tạm thời giữ nguyên

def auto_pading(data_numpy, size, random_pad=False):
    """Hàm Resize Frame thông minh"""
    C, T, V = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V))
        data_numpy_paded[:, begin:begin + T, :] = data_numpy
        return data_numpy_paded
    else:
        if random_pad:
            sample_idx = np.random.choice(T, size, replace=False)
            sample_idx.sort()
        else:
            sample_idx = np.linspace(0, T - 1, size).astype(int)
        return data_numpy[:, sample_idx, :]

# ==========================================
# 3. DATASET CLASS CHÍNH
# ==========================================
class FeatureReader(Dataset):
    def __init__(self, data_path, split_file, window_size=64, 
                 num_classes=200, debug=False, normalization=False, 
                 select_channel=None, # <--- [NEW] THÊM THAM SỐ CHỌN KÊNH
                 **kwargs):
        """
        **kwargs: Hứng các tham số augmentation từ Config
        """
        self.feature_dir = data_path
        self.window_size = window_size
        self.normalization = normalization
        self.select_channel = select_channel # <--- [NEW] LƯU LẠI
        
        # Lấy tham số Augmentation
        self.random_choose = kwargs.get('random_choose', False)
        self.random_shift  = kwargs.get('random_shift', False)
        self.random_move   = kwargs.get('random_move', False)
        
        # Đọc CSV
        self.data_list = pd.read_csv(split_file, header=None)
        if debug:
            self.data_list = self.data_list[:100]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 1. Load Info (Giữ nguyên)
        row = self.data_list.iloc[index]
        file_name = row[0]
        label = int(row[1])

        vid = file_name.replace(".mp4", "") if file_name.endswith(".mp4") else file_name
        if not vid.endswith(".npy"):
            vid = vid + ".npy"

        # 2. Load Data
        npy_path = os.path.join(self.feature_dir, vid)
        try:
            data_numpy = np.load(npy_path) 
        except FileNotFoundError:
            return torch.zeros((9, self.window_size, 46)), label

        # ===============================================
        # [MODIFIED] LẤY 6 KÊNH (VEL + BONE) & KÍCH SÓNG
        # ===============================================
        if self.select_channel is not None:
            # Lọc lấy kênh
            data_numpy = data_numpy[self.select_channel, :, :]
            
            # --- CHIẾN THUẬT MỚI: NHÂN TO CẢ VELOCITY VÀ BONE ---
            # Kiểm tra xem có đang lấy kênh Velocity (3,4,5) hoặc Bone (6,7,8) không
            # Nếu có bất kỳ kênh nào từ 3 đến 8, ta nhân 10 lên
            check_channels = [3, 4, 5, 6, 7, 8]
            if any(ch in self.select_channel for ch in check_channels):
                 data_numpy = data_numpy * 10.0
                 
        # 3. Normalization (Giữ nguyên)
        if self.normalization:
            data_numpy = center_pose(data_numpy)

        # 4. Augmentation (Giữ nguyên)
        if self.random_shift:
            data_numpy = random_shift(data_numpy)

        # 5. Temporal Resize (Giữ nguyên)
        data_numpy = auto_pading(data_numpy, self.window_size, random_pad=self.random_choose)

        # 6. Convert to Tensor
        feat = torch.from_numpy(data_numpy.astype(np.float32))
        
        return feat, label