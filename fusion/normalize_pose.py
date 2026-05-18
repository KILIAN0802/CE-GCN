import numpy as np

def normalize_pose(pose):
    """
    Input: pose (T, V, C)  -> Ví dụ: (T, 50, 3) hoặc (T, 50, 2)
    Output: Normalized pose (T, V, C)
    """
    data = np.copy(pose)
    T, V, C = data.shape

    if V < 50:
        return data

    HIP_L = 48
    HIP_R = 49

    # Tách tọa độ
    x = data[:, :, 0] # (T, V)
    y = data[:, :, 1] # (T, V)
    z = data[:, :, 2] if C > 2 else None

    # 1. Translation: Tính trung điểm hông (Mid-hip)
    center_x = (x[:, HIP_L] + x[:, HIP_R]) / 2.0
    center_y = (y[:, HIP_L] + y[:, HIP_R]) / 2.0
    center_z = (z[:, HIP_L] + z[:, HIP_R]) / 2.0 if z is not None else None

    # Dịch chuyển về Mid-hip (0,0,0)
    data[:, :, 0] = x - center_x[:, None]
    data[:, :, 1] = y - center_y[:, None]
    if z is not None:
        data[:, :, 2] = z - center_z[:, None]

    # 2. Scale Normalization
    # Tính khoảng cách Euclide giữa khớp 48 và 49 (độ rộng hông) tại mỗi frame
    diff_x = data[:, HIP_L, 0] - data[:, HIP_R, 0]
    diff_y = data[:, HIP_L, 1] - data[:, HIP_R, 1]
    
    if z is not None:
        diff_z = data[:, HIP_L, 2] - data[:, HIP_R, 2]
        hip_width = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
    else:
        hip_width = np.sqrt(diff_x**2 + diff_y**2)

    # Tính giá trị trung bình độ rộng hông của toàn bộ video
    mean_hip_width = np.mean(hip_width)

    # Chia toàn bộ tọa độ cho mean_hip_width để chống nhiễu scale (bất biến khoảng cách camera)
    if mean_hip_width > 1e-6:
        data[:, :, 0] /= mean_hip_width
        data[:, :, 1] /= mean_hip_width
        if z is not None:
            data[:, :, 2] /= mean_hip_width

    return data.astype(np.float32)