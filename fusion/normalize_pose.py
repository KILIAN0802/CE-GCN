import numpy as np

def normalize_pose(pose):
    """
    Input: pose (T, V, C)  -> Ví dụ: (T, 50, 3)
    Output: Normalized pose (T, V, C)
    """
    data = np.copy(pose)
    T, V, C = data.shape

    if V < 50:
        return data

    # 1. Hip-centric Normalization
    # Xác định 2 điểm hông (indices 48 và 49)
    HIP_L = 48
    HIP_R = 49

    x = data[:, :, 0] # (T, V)
    y = data[:, :, 1] # (T, V)
    z = data[:, :, 2] if C > 2 else np.zeros((T, V))
    
    # Tính trung điểm hông (Mid-hip)
    center_x = (x[:, HIP_L] + x[:, HIP_R]) / 2.0
    center_y = (y[:, HIP_L] + y[:, HIP_R]) / 2.0
    center_z = (z[:, HIP_L] + z[:, HIP_R]) / 2.0 if C > 2 else np.zeros(T)

    # Dịch chuyển toàn bộ 50 điểm mấu chốt về Mid-hip (0,0,0)
    data[:, :, 0] = x - center_x[:, None]
    data[:, :, 1] = y - center_y[:, None]
    if C > 2:
        data[:, :, 2] = z - center_z[:, None]

    return data.astype(np.float32)