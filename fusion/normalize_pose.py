import numpy as np

def normalize_pose(pose):
    """
    Input: pose (T, V, C)  -> Ví dụ: (64, 46, 3)
           C=3 thường là (x, y, z) hoặc (x, y, score)
    Output: Normalized pose (T, V, C)
    """
    # Copy để không sửa trực tiếp vào dữ liệu gốc
    data = np.copy(pose)
    T, V, C = data.shape

    # INDEX (Dựa trên giả định của bạn)
    SHO_L = 42
    SHO_R = 43
    HIP_L = 44
    HIP_R = 45

    # 1. Tách X, Y (và Z nếu có)
    # Lưu ý: shape data là (T, V, C) nên phải slice kiểu data[:, :, :2]
    x = data[:, :, 0] # (T, V)
    y = data[:, :, 1] # (T, V)
    
    # 2. Tính toán Center (Trung bình cộng Vai và Hông)
    # shape các biến này sẽ là (T,)
    center_x = (x[:, SHO_L] + x[:, SHO_R] + x[:, HIP_L] + x[:, HIP_R]) / 4
    center_y = (y[:, SHO_L] + y[:, SHO_R] + y[:, HIP_L] + y[:, HIP_R]) / 4
    
    # 3. Dịch chuyển (Translation) -> Đưa body về gốc tọa độ (0,0)
    # Dùng broadcasting: (T, V) - (T, 1)
    data[:, :, 0] = x - center_x[:, None]
    data[:, :, 1] = y - center_y[:, None]
    
    # (Optional) Nếu muốn căn giữa cả trục Z (nếu C=3 là Z)
    # Thường với bài toán 2D->3D, ta chỉ cần căn giữa XY. 
    # Nhưng nếu Z là tọa độ tương đối, có thể giữ nguyên hoặc trừ đi Z trung bình.
    # Ở đây tôi giữ nguyên vị trí Z, chỉ scale nó.

    # 4. Tính toán Scale (Chiều rộng vai)
    # Tính khoảng cách Euclidean giữa 2 vai ở từng frame
    # dx, dy là vector nối 2 vai
    dx = x[:, SHO_L] - x[:, SHO_R]
    dy = y[:, SHO_L] - y[:, SHO_R]
    dist = np.sqrt(dx**2 + dy**2) # (T,)

    # 5. Lấy trung bình scale qua thời gian (Quan trọng!)
    # Tại sao dùng mean? Để giữ lại chuyển động phóng to/thu nhỏ tự nhiên (nếu có).
    # Nếu scale từng frame, nhân vật sẽ bị "cố định kích thước", mất thông tin chiều sâu.
    scale = dist.mean()
    
    if scale < 1e-6:
        scale = 1  # Tránh chia cho 0 nếu data lỗi

    # 6. Scale toàn bộ dữ liệu (X, Y và cả Z nếu là tọa độ 3D)
    # Nếu kênh 3 là confidence score (giá trị 0-1) thì KHÔNG scale kênh 3.
    # Giả sử kênh 3 là Z (coordinate) -> Cần scale.
    
    # Cách an toàn: Chỉ scale X, Y, Z (nếu Z là toạ độ)
    data[:, :, 0] /= scale # X
    data[:, :, 1] /= scale # Y
    
    # Kiểm tra kênh thứ 3 là gì?
    if C > 2:
        # Nếu giá trị trung bình kênh 3 > 1 (khả năng cao là toạ độ pixel Z), thì scale luôn
        # Nếu nhỏ hơn 1 (khả năng là confidence score), thì kệ nó.
        if np.mean(np.abs(data[:, :, 2])) > 1.0: 
             data[:, :, 2] /= scale

    return data.astype(np.float32)