import numpy as np

# Định nghĩa các kết nối xương (Bone pairs) dựa trên cấu trúc 50 khớp
# Dựa trên file vsl_graph.py và compute_bone.py trong dự án
RIGHT_HAND = [
    (0,1),(1,2),(2,3),(3,4),        
    (0,5),(5,6),(6,7),(7,8),        
    (0,9),(9,10),(10,11),(11,12),   
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# Tay trái là đối xứng của tay phải với offset +21
LEFT_HAND = [(a+21, b+21) for (a,b) in RIGHT_HAND]

# Phần thân và hông
BODY = [
    (42, 43),   # Nose -> L-shoulder
    (42, 44),   # Nose -> R-shoulder
    (43, 45),   # L-shoulder -> hip_center
    (44, 45),   # R-shoulder -> hip_center
    (45, 48),   # hip_center -> L-hip
    (45, 49)    # hip_center -> R-hip
]

# Cánh tay kết nối vào bàn tay
ARMS = [
    (43, 46), (46, 0),   # Left arm (L-Sho -> L-Elbow -> L-Wrist)
    (44, 47), (47, 21)   # Right arm (R-Sho -> R-Elbow -> R-Wrist)
]

BONE_PAIRS = RIGHT_HAND + LEFT_HAND + BODY + ARMS

def preprocess_skeleton(data):
    """
    Thực hiện tiền xử lý cho dữ liệu skeleton 50 khớp.
    
    Args:
        data: numpy array shape (N, T, V, C) 
              Trong đó: N = số mẫu, T = số khung hình, V = 50 (khớp), C = 3 (x,y,z)
              
    Returns:
        Một dictionary chứa 3 luồng: 'joint', 'bone', 'velocity'.
        Mỗi luồng có shape định dạng cho GCN: (N, 3, T, 50, 1)
    """
    N, T, V, C = data.shape
    if V != 50 or C != 3:
        raise ValueError(f"Dữ liệu đầu vào phải có shape (N, T, 50, 3). Nhận được: {data.shape}")
        
    # -------------------------------------------------------------
    # 1. Hip-centric Normalization
    # -------------------------------------------------------------
    # Xác định 2 điểm hông (indices 48 và 49)
    # Tính trung điểm hông (Mid-hip) cho mỗi frame: shape (N, T, 1, 3)
    mid_hip = (data[:, :, 48, :] + data[:, :, 49, :]) / 2.0
    mid_hip = np.expand_dims(mid_hip, axis=2)
    
    # Lấy Mid-hip làm gốc tọa độ (0,0,0) bằng cách trừ đi mid_hip
    joint_stream = data - mid_hip
    
    # -------------------------------------------------------------
    # 2. Multi-stream Generation
    # -------------------------------------------------------------
    # A. Bone stream (Vector hiệu giữa các khớp kết nối)
    # Khởi tạo ma trận zero cho bone_stream
    bone_stream = np.zeros_like(joint_stream)
    for src, dst in BONE_PAIRS:
        # Vector xương chỉ từ src đến dst: X_dst - X_src
        if src < V and dst < V:
            bone_stream[:, :, dst, :] = joint_stream[:, :, dst, :] - joint_stream[:, :, src, :]
            
    # B. Velocity stream (Động lực học chuyển động)
    # X_{t+1, v} - X_{t, v}
    velocity_stream = np.zeros_like(joint_stream)
    # Tính hiệu số giữa 2 frame liên tiếp
    velocity_stream[:, :-1, :, :] = joint_stream[:, 1:, :, :] - joint_stream[:, :-1, :, :]
    # Frame cuối không có frame t+1 nên gán bằng 0 (hoặc có thể giữ giá trị của frame t-1 tùy thuộc vào thiết kế cụ thể)
    velocity_stream[:, -1, :, :] = 0 
    
    # -------------------------------------------------------------
    # 3. Data Formatting
    # -------------------------------------------------------------
    # Định dạng đầu vào: (N, T, V, C)
    # Định dạng GCN yêu cầu: (N, C, T, V, M) trong đó M=1 (số người = 1)
    
    def format_for_gcn(stream):
        # Bước 1: Đổi trục (N, T, V, C) -> (N, C, T, V)
        # Bằng cách chuyển vị: trục 0 giữ nguyên, trục 3 lên vị trí 1, trục 1 sang vị trí 2, trục 2 sang vị trí 3
        stream_transposed = np.transpose(stream, (0, 3, 1, 2))
        
        # Bước 2: Thêm chiều M=1 ở cuối cùng -> (N, C, T, V, 1)
        stream_formatted = np.expand_dims(stream_transposed, axis=-1)
        
        return stream_formatted.astype(np.float32)

    return {
        'joint': format_for_gcn(joint_stream),
        'bone': format_for_gcn(bone_stream),
        'velocity': format_for_gcn(velocity_stream)
    }

if __name__ == '__main__':
    # Test thử module với dữ liệu giả lập
    N, T, V, C = 2, 64, 50, 3
    dummy_data = np.random.rand(N, T, V, C)
    
    print(f"Input shape: {dummy_data.shape}")
    
    streams = preprocess_skeleton(dummy_data)
    
    print("\nOutput shapes:")
    print(f"Joint stream: {streams['joint'].shape}")
    print(f"Bone stream: {streams['bone'].shape}")
    print(f"Velocity stream: {streams['velocity'].shape}")
    
    print("\nKiểm tra Hip-centric Normalization:")
    # Kiểm tra xem Mid-hip của joint_stream có xấp xỉ 0 hay không
    mid_hip_check = (streams['joint'][0, :, 0, 48, 0] + streams['joint'][0, :, 0, 49, 0]) / 2
    print(f"Giá trị (L_Hip + R_Hip)/2 ở mẫu 0, frame 0, kênh X: {mid_hip_check}")
