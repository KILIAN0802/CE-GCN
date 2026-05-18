import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib
import os
import sys

# Thêm thư mục cha vào sys.path để có thể import được package models.*
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def import_class(name):
    try:
        module_name, class_name = name.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        klass = getattr(mod, class_name)
        return klass
    except Exception as e:
        raise ImportError(f"Cannot load class '{name}'. Error: {e}")

# ==========================================
# 0. KHỞI TẠO MA TRẬN SH (Structured Hand)
# ==========================================
def get_SH(num_node=50):
    SH = np.zeros((num_node, num_node))
    left_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    right_edges = [(i+21, j+21) for (i, j) in left_edges]
    for i, j in left_edges + right_edges:
        SH[i, j] = 1
        SH[j, i] = 1
    return SH

# ==========================================
# 1. THÀNH PHẦN 1: HAND-AWARE GCN LAYER
# ==========================================
class HandAwareLayer(nn.Module):
    def __init__(self, in_channels, out_channels, A_body, SH_hand):
        super(HandAwareLayer, self).__init__()
        # 1. A: Ma trận kề vật lý cơ thể (Cố định)
        self.register_buffer('A', torch.from_numpy(A_body.astype('float32')))
        
        # 2. B: Ma trận kề tự học toàn thân (Learnable)
        self.B = nn.Parameter(torch.zeros_like(self.A))
        
        # 3. SH: Ma trận kề cấu trúc bàn tay (Cố định - chỉ nối các ngón tay)
        self.register_buffer('SH', torch.from_numpy(SH_hand.astype('float32')))
        
        # 4. PH: Ma trận kề bàn tay tham số hóa (Tự học các kết nối ẩn giữa các ngón)
        self.PH = nn.Parameter(torch.zeros_like(self.SH))
        
        # Trọng số điều tiết (Alpha & Beta)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
        # Phép chiếu đặc trưng (Linear Transformation)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x shape: (N, C, T, V)
        # Tổng hợp đồ thị lai: Cơ thể + Bàn tay
        # PH và SH chỉ có giá trị tại các vị trí của 21 khớp bàn tay (x2)
        H = self.A + self.B + (self.alpha * self.SH) + (self.beta * self.PH)
        
        # Graph Convolution sử dụng Einstein Summation để tính toán hiệu năng cao
        out = torch.einsum('nctv,vw->nctw', (x, H))
        out = self.conv(out)
        return self.bn(out)

# ==========================================
# 2. THÀNH PHẦN 2: SPLIT-HEAD INPUT MODULE
# ==========================================
class SplitHeadInput(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(SplitHeadInput, self).__init__()
        # Mỗi luồng có một phép chiếu tuyến tính riêng để đưa về cùng không gian đặc trưng
        self.joint_head = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1), nn.BatchNorm2d(mid_channels))
        self.bone_head = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1), nn.BatchNorm2d(mid_channels))
        self.vel_head = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1), nn.BatchNorm2d(mid_channels))

    def forward(self, x_j, x_b, x_v):
        # x_j, x_b, x_v: (N, C, T, V)
        f_j = self.joint_head(x_j)
        f_b = self.bone_head(x_b)
        f_v = self.vel_head(x_v)
        
        # Late Fusion bằng phương pháp cộng gộp (Summation)
        return f_j + f_b + f_v

# ==========================================
# 3. THÀNH PHẦN 3: BACKBONE (ST-GCN BLOCK)
# ==========================================
class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A_body, SH_hand, stride=1):
        super(ST_GCN_Block, self).__init__()
        # Spatial GCN (Hand-Aware)
        self.gcn = HandAwareLayer(in_channels, out_channels, A_body, SH_hand)
        
        # Tính năng hỗ trợ downsample (Kế thừa tư tưởng chuẩn bị cho phần Residual)
        if in_channels != out_channels or stride != 1:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
            
        # Temporal GCN (Multi-scale)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_channels),
        )
        
        # Cơ chế chú ý (Spatial-Temporal Attention)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = self.down(x)
        x = self.gcn(x)
        x = self.tcn(x)
        
        # Áp dụng Attention
        attn = self.attention(x)
        x = x * attn
        
        # Residual Connection linh hoạt hơn một chút để xử lý được cả downsample
        if identity.shape == x.shape:
            x += identity
        return F.relu(x, inplace=True)

# ==========================================
# 4. MAIN MODEL (HybridGCN)
# ==========================================
class HybridGCN(nn.Module):
    def __init__(self, num_class=200, num_point=50, num_person=1, graph='models.graph.vsl_graph.Graph', graph_args=dict(), drop_out=0.5, base_channel=64):
        super(HybridGCN, self).__init__()

        # Khởi tạo Graph (A_body)
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = self.graph.A 
        if A.ndim == 3:
            A = A.sum(0) # Đưa về (V, V)
            
        self.num_point = num_point
        self.num_person = num_person
        
        # Khởi tạo ma trận tay SH_hand
        SH = get_SH(num_node=num_point)
        
        # BatchNorm ban đầu cho mỗi luồng (Joint, Bone, Vel) - mỗi luồng 3 kênh
        self.data_bn_j = nn.BatchNorm1d(num_person * 3 * num_point)
        self.data_bn_v = nn.BatchNorm1d(num_person * 3 * num_point)
        self.data_bn_b = nn.BatchNorm1d(num_person * 3 * num_point)
        
        # 1. Split-Head Input (Nhận 3 kênh, trả về base_channel)
        self.split_head = SplitHeadInput(3, base_channel)

        # 2. Backbone 10 khối ST-GCN Block
        self.l1 = ST_GCN_Block(base_channel, base_channel, A, SH, stride=1)
        self.l2 = ST_GCN_Block(base_channel, base_channel, A, SH, stride=1)
        self.l3 = ST_GCN_Block(base_channel, base_channel, A, SH, stride=1)
        self.l4 = ST_GCN_Block(base_channel, base_channel*2, A, SH, stride=2)
        self.l5 = ST_GCN_Block(base_channel*2, base_channel*2, A, SH, stride=1)
        self.l6 = ST_GCN_Block(base_channel*2, base_channel*2, A, SH, stride=1)
        self.l7 = ST_GCN_Block(base_channel*2, base_channel*4, A, SH, stride=2)
        self.l8 = ST_GCN_Block(base_channel*4, base_channel*4, A, SH, stride=1)
        self.l9 = ST_GCN_Block(base_channel*4, base_channel*4, A, SH, stride=1)
        self.l10 = ST_GCN_Block(base_channel*4, base_channel*4, A, SH, stride=1)

        # 3. Classifier
        self.fc = nn.Linear(base_channel*4, num_class)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        # Format input (N, C, T, V, M)
        if len(x.shape) == 4:
            x = x.unsqueeze(-1)
            
        N, C, T, V, M = x.size()
        
        # Bóc tách 3 luồng từ tensor fused 9 channels:
        # Joint (0-2), Velocity (3-5), Bone (6-8)
        x_j = x[:, 0:3, :, :, :]
        x_v = x[:, 3:6, :, :, :]
        x_b = x[:, 6:9, :, :, :]
        
        def normalize_stream(stream, bn):
            # stream: (N, 3, T, V, M) -> (N, M*V*3, T) -> bn -> (N*M, 3, T, V)
            stream = stream.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * 3, T)
            stream = bn(stream)
            stream = stream.view(N, M, V, 3, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, 3, T, V)
            return stream
            
        x_j = normalize_stream(x_j, self.data_bn_j)
        x_v = normalize_stream(x_v, self.data_bn_v)
        x_b = normalize_stream(x_b, self.data_bn_b)
        
        # Split-Head Module: Projection + Summation
        x = self.split_head(x_j, x_b, x_v)
        
        # Backbone 10 blocks
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        
        # Global Average Pooling
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1) # (N, c_new)
        x = self.drop_out(x)
        
        # Classifier
        out = self.fc(x)
        return out

if __name__ == '__main__':
    N, C, T, V, M = 2, 9, 64, 50, 1
    num_class = 200
    
    dummy_input = torch.randn(N, C, T, V, M)
    print(f"Input shape: {dummy_input.shape}")
    
    model = HybridGCN(num_class=num_class, num_point=V, num_person=M, graph='models.graph.vsl_graph.Graph')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f} M")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (N, num_class), f"Lỗi shape! Mong đợi {(N, num_class)} nhưng nhận được {output.shape}"
    print("Kiểm tra kiến trúc thành công!")
