import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib

def import_class(name):
    try:
        module_name, class_name = name.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        klass = getattr(mod, class_name)
        return klass
    except Exception as e:
        raise ImportError(f"Không load được class '{name}'. Lỗi: {e}")

# ==========================================
# 1. CORE GRAPH CONVOLUTION (CTR-GC)
# ==========================================
class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
            
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out')

    def forward(self, x, A=None, alpha=1):
        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

# ==========================================
# 2. BASIC BLOCK (TCN + GCN)
# ==========================================
class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.adaptive = adaptive
        self.conv1 = CTRGC(in_channels, out_channels, rel_reduction=8)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        self.alpha = nn.Parameter(torch.zeros(1))
        
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.normal_(self.PA, 0, 0.02)

    def forward(self, x):
        A = self.PA
        out = self.conv1(x, A, self.alpha)
        out = self.bn1(out)
        out = out + self.down(x)
        return self.relu(out)

class TCN_GCN_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_Unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)

# ==========================================
# 3. MAIN MODEL (CTRGCN ORIGINAL)
# ==========================================
class CTRGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, drop_out=0.5, base_channel=64):
        super(CTRGCN, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A 
        if A.ndim == 3:
            A = A.sum(0)
            
        self.num_class = num_class
        self.num_point = num_point
        
        # [QUAN TRỌNG] Input layer nhận đúng in_channels (ví dụ: 9)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l1 = TCN_GCN_Unit(in_channels, base_channel, A, residual=False)
        
        # Backbone layers
        self.l2 = TCN_GCN_Unit(base_channel, base_channel, A)
        self.l3 = TCN_GCN_Unit(base_channel, base_channel, A)
        self.l4 = TCN_GCN_Unit(base_channel, base_channel*2, A, stride=2)
        self.l5 = TCN_GCN_Unit(base_channel*2, base_channel*2, A)
        self.l6 = TCN_GCN_Unit(base_channel*2, base_channel*2, A)
        self.l7 = TCN_GCN_Unit(base_channel*2, base_channel*4, A, stride=2)
        self.l8 = TCN_GCN_Unit(base_channel*4, base_channel*4, A)
        self.l9 = TCN_GCN_Unit(base_channel*4, base_channel*4, A)
        self.l10 = TCN_GCN_Unit(base_channel*4, base_channel*4, A)

        self.fc = nn.Linear(base_channel*4, num_class)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        # x shape: (N, C, T, V, M)
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        
        if len(x.shape) == 4:
            x = x.unsqueeze(-1)
            
        N, C, T, V, M = x.size()

        # Normalization
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # Backbone (Chạy xuyên suốt, không cắt luồng)
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

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)