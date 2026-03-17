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
        raise ImportError(f"Cannot load class '{name}'. Error: {e}")

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A

class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=1, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, edge_importance_weighting=True, **kwargs):
        super().__init__()

        if graph is None:
            raise ValueError("Graph must be specified")

        # Load Graph bằng importlib chuẩn
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)

        # Lấy ma trận kề A
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # Edge Importance Weighting
        spatial_kernel_size = A.size(0)
        self.edge_importance_weighting = edge_importance_weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in range(10)
            ])
        else:
            self.edge_importance = [1] * 10

        # Build Networks
        kernel_size = (9, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 128, kernel_size, 2),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 256, kernel_size, 2),
            st_gcn(256, 256, kernel_size, 1),
            st_gcn(256, 256, kernel_size, 1),
        ))

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # Input shape handling: (N, C, T, V, M) -> (N, M, V, C, T) -> ...
        if len(x.shape) == 4:
            # (N, C, T, V) -> Add Person dim
            x = x.unsqueeze(-1)
            
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            if self.edge_importance_weighting:
                x, _ = gcn(x, self.A * importance)
            else:
                x, _ = gcn(x, self.A)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x