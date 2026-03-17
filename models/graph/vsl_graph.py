import numpy as np

# =========================================================
# HÀM HỖ TRỢ TÍNH TOÁN MA TRẬN KỀ (SPATIAL STRATEGY)
# =========================================================

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # Tính khoảng cách hop
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, self_link, inward, outward):
    I = np.array(self_link) # Self loop
    In = np.array(inward)   # Hướng vào
    Out = np.array(outward) # Hướng ra
    neighbor_link = [(i, j) for (i, j) in inward + outward]
    
    # Tạo ma trận A ban đầu
    A = np.zeros((num_node, num_node))
    for i, j in neighbor_link:
        A[j, i] = 1
        A[i, j] = 1 # Vô hướng để tính khoảng cách

    # 1. TÍNH KHOẢNG CÁCH ĐẾN TÂM (ROOT)
    # Chọn Node 45 (Hip Center) làm tâm của cơ thể
    center = 45 
    
    hop_dis = get_hop_distance(num_node, neighbor_link, max_hop=num_node)
    
    # Khoảng cách từ mỗi node đến tâm
    dist_center = hop_dis[center, :]

    # 2. CHIA THÀNH 3 NHÓM (SPATIAL CONFIGURATION)
    # Nhóm 0: Chính nó (Uniform)
    A0 = np.zeros((num_node, num_node))
    for i, j in self_link:
        A0[i, j] = 1
    A0 = normalize_digraph(A0)

    # Nhóm 1: Closer (Node j gần tâm hơn Node i) -> Centripetal
    A1 = np.zeros((num_node, num_node))
    for i, j in neighbor_link:
        if dist_center[i] < dist_center[j]: # i gần tâm hơn j -> cạnh j nối vào i là hướng tâm
             A1[i, j] = 1
    A1 = normalize_digraph(A1)

    # Nhóm 2: Further (Node j xa tâm hơn Node i) -> Centrifugal
    A2 = np.zeros((num_node, num_node))
    for i, j in neighbor_link:
        if dist_center[i] > dist_center[j]:
            A2[i, j] = 1
    A2 = normalize_digraph(A2)

    # Gộp lại thành (3, V, V)
    A = np.stack((A0, A1, A2))
    return A


# =========================================================
# CLASS GRAPH CHÍNH
# =========================================================

class Graph:
    """ Graph layout for 46 Mediapipe keypoints """

    def __init__(self, layout='vsl_layout', strategy='spatial', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node = 46
        
        self.layout = layout
        self.strategy = strategy

        # Định nghĩa cạnh
        self.inward = self._get_inward_edges()
        self.outward = [(j, i) for (i, j) in self.inward]
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.neighbor = self.inward + self.outward

        # Tạo ma trận kề
        self.A = self.get_adjacency_matrix()

    def _get_inward_edges(self):
        inward = []
        # ---- LEFT HAND (0-20) ----
        left_offset = 0
        left_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]
        inward += [(i + left_offset, j + left_offset) for (i, j) in left_edges]

        # ---- RIGHT HAND (21-41) ----
        right_offset = 21
        inward += [(i + right_offset, j + right_offset) for (i, j) in left_edges]

        # ---- BODY (42-45) ----
        # 42:Nose, 43:L-Sho, 44:R-Sho, 45:Hip-Cen
        body_edges = [
            (42, 43), (42, 44),
            (43, 45), (44, 45),
        ]
        inward += body_edges

        # ---- CONNECTIONS (Body -> Hands) ----
        # L-Shoulder(43) -> L-Wrist(0)
        # R-Shoulder(44) -> R-Wrist(21)
        inward += [(43, 0), (44, 21)] 

        return inward

    def get_adjacency_matrix(self):
        if self.strategy == 'spatial':
            # Sử dụng hàm Spatial đã viết ở trên
            A = get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            # Fallback về Uniform (1, V, V) nếu không dùng Spatial
            # Nhưng model CTRGCN của bạn đang mặc định spatial (3 channels)
            raise ValueError("Hiện tại chỉ hỗ trợ strategy='spatial'")
            
        return A