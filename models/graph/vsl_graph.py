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
    center = 25 
    
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
    """ 
    Graph layout được tùy chỉnh để chỉ sử dụng 26 điểm từ bộ 46 điểm Mediapipe gốc.
    Dùng cho mô hình CTR-GCN với input shape (Batch, C, T, 26).
    """

    def __init__(self, layout='vsl_layout', strategy='spatial', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        
        # 🚨 QUAN TRỌNG: Model bây giờ chỉ nhìn thấy 26 node
        self.num_node = 26
        
        self.layout = layout
        self.strategy = strategy

        # --- DANH SÁCH 26 ĐIỂM BẠN ĐÃ CHỌN (TỪ 46 ĐIỂM GỐC) ---
        self.selected_indices = [
            0, 2, 4, 5, 8, 9, 12, 13, 16, 17, 20,       # Tay phải (11 điểm)
            21, 23, 25, 26, 29, 30, 33, 34, 37, 38, 41, # Tay trái (11 điểm)
            42, 43, 44, 45                             # Cơ thể (4 điểm)
        ]
        
        # Tạo bảng tra cứu: Chuyển ID 46 gốc -> ID 26 mới (0 đến 25)
        # Ví dụ: Gốc là 45 (Hip Center) -> Mới sẽ là index 25
        self.mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.selected_indices)}

        # Định nghĩa cạnh (Dùng ID gốc để định nghĩa, sau đó ánh xạ sang ID mới)
        self.inward = self._get_inward_edges()
        self.outward = [(j, i) for (i, j) in self.inward]
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.neighbor = self.inward + self.outward

        # Tạo ma trận kề (Kích thước sẽ là 3, 26, 26)
        self.A = self.get_adjacency_matrix()

    def _get_inward_edges(self):
        """ 
        Định nghĩa các xương nối dựa trên các index bạn đã giữ lại.
        Tôi nối trực tiếp các điểm chính vì bạn đã bỏ qua các điểm trung gian.
        """
        # 1. Định nghĩa các cạnh theo ID 46 điểm gốc
        old_edges = [
            # --- Tay phải (Nối các điểm bạn giữ lại) ---
            (0, 2), (2, 4),   # Ngón cái
            (0, 5), (5, 8),   # Ngón trỏ
            (0, 9), (9, 12),  # Ngón giữa
            (0, 13), (13, 16),# Ngón áp út
            (0, 17), (17, 20),# Ngón út

            # --- Tay trái (Nối các điểm bạn giữ lại) ---
            (21, 23), (23, 25), # Ngón cái
            (21, 26), (26, 29), # Ngón trỏ
            (21, 30), (30, 33), # Ngón giữa
            (21, 34), (34, 37), # Ngón áp út
            (21, 38), (38, 41), # Ngón út

            # --- Thân (Nose, Shoulder, Hip) ---
            (42, 43), (42, 44), # Mũi nối 2 vai
            (43, 45), (44, 45), # Vai nối Hông

            # --- Kết nối Thân -> Tay ---
            (43, 0),  # Vai trái -> Cổ tay trái (0)
            (44, 21)  # Vai phải -> Cổ tay phải (21)
        ]

        # 2. Ánh xạ (Map) các cạnh này sang hệ thống 26 điểm mới
        inward_26 = []
        for (i, j) in old_edges:
            if i in self.mapping and j in self.mapping:
                inward_26.append((self.mapping[i], self.mapping[j]))
        
        return inward_26

    def get_adjacency_matrix(self):
        if self.strategy == 'spatial':
            # Hàm get_spatial_graph sẽ nhận num_node=26 
            # Ma trận A trả về sẽ có shape (3, 26, 26)
            A = get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError("Mô hình CTR-GCN yêu cầu strategy='spatial'")
            
        return A