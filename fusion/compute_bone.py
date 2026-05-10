import numpy as np

# ----- 21 joints hand definitions -----
RIGHT_HAND = [
    (0,1),(1,2),(2,3),(3,4),        
    (0,5),(5,6),(6,7),(7,8),        
    (0,9),(9,10),(10,11),(11,12),   
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

LEFT_HAND = [(a+21, b+21) for (a,b) in RIGHT_HAND]

# ----- Body joints (42-45) + NEW HIPS (48-49) -----
BODY = [
    (42, 43),   # Nose → L-shoulder
    (42, 44),   # Nose → R-shoulder
    (43, 45),   # L-shoulder → hip_center
    (44, 45),   # R-shoulder → hip_center
    (45, 48),   # hip_center → L-hip
    (45, 49)    # hip_center → R-hip
]

# ----- Arms (Shoulder → Elbow → Wrist) -----
# 46: L-Elbow, 47: R-Elbow
# 0: L-Wrist, 21: R-Wrist
ARMS = [
    (43, 46), (46, 0),  # Left arm
    (44, 47), (47, 21)  # Right arm
]

# FULL BONE GRAPH
BONE_PAIRS = RIGHT_HAND + LEFT_HAND + BODY + ARMS



def compute_bone(joints, bone_pairs=BONE_PAIRS):
    """
    joints: (3, T, V=50)
    Output: bone (3, T, V)
    """
    C, T, V = joints.shape
    bone = np.zeros_like(joints)

    for src, dst in bone_pairs:
        if src < V and dst < V:
            bone[:, :, src] = joints[:, :, src] - joints[:, :, dst]

    return bone.astype(np.float32)
