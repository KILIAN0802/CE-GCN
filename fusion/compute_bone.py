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

# ----- Body joints -----
BODY = [
    (42, 44),   # L-shoulder → neck
    (43, 44),   # R-shoulder → neck
    (44, 45)    # neck → hip_center
]

# FULL BONE GRAPH
BONE_PAIRS = RIGHT_HAND + LEFT_HAND + BODY



def compute_bone(joints, bone_pairs=BONE_PAIRS):
    """
    joints: (3, T, V=46)
    Output: bone (3, T, V)
    """
    C, T, V = joints.shape
    bone = np.zeros_like(joints)

    for src, dst in bone_pairs:
        if src < V and dst < V:
            bone[:, :, src] = joints[:, :, src] - joints[:, :, dst]

    return bone.astype(np.float32)
