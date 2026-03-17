import numpy as np
from fusion.compute_joint import compute_joint
from fusion.compute_velocity import compute_velocity
from fusion.compute_bone import compute_bone


def early_fusion(kpts, max_frames=64):
    """
    kpts: numpy (T, V, 3)
    Output: fused = (9, max_frames, V)
    """

    # Step 1: Joint Stream (3, 64, 46)
    joints = compute_joint(kpts, max_frames=max_frames)

    # Step 2: Velocity Stream (3, 64, 46)
    velocity = compute_velocity(joints)

    # Step 3: Bone Stream (3, 64, 46)
    bone = compute_bone(joints)

    # Step 4: Concatenate → (9, T, V)
    fused = np.concatenate([joints, velocity, bone], axis=0)

    return fused.astype(np.float32)
