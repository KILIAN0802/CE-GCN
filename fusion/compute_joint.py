import numpy as np

def compute_joint(kpts, max_frames=64):
    """
    kpts: numpy (T, V, 3)
    Output: (3, max_frames, V)
    """

    T, V, C = kpts.shape
    assert C == 3, "Keypoints must have 3 channels (x,y,z)"

    # Pad hoặc crop về đúng max_frames
    if T < max_frames:
        pad = np.zeros((max_frames - T, V, 3))
        kpts = np.concatenate([kpts, pad], axis=0)
    else:
        kpts = kpts[:max_frames]

    # (T, V, 3) → (C=3, T, V)
    joints = kpts.transpose(2, 0, 1)
    return joints.astype(np.float32)
