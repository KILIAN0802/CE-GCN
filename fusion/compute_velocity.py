import numpy as np

def compute_velocity(joints):
    """
    joints: (3, T, V)
    Output: (3, T, V)
    """

    C, T, V = joints.shape
    velocity = np.zeros_like(joints)

    # Compute difference frame-to-frame (X_{t+1} - X_{t})
    velocity[:, :-1, :] = joints[:, 1:, :] - joints[:, :-1, :]
    velocity[:, -1, :] = 0 # Frame cuối cùng không có t+1 nên gán bằng 0

    return velocity.astype(np.float32)
