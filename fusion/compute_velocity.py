import numpy as np

def compute_velocity(joints):
    """
    joints: (3, T, V)
    Output: (3, T, V)
    """

    C, T, V = joints.shape
    velocity = np.zeros_like(joints)

    # Compute difference frame-to-frame
    velocity[:, 1:, :] = joints[:, 1:, :] - joints[:, :-1, :]

    return velocity.astype(np.float32)
