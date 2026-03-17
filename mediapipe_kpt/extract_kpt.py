import cv2
import numpy as np
import mediapipe as mp
import os
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ==============================
# Mediapipe Modules
# ==============================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# ------------------------------
# CONSTANTS
# ------------------------------
NUM_RIGHT_HAND = 21          # 0–20
NUM_LEFT_HAND = 21           # 21–41
NUM_BODY = 4                 # 42–45
TOTAL_JOINTS = 46            # Full output


# ==============================
# Extract one frame
# ==============================
def extract_from_frame(rgb_frame, hands_model, pose_model):
    """
    Return: (46, 3)
    If missing → pad = 0
    """

    H, W, _ = rgb_frame.shape

    keypoints = np.zeros((TOTAL_JOINTS, 3), dtype=np.float32)

    # --------------------------------------
    # 1) Pose landmarks (body: shoulders, neck, hip)
    # --------------------------------------
    pose = pose_model.process(rgb_frame)

    if pose.pose_landmarks:
        lm = pose.pose_landmarks.landmark

        # Body indexes in Mediapipe pose:
        # Left shoulder: 11
        # Right shoulder: 12
        # Nose: 0   (used as neck approximate)
        # Left hip: 23
        # Right hip: 24

        # (42) LEFT SHOULDER
        keypoints[42] = [lm[11].x, lm[11].y, lm[11].z]

        # (43) RIGHT SHOULDER
        keypoints[43] = [lm[12].x, lm[12].y, lm[12].z]

        # (44) NECK ≈ midpoint(nose, shoulders)
        nx, ny, nz = lm[0].x, lm[0].y, lm[0].z
        keypoints[44] = [nx, ny, nz]

        # (45) HIP_CENTER = midpoint(left hip, right hip)
        lx, ly, lz = lm[23].x, lm[23].y, lm[23].z
        rx, ry, rz = lm[24].x, lm[24].y, lm[24].z
        keypoints[45] = [(lx + rx) / 2, (ly + ry) / 2, (lz + rz) / 2]


    # --------------------------------------
    # 2) Hand landmarks (right + left)
    # --------------------------------------
    hands = hands_model.process(rgb_frame)

    if hands.multi_hand_landmarks and hands.multi_handedness:

        # Loop through detected hands
        for hand_landmarks, handedness in zip(hands.multi_hand_landmarks,
                                              hands.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'

            if label == "Right":
                base = 0       # right hand index range: 0–20
            else:
                base = 21      # left hand index range: 21–41

            for i, lm in enumerate(hand_landmarks.landmark):
                keypoints[base + i] = [lm.x, lm.y, lm.z]

    return keypoints



# ==============================
# Extract keypoints from a video
# ==============================
def extract_46_keypoints(video_path, save_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return False

    frames_kpts = []

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as hands, mp_pose.Pose(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            kpts = extract_from_frame(frame_rgb, hands, pose)
            frames_kpts.append(kpts)

    cap.release()

    frames_kpts = np.array(frames_kpts)     # (T, 46, 3)
    np.save(save_path, frames_kpts)

    print(f"[SAVED] {save_path} | shape={frames_kpts.shape}")
    return True



# ==============================
# Batch processing full folder
# ==============================
def extract_keypoints_folder(video_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    videos = [v for v in os.listdir(video_dir)
              if v.endswith(".mp4") or v.endswith(".avi")]

    for v in tqdm(videos, desc="Extracting keypoints"):
        vid_path = os.path.join(video_dir, v)
        save_path = os.path.join(out_dir, v.replace(".mp4", ".npy"))

        extract_46_keypoints(vid_path, save_path)



# ==============================
# Main (run from terminal)
# ==============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Folder chứa video thô")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Folder lưu keypoints .npy")

    args = parser.parse_args()

    extract_keypoints_folder(args.video_dir, args.out_dir)
