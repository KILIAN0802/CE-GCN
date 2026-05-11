import cv2
import numpy as np
import mediapipe as mp
import os
import glob
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    MP_SOLUTIONS = mp.solutions
except AttributeError:
    # Some mediapipe builds expose solutions under mediapipe.python.
    from mediapipe.python import solutions as MP_SOLUTIONS


# ==============================
# Mediapipe Modules
# ==============================
mp_hands = MP_SOLUTIONS.hands
mp_pose = MP_SOLUTIONS.pose

# ------------------------------
# CONSTANTS
# ------------------------------
NUM_RIGHT_HAND = 21          # 0–20
NUM_LEFT_HAND = 21           # 21–41
NUM_BODY = 4                 # 42–45
TOTAL_JOINTS = 50            # Full output


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
        # Nose: 0
        # Left shoulder: 11
        # Right shoulder: 12
        # Left elbow: 13
        # Right elbow: 14
        # Left hip: 23
        # Right hip: 24

        # (42) NOSE
        keypoints[42] = [lm[0].x, lm[0].y, lm[0].z]

        # (43) LEFT SHOULDER
        keypoints[43] = [lm[11].x, lm[11].y, lm[11].z]

        # (44) RIGHT SHOULDER
        keypoints[44] = [lm[12].x, lm[12].y, lm[12].z]

        # (45) HIP_CENTER = midpoint(left hip, right hip)
        lx, ly, lz = lm[23].x, lm[23].y, lm[23].z
        rx, ry, rz = lm[24].x, lm[24].y, lm[24].z
        keypoints[45] = [(lx + rx) / 2, (ly + ry) / 2, (lz + rz) / 2]

        # (46) LEFT ELBOW
        keypoints[46] = [lm[13].x, lm[13].y, lm[13].z]

        # (47) RIGHT ELBOW
        keypoints[47] = [lm[14].x, lm[14].y, lm[14].z]

        # (48) LEFT HIP
        keypoints[48] = [lm[23].x, lm[23].y, lm[23].z]

        # (49) RIGHT HIP
        keypoints[49] = [lm[24].x, lm[24].y, lm[24].z]


    # --------------------------------------
    # 2) Hand landmarks (right + left)
    # --------------------------------------
    hands = hands_model.process(rgb_frame)

    if hands.multi_hand_landmarks and hands.multi_handedness:

        # Loop through detected hands
        for hand_landmarks, handedness in zip(hands.multi_hand_landmarks,
                                              hands.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'

            if label == "Left":
                base = 0       # left hand index range: 0–20
            else:
                base = 21      # right hand index range: 21–41

            for i, lm in enumerate(hand_landmarks.landmark):
                keypoints[base + i] = [lm.x, lm.y, lm.z]

    return keypoints



# ==============================
# Extract keypoints from a video
# ==============================
def extract_50_keypoints(video_path, save_path):
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

    frames_kpts = np.array(frames_kpts)     # (T, 50, 3)
    np.save(save_path, frames_kpts)

    print(f"[SAVED] {save_path} | shape={frames_kpts.shape}")
    return True



# ==============================
# Batch processing full folder
# ==============================
def extract_keypoints_folder(video_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    all_video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
    
    # Create a set of existing output file basenames for quick lookup
    try:
        existing_files = {os.path.splitext(f)[0] for f in os.listdir(out_dir)}
    except FileNotFoundError:
        existing_files = set()

    # Filter out videos that have already been processed
    video_files = [v for v in all_video_files if os.path.splitext(os.path.basename(v))[0] not in existing_files]
    
    print(f"Total videos: {len(all_video_files)}")
    print(f"Already processed: {len(existing_files)}")
    print(f"Remaining to process: {len(video_files)}")

    for video_path in tqdm(video_files):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        extract_50_keypoints(video_path, os.path.join(out_dir, video_id))



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
