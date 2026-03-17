import os
import cv2
import argparse
import pandas as pd

from ultralytics import YOLO
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_video_list(csv_paths):
    """Load tất cả video từ train/val/test CSV."""
    video_list = []
    for path in csv_paths:
        df = pd.read_csv(path)
        video_list += df["file_name"].tolist()
    return sorted(list(set(video_list)))  # unique + sorted


def auto_center_and_scale(xyxy, img_w, img_h, target_ratio=0.80):
    """
    Tăng chiều ngang 8 cm (~15% mỗi bên)
    Tăng chiều trên 3 cm (~10% chiều cao)
    """

    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1

    # Scale để chiều cao ~80% frame
    desired_h = img_h * target_ratio
    scale = desired_h / h

    new_w = w * scale
    new_h = h * scale

    # Center
    cx = x1 + w / 2
    cy = y1 + h / 2

    # Base expanded bbox
    nx1 = cx - new_w / 2
    ny1 = cy - new_h / 2
    nx2 = cx + new_w / 2
    ny2 = cy + new_h / 2

    # ===============================
    # CUSTOM EXPANSION
    # ===============================

    # 1) Expand left + right = thêm 8cm ≈ 30% chiều rộng
    horizontal_expand = 0.60 * (nx2 - nx1)
    nx1 -= horizontal_expand / 2
    nx2 += horizontal_expand / 2

    # 2) Expand lên trên = thêm 3cm ≈ 10% chiều cao
    vertical_expand_top = 0.20 * (ny2 - ny1)
    ny1 -= vertical_expand_top

    # Clamp vào frame
    nx1 = max(0, int(nx1))
    ny1 = max(0, int(ny1))
    nx2 = min(img_w - 1, int(nx2))
    ny2 = min(img_h - 1, int(ny2))

    return nx1, ny1, nx2, ny2



def crop_video(model, input_path, output_path):
    """Crop 1 video bằng phương pháp upper-body autoscale."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (256, 256)
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, verbose=False)

        person_box = None
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0].cpu())
                if cls == 0:  # PERSON
                    person_box = box.xyxy[0].cpu().numpy().tolist()
                    break

        if person_box is None:
            crop = cv2.resize(frame, (256, 256))
            out.write(crop)
            continue

        x1, y1, x2, y2 = auto_center_and_scale(person_box, W, H)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame

        crop = cv2.resize(crop, (256, 256))
        out.write(crop)

    cap.release()
    out.release()
    return True


def main(model_path, raw_dir, out_dir, train_csv, val_csv, test_csv):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading YOLO model...")
    model = YOLO(model_path)

    print("Loading dataset split CSVs...")
    video_list = load_video_list([train_csv, val_csv, test_csv])
    print(f"Total videos to crop: {len(video_list)}")

    for v in tqdm(video_list, desc="Cropping VSL200 videos"):
        input_path = os.path.join(raw_dir, v)
        output_path = os.path.join(out_dir, v)

        if not os.path.exists(input_path):
            print(f"[WARNING] Missing raw video: {input_path}")
            continue

        crop_video(model, input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)

    args = parser.parse_args()

    main(
        args.model_path,
        args.raw_dir,
        args.out_dir,
        args.train_csv,
        args.val_csv,
        args.test_csv,
    )
