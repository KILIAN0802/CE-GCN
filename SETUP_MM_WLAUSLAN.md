# 🎯 Hướng dẫn Train CE-GCN trên MM-WLAuslan Dataset

## 📋 Tóm tắt thay đổi cần làm

Bạn cần thực hiện **4 bước** để train model trên MM-WLAuslan:

---

## **BƯỚC 1: Tạo CSV Label Files** ✅

Script tự động đã được tạo: `prepare_mm_wlauslan_labels.py`

**Chạy lệnh:**
```bash
python prepare_mm_wlauslan_labels.py \
    --video_dir /mnt/sda1/VSLR_Storage/MM-WLAuslan/videos \
    --output_dir ./data1 \
    --train_splits train \
    --val_splits valid \
    --test_splits testTW,testSTU,testSYN,testTED
```

**Nếu muốn train riêng trên STU:**
```bash
python prepare_mm_wlauslan_labels.py \
    --video_dir /mnt/sda1/VSLR_Storage/MM-WLAuslan/videos \
    --output_dir ./data1 \
    --train_splits testSTU \
    --val_splits valid \
    --test_splits testTW,testSYN,testTED
```

**Kết quả:**
- `./data1/train_fused_features/train_label.csv` (chứa danh sách videos + class IDs)
- `./data1/val_fused_features/val_label.csv`
- `./data1/test_fused_features/test_label.csv`

---

## **BƯỚC 2: Crop Videos bằng YOLO** 

Run script: `detection_yolo/yolo_crop_VSL200.py`

```bash
python detection_yolo/yolo_crop_VSL200.py \
    --model_path detection_yolo/weights/yolov8n-pose.pt \
    --raw_dir /mnt/sda1/VSLR_Storage/MM-WLAuslan/videos \
    --out_dir ./data1/cropped_videos \
    --train_csv ./data1/train_fused_features/train_label.csv \
    --val_csv ./data1/val_fused_features/val_label.csv \
    --test_csv ./data1/test_fused_features/test_label.csv \
    --exclude_dirs cropping,cropped_videos,data1
```

**Output:** `./data1/cropped_videos/` (cropped & resized 256x256 videos)

---

## **BƯỚC 3: Extract Keypoints bằng MediaPipe**

Run script: `mediapipe_kpt/extract_kpt.py` (hoặc script tương tự)

```bash
python mediapipe_kpt/extract_kpt.py \
    --video_dir ./data1/cropped_videos \
    --out_dir ./data1/keypoints
```

**Output:** `./data1/keypoints/` (files .npy với shape `(T, 46, 3)`)

---

## **BƯỚC 4: Generate 9-Channel Fused Features**

Run script: `fusion/generate_fused_features.py`

(Cần config đường dẫn input/output)

**Output:**
```
./data1/
├── train_fused_features/
│   ├── train_label.csv      ✅ (đã tạo ở Bước 1)
│   └── *.npy files          ← Được tạo ở bước này
├── val_fused_features/
│   ├── val_label.csv        ✅ (đã tạo ở Bước 1)
│   └── *.npy files
└── test_fused_features/
    ├── test_label.csv       ✅ (đã tạo ở Bước 1)
    └── *.npy files
```

---

## **BƯỚC 5: Train Model**

Dùng config file mới: `configs/CTRGCN_mm_wlauslan.yaml`

```bash
python trainer/trainer.py \
    --config configs/CTRGCN_mm_wlauslan.yaml
```

**Training output sẽ lưu vào:** `./results/mm_wlauslan/ctrgcn_scratch/`

---

## 📝 Danh sách Files đã tạo/sửa

✅ **Tạo mới:**
- `prepare_mm_wlauslan_labels.py` - Script tự động tạo CSV từ folder structure
- `configs/CTRGCN_mm_wlauslan.yaml` - Config file cho MM-WLAuslan

⚠️ **Cần kiểm tra / có thể cần sửa:**
- `fusion/generate_fused_features.py` - Cần config đường dẫn input/output
- `mediapipe_kpt/extract_kpt.py` - Cần kiểm tra tên script chính xác
- `app.py` - Sửa `weights_path` sau khi training xong

---

## 🔍 Chi tiết từng thay đổi

### 1. **prepare_mm_wlauslan_labels.py**
- Scan folders: `train/`, `valid/`, `testTW/`, `testSTU/`
- Có thể chọn split linh hoạt bằng `--train_splits`, `--val_splits`, `--test_splits`
- Extract class ID từ tên file (vd: `00000_kf_rgb.mp4` → class 0)
- Tạo CSV files với cấu trúc: `file_name,label_id`

### 2. **CTRGCN_mm_wlauslan.yaml**
```yaml
work_dir: ./results/mm_wlauslan/ctrgcn_scratch
model_args:
  num_class: 200                    # MM-WLAuslan có 200 classes
  num_point: 46                     # 46 keypoints (tương tự VSL)
  in_channels: 9                    # 9-channel fusion (Joint + Velocity + Bone)
  
train_feeder_args:
  csv_path: ./data1/train_fused_features/train_label.csv
  feature_dir: ./data1/train_fused_features
  ...
```

### 3. **app.py** (sửa sau training)
```python
# Sau khi training xong, update path này:
weights_path = "./results/mm_wlauslan/ctrgcn_scratch/best_model.pth"
```

---

## ⚡ Tóm tắt lệnh cần chạy

```bash
# 1. Tạo CSV files
python prepare_mm_wlauslan_labels.py \
    --video_dir /mnt/sda1/VSLR_Storage/MM-WLAuslan/videos \
    --output_dir ./data1

# 2. Crop videos (YOLO)
python detection_yolo/yolo_crop_VSL200.py \
    --model_path detection_yolo/weights/yolov8n-pose.pt \
    --raw_dir /mnt/sda1/VSLR_Storage/MM-WLAuslan/videos \
    --out_dir ./data1/cropped_videos \
    --train_csv ./data1/train_fused_features/train_label.csv \
    --val_csv ./data1/val_fused_features/val_label.csv \
    --test_csv ./data1/test_fused_features/test_label.csv

# 3. Extract keypoints (MediaPipe)
python mediapipe_kpt/extract_kpt.py \
    --video_dir ./data1/cropped_videos \
    --out_dir ./data1/keypoints

# 4. Generate fused features (Fusion)
python fusion/generate_fused_features.py  # Cần config thêm

# 5. Train model
python trainer/trainer.py --config configs/CTRGCN_mm_wlauslan.yaml
```

---

## ✨ Ready to go!
Bro bạn có thể bắt đầu từ **BƯỚC 1** luôn nhé! 🚀
