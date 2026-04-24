import os
import csv
import random

# 1. Đường dẫn tới thư mục chứa tất cả video
video_dir = r"d:\20252\Lab\viết báo\References\CE-GCN\Code\CE-GCN\data\Multi-VSL200"
output_dir = r"d:\20252\Lab\viết báo\References\CE-GCN\Code\CE-GCN\data\Multi-VSL200\Multi-VSL200"

os.makedirs(output_dir, exist_ok=True)

data = []

if not os.path.exists(video_dir):
    print(f"Lỗi: Thư mục {video_dir} không tồn tại!")
    exit(1)

# Lấy tất cả file .mp4 trong thư mục data
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

# 2. Tạo lookuptable.csv
with open(os.path.join(output_dir, "lookuptable.csv"), "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label_id", "gloss"])
    writer.writerow([0, "sign"])

# Đọc tất cả video trong thư mục data
for video_file in video_files:
    data.append({"file_name": video_file, "label_id": 0})

if not data:
    print(f"Không tìm thấy file .mp4 nào trong {video_dir}. Vui lòng kiểm tra lại!")
    exit(1)

# Gom nhóm dữ liệu theo từng nhãn để chia đều (stratified split)
data_by_label = {}
for item in data:
    lbl = item["label_id"]
    if lbl not in data_by_label:
        data_by_label[lbl] = []
    data_by_label[lbl].append(item)

train_data, val_data, test_data = [], [], []

# 3. Chia tập dữ liệu: Train (70%), Val (15%), Test (15%)
random.seed(42)
for lbl, items in data_by_label.items():
    random.shuffle(items)
    n = len(items)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train_data.extend(items[:n_train])
    val_data.extend(items[n_train:n_train + n_val])
    test_data.extend(items[n_train + n_val:])

# Trộn ngẫu nhiên lại các tập
random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

# 4. Xuất ra 3 file CSV
def write_csv(filename, dataset):
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "label_id"])
        for item in dataset:
            writer.writerow([item["file_name"], item["label_id"]])

write_csv("train_labels.csv", train_data)
write_csv("val_labels.csv", val_data)
write_csv("test_labels.csv", test_data)

print(f"Đã tạo xong CSV! Tổng cộng: Train({len(train_data)}), Val({len(val_data)}), Test({len(test_data)})")
