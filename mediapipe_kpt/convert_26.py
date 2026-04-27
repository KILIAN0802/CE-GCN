import os
import numpy as np
from tqdm import tqdm
import argparse

# --- DANH SÁCH 26 ĐIỂM CẦN GIỮ LẠI ---
SELECTED_INDICES = [
    # Tay phải (11 điểm)
    0, 2, 4, 5, 8, 9, 12, 13, 16, 17, 20,
    
    # Tay trái (11 điểm)
    21, 23, 25, 26, 29, 30, 33, 34, 37, 38, 41,
    
    # Cơ thể (4 điểm)
    42, 43, 44, 45
]

def convert_checkpoints(input_dir, output_dir):
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # Lấy tất cả các file .npy trong thư mục
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"[CẢNH BÁO] Không tìm thấy file .npy nào trong thư mục: {input_dir}")
        return

    for f in tqdm(npy_files, desc="Đang convert 46 -> 26 keypoints"):
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f)

        try:
            # Load file checkpoint cũ (Shape dự kiến: T, 46, 3)
            data_46 = np.load(in_path)

            # Kiểm tra xem file có đúng format 46 điểm ở trục 1 không
            if len(data_46.shape) == 3 and data_46.shape[1] == 46:
                # Lọc lấy 26 điểm
                data_26 = data_46[:, SELECTED_INDICES, :]
                
                # Lưu file mới
                np.save(out_path, data_26)
            else:
                print(f"\n[BỎ QUA] {f} có shape không đúng chuẩn 46 điểm: {data_46.shape}")
                
        except Exception as e:
            print(f"\n[LỖI] Không thể đọc/ghi file {f}. Chi tiết: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert checkpoints từ 46 điểm xuống 26 điểm")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Thư mục chứa các file .npy 46 điểm cũ")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Thư mục lưu các file .npy 26 điểm mới")

    args = parser.parse_args()

    convert_checkpoints(args.input_dir, args.output_dir)
    print("\nHoàn tất convert!")