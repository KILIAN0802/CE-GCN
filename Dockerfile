# Bước 1: Sử dụng Python 3.11 làm nền tảng
FROM python:3.11-slim

# Bước 2: Cài đặt các thư viện hệ thống cần thiết cho OpenCV và MediaPipe
# Trên Debian slim mới, libgl1-mesa-glx đã được thay bởi libgl1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Bước 3: Thiết lập thư mục làm việc
WORKDIR /app

# Bước 4: Sao chép và cài đặt các thư viện Python
# Tận dụng cache của Docker bằng cách copy requirements trước
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Cài đặt thêm các thư viện cho API Server
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Bước 5: Sao chép toàn bộ mã nguồn vào Container
# Lưu ý: .gitignore sẽ tự động loại bỏ các thư mục nặng như data/, results/
COPY . .

# Bước 6: Tạo thư mục chứa trọng số (để mount từ bên ngoài vào)
# RUN mkdir -p results/noJDMA/transfer_joint

# Bước 7: Mở cổng 8000 cho FastAPI
EXPOSE 8000

# Bước 8: Lệnh khởi chạy API Server
CMD ["uvicorn", "app.py:app", "--host", "0.0.0.0", "--port", "8000"]