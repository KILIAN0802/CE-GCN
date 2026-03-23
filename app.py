import torch
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import mediapipe as mp
import shutil
import os

try:
    MP_SOLUTIONS = mp.solutions
except AttributeError:
    # Some mediapipe builds expose solutions under mediapipe.python.
    from mediapipe.python import solutions as MP_SOLUTIONS

# Import các module từ dự án của bạn
from models.CTRGCN.ctrgcn_baseline import Model
from fusion.early_fusion import early_fusion
from mediapipe_kpt.extract_kpt import extract_from_frame
from detection_yolo.yolo_crop_VSL200 import auto_center_and_scale

app = FastAPI()

# --- 1. KHỞI TẠO MODEL & TRỌNG SỐ ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_args = {
    "num_class": 200,
    "num_point": 46,
    "num_person": 1,
    "in_channels": 9,
    "graph": "models.graph.vsl_graph.Graph",
    "graph_args": {"strategy": "spatial", "layout": "vsl_layout"}
}

# Khởi tạo kiến trúc CTR-GCN
model = Model(**model_args).to(DEVICE)

weights_path = "./results/noJDMA/transfer_joint/best_model.pth"
ckpt = torch.load(weights_path, map_location=DEVICE)
model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
model.eval()

# Load YOLO để crop người
yolo_model = YOLO("yolov8n-pose.pt") 

# --- 2. PIPELINE XỬ LÝ ---
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_frames_kpts = []
    
    with MP_SOLUTIONS.hands.Hands() as hands, MP_SOLUTIONS.pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Bước A: YOLO Crop (Tùy chọn nhưng khuyến khích để tăng độ chính xác)
            # Tạm thời lấy frame gốc để MediaPipe trích xuất
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Bước B: Trích xuất 46 keypoints
            kpts = extract_from_frame(rgb_frame, hands, pose)
            all_frames_kpts.append(kpts)
            
    cap.release()
    
    # Bước C: Early Fusion 9 kênh
    # Chuyển list thành numpy (T, 46, 3)
    kpts_np = np.array(all_frames_kpts)
    fused_tensor = early_fusion(kpts_np, max_frames=64) # (9, 64, 46)
    
    return torch.from_numpy(fused_tensor).unsqueeze(0).to(DEVICE) # (1, 9, 64, 46)

# --- 3. ENDPOINT API ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lưu tạm video
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Chạy pipeline
        input_tensor = process_video(temp_path)
        
        # Dự đoán
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            
        return {"label_id": prediction, "status": "success"}
    
    except Exception as e:
        return {"error": str(e), "status": "failed"}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
# Hàm chính 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)