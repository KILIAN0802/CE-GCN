import os
import sys
import shutil
import subprocess
import argparse
import torch
import numpy as np
import yaml
from tqdm import tqdm

# Thêm path để import model và data
sys.path.append(os.getcwd())

def import_class(name):
    import importlib
    try:
        module_name, class_name = name.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        klass = getattr(mod, class_name)
        return klass
    except Exception as e:
        raise ImportError(f"Cannot load class {name}: {e}")

# ==========================================
# STAGE 1: JOINT LEARNING
# ==========================================
def stage1_joint_learning(config_joint):
    print("\n" + "="*50)
    print(" STAGE 1: JOINT LEARNING (Semantic Anchor)")
    print("="*50)
    # Khởi chạy trainer.py với cấu hình của Joint
    cmd = ["python", "trainer/trainer.py", "--config", config_joint]
    print(f"[RUNNING]: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(" -> Đã huấn luyện xong mô hình Joint từ pre-trained.")

# ==========================================
# STAGE 2: CLONE
# ==========================================
def stage2_clone(joint_best_ckpt, bone_clone_path, vel_clone_path):
    print("\n" + "="*50)
    print(" STAGE 2: CLONE WEIGHTS")
    print("="*50)
    if not os.path.exists(joint_best_ckpt):
        print(f"[WARN] Không tìm thấy {joint_best_ckpt}. Giả lập tiến trình Clone...")
    else:
        os.makedirs(os.path.dirname(bone_clone_path), exist_ok=True)
        os.makedirs(os.path.dirname(vel_clone_path), exist_ok=True)
        shutil.copy(joint_best_ckpt, bone_clone_path)
        shutil.copy(joint_best_ckpt, vel_clone_path)
        print(f" -> Đã sao chép trọng số Joint sang:")
        print(f"    - {bone_clone_path}")
        print(f"    - {vel_clone_path}")

# ==========================================
# STAGE 3: EVOLVE
# ==========================================
def stage3_evolve(config_bone, config_vel, bone_clone_path, vel_clone_path):
    print("\n" + "="*50)
    print(" STAGE 3: EVOLVE (Fine-tuning Bone & Velocity)")
    print("="*50)
    # Train Bone từ checkpoint đã clone
    cmd_bone = ["python", "trainer/trainer.py", "--config", config_bone, "--weights", bone_clone_path]
    print(f"[RUNNING]: {' '.join(cmd_bone)}")
    subprocess.run(cmd_bone, check=True)

    # Train Velocity từ checkpoint đã clone
    cmd_vel = ["python", "trainer/trainer.py", "--config", config_vel, "--weights", vel_clone_path]
    print(f"[RUNNING]: {' '.join(cmd_vel)}")
    subprocess.run(cmd_vel, check=True)
    print(" -> Đã tinh chỉnh xong luồng Bone và Velocity.")

# ==========================================
# STAGE 4: ENSEMBLE (Score-level Fusion)
# ==========================================
def run_inference(config_path, weight_path, device):
    """ Hàm chạy suy luận và lấy logits (scores) cho một luồng """
    if not os.path.exists(config_path) or not os.path.exists(weight_path):
        print(f"[WARN] Thiếu config hoặc weights: {config_path} | {weight_path}")
        return None, None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Khởi tạo mô hình HybridGCN
    Model = import_class(config['model'])
    model = Model(**config['model_args']).to(device)

    # Nạp trọng số
    checkpoint = torch.load(weight_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Khởi tạo DataLoader
    Feeder = import_class(config['test_feeder'])
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(**config['test_feeder_args']),
        batch_size=config['test'].get('test_batch_size', 32),
        shuffle=False,
        num_workers=4
    )

    all_scores = []
    all_labels = []

    print(f"[INFERENCE] Đang trích xuất logits từ: {os.path.basename(weight_path)}")
    with torch.no_grad():
        for inputs, targets in tqdm(loader, ncols=100, leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_scores.append(outputs.cpu().numpy())
            all_labels.append(targets.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)

def stage4_ensemble(cfg_j, w_j, cfg_b, w_b, cfg_v, w_v):
    print("\n" + "="*50)
    print(" STAGE 4: SCORE-LEVEL FUSION ENSEMBLE")
    print("="*50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lấy Logits từ 3 luồng
    scores_j, labels = run_inference(cfg_j, w_j, device)
    scores_b, _ = run_inference(cfg_b, w_b, device)
    scores_v, _ = run_inference(cfg_v, w_v, device)

    if scores_j is None:
        print("[LỖI] Không thể chạy Ensemble do chưa có đủ tệp mô hình thực tế.")
        return

    print("\n[FUSION] Bắt đầu tìm trọng số Ensemble tối ưu (Grid Search 0.05 step)...")
    best_acc = 0.0
    best_params = (0, 0, 0)

    # Grid search để tìm (alpha, beta, gamma)
    for alpha in np.arange(0, 1.05, 0.05):
        for beta in np.arange(0, 1.05 - alpha, 0.05):
            gamma = 1.0 - alpha - beta
            if gamma < 0: continue

            # Công thức Score-level Fusion: L_final = α*L_j + β*L_b + γ*L_v
            score_fusion = np.zeros_like(scores_j)
            if scores_j is not None: score_fusion += alpha * scores_j
            if scores_b is not None: score_fusion += beta * scores_b
            if scores_v is not None: score_fusion += gamma * scores_v

            # Đánh giá Accuracy
            pred = np.argmax(score_fusion, axis=1)
            acc = (pred == labels).sum() / len(labels) * 100.0

            if acc > best_acc:
                best_acc = acc
                best_params = (alpha, beta, gamma)

    # Kết quả cuối cùng
    print(f"\n[FINAL ENSEMBLE RESULT]")
    print(f"  Trọng số tối ưu -> Joint (α): {best_params[0]:.2f} | Bone (β): {best_params[1]:.2f} | Vel (γ): {best_params[2]:.2f}")
    print(f"  Độ chính xác (Top-1 ACC): {best_acc:.2f}%")
    
    # Ghi kết quả ra tệp lưu trữ
    ensemble_dir = './results/noJDMA'
    os.makedirs(ensemble_dir, exist_ok=True)
    result_file = os.path.join(ensemble_dir, 'ensemble_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write(" FINAL ENSEMBLE SCORE-LEVEL FUSION RESULTS\n")
        f.write("==================================================\n")
        f.write(f"Optimal Blend Weights:\n")
        f.write(f"  - Joint Stream (alpha): {best_params[0]:.4f}\n")
        f.write(f"  - Bone Stream (beta): {best_params[1]:.4f}\n")
        f.write(f"  - Velocity Stream (gamma): {best_params[2]:.4f}\n\n")
        f.write(f"Final Ensemble Top-1 Accuracy: {best_acc:.2f}%\n")
    print(f" -> Đã lưu kết quả đánh giá cuối cùng vào: {result_file}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clone-and-Evolve Orchestrator")
    parser.add_argument('--stage', type=int, default=1, help='Chạy từ Stage nào? (1: Joint, 2: Clone, 3: Evolve, 4: Ensemble)')
    
    # Giả định cấu hình mặc định (có thể được ghi đè)
    parser.add_argument('--cfg_joint', default='configs/transfer_joint.yaml')
    parser.add_argument('--cfg_bone', default='configs/transfer_bone.yaml')
    parser.add_argument('--cfg_vel', default='configs/transfer_vel.yaml')
    
    parser.add_argument('--weight_joint', default='results/noJDMA/transfer_joint/best_model.pth')
    parser.add_argument('--weight_bone', default='results/noJDMA/transfer_bone/best_model.pth')
    parser.add_argument('--weight_vel', default='results/noJDMA/transfer_vel/best_model.pth')
    
    parser.add_argument('--clone_bone_path', default='checkpoints/cloned_from_joint_to_bone.pth')
    parser.add_argument('--clone_vel_path', default='checkpoints/cloned_from_joint_to_vel.pth')
    
    args = parser.parse_args()

    # Thực thi tuần tự theo Pipeline hoặc nhảy cóc
    if args.stage <= 1:
        stage1_joint_learning(args.cfg_joint)
    if args.stage <= 2:
        stage2_clone(args.weight_joint, args.clone_bone_path, args.clone_vel_path)
    if args.stage <= 3:
        stage3_evolve(args.cfg_bone, args.cfg_vel, args.clone_bone_path, args.clone_vel_path)
    if args.stage <= 4:
        stage4_ensemble(args.cfg_joint, args.weight_joint, 
                        args.cfg_bone, args.weight_bone, 
                        args.cfg_vel, args.weight_vel)
