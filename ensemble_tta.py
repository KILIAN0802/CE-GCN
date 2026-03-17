import argparse
import yaml
import torch
import numpy as np
import importlib
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

# Thêm thư viện vẽ đồ thị
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append(os.getcwd())

def import_class(name):
    try:
        module_name, class_name = name.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        klass = getattr(mod, class_name)
        return klass
    except Exception as e:
        raise ImportError(f"Cannot load class {name}: {e}")

def get_tta_loader(config, tta_enabled=False):
    Feeder = import_class(config['test_feeder'])
    
    args = config['test_feeder_args']
    
    # [TTA MAGIC] Bật tính năng Augmentation cho tập Test
    if tta_enabled:
        args['random_shift'] = True
        args['random_move'] = True
        # args['random_choose'] = True # Có thể bật nếu muốn
        # print("   -> [TTA] Enabled Random Shift & Move")
    else:
        args['random_shift'] = False
        args['random_move'] = False
    
    # Xác định batch size
    batch_size = 32
    if 'test' in config and 'test_batch_size' in config['test']:
        batch_size = config['test']['test_batch_size']
        
    loader = DataLoader(Feeder(**args), batch_size=batch_size, 
                        shuffle=False, num_workers=4, drop_last=False)
    return loader

def run_inference_tta(config_path, weight_path, device_id, tta_times=1):
    if not config_path or not weight_path: return None, None
    
    print(f"\n[INFO] >>> Inference TTA ({tta_times}x): {os.path.basename(config_path)}")
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    Model = import_class(config['model'])
    model = Model(**config['model_args']).to(device)
    
    # Load Weights
    checkpoint = torch.load(weight_path, map_location=device)
    if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint: model.load_state_dict(checkpoint['model'])
    else: model.load_state_dict(checkpoint)
    model.eval()
    
    final_scores = None
    final_labels = None

    # [LOOP TTA] Chạy nhiều vòng lặp
    for i in range(tta_times):
        # Chỉ bật Augmentation từ vòng thứ 2 trở đi (Vòng 0 là dữ liệu gốc sạch)
        is_aug = (i > 0)
        loader = get_tta_loader(config, tta_enabled=is_aug)
        
        score_frag = []
        label_frag = []
        
        desc = f"   Round {i+1}/{tta_times} [{'Aug' if is_aug else 'Orig'}]"
        
        with torch.no_grad():
            for inputs, targets in tqdm(loader, ncols=100, unit="batch", desc=desc):
                inputs = inputs.to(device)
                outputs = model(inputs)
                score_frag.append(outputs.cpu().numpy())
                label_frag.append(targets.numpy())
        
        scores = np.concatenate(score_frag)
        labels = np.concatenate(label_frag)
        
        if final_scores is None:
            final_scores = scores
            final_labels = labels
        else:
            final_scores += scores # Cộng dồn điểm
            
    # Chia trung bình
    final_scores /= tta_times
    
    # Tính Acc riêng lẻ
    acc = (np.argmax(final_scores, axis=1) == final_labels).sum() / len(final_labels) * 100
    print(f"   -> Accuracy TTA: {acc:.2f}%")
    
    return final_scores, final_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config1', default='') 
    parser.add_argument('--weight1', default='')
    parser.add_argument('--config2', default='') 
    parser.add_argument('--weight2', default='')
    parser.add_argument('--config3', default='') 
    parser.add_argument('--weight3', default='')
    parser.add_argument('--tta_times', type=int, default=5, help='Số lần lặp TTA (nên là 3-5)')
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()

    # 1. Joint TTA
    s1, l1 = run_inference_tta(args.config1, args.weight1, args.device, args.tta_times)

    # 2. Bone TTA
    torch.cuda.empty_cache()
    s2, l2 = run_inference_tta(args.config2, args.weight2, args.device, args.tta_times)

    # 3. Velocity TTA
    torch.cuda.empty_cache()
    s3, l3 = run_inference_tta(args.config3, args.weight3, args.device, args.tta_times)

    # GRID SEARCH CHI TIẾT HƠN (Bước nhảy 0.05)
    print("\n[INFO] >>> Searching Best Ensemble (Finer Grid)...")
    labels = l1
    best_acc = 0
    best_params = (0, 0, 0)
    
    # Bước nhảy 0.05 để tìm kỹ hơn
    for alpha in np.arange(0, 1.05, 0.05):
        for beta in np.arange(0, 1.05 - alpha, 0.05):
            gamma = 1 - alpha - beta
            if gamma < 0: continue
            
            score_fusion = np.zeros_like(s1)
            if s1 is not None: score_fusion += alpha * s1
            if s2 is not None: score_fusion += beta * s2
            if s3 is not None: score_fusion += gamma * s3
            
            pred = np.argmax(score_fusion, axis=1)
            acc = (pred == labels).sum() / len(labels) * 100
            
            if acc > best_acc:
                best_acc = acc
                best_params = (alpha, beta, gamma)

    print("-" * 50)
    print(f"   FINAL RESULT (With TTA x{args.tta_times}):")
    print(f"   Joint: {best_params[0]:.2f} | Bone: {best_params[1]:.2f} | Vel: {best_params[2]:.2f}")
    print(f"   ACCURACY: {best_acc:.2f}%")
    print("-" * 50)
    
    # Check Top-5 Accuracy
    final_score = np.zeros_like(s1)
    if s1 is not None: final_score += best_params[0] * s1
    if s2 is not None: final_score += best_params[1] * s2
    if s3 is not None: final_score += best_params[2] * s3

    _, pred5 = torch.tensor(final_score).topk(5, 1, True, True)
    pred5 = pred5.t()
    correct5 = pred5.eq(torch.tensor(labels).view(1, -1).expand_as(pred5))
    top5_acc = correct5[:5].reshape(-1).float().sum(0) * 100. / len(labels)
    print(f"   TOP-5 ACCURACY: {top5_acc:.2f}%")

    # ==========================================
    # [NEW] GENERATE & SAVE CONFUSION MATRIX
    # ==========================================
    print("\n[INFO] >>> Generating Confusion Matrix...")
    
    # Lấy nhãn dự đoán từ final_score đã tính ở trên
    y_pred = np.argmax(final_score, axis=1)
    y_true = labels

    # Tính Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Vẽ biểu đồ Heatmap
    plt.figure(figsize=(20, 20)) # Kích thước lớn vì 200 lớp
    sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
    
    plt.title(f'Confusion Matrix (Ensemble Acc: {best_acc:.2f}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Lưu file
    save_path = './results/confusion_matrix_final.png'
    if not os.path.exists('./results'):
        os.makedirs('./results')
        
    plt.savefig(save_path)
    plt.close()
    
    print(f"[SUCCESS] Đã lưu Confusion Matrix tại: {save_path}")

if __name__ == '__main__':
    main()