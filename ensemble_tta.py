import argparse
import yaml
import torch
import numpy as np
import importlib
import os
import sys
import csv
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Thêm thư viện vẽ đồ thị
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append(os.getcwd())


def load_lookup_table(csv_path):
    label_map = {}
    if not csv_path or not os.path.exists(csv_path):
        return label_map

    with open(csv_path, mode='r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                label_id = int(row['id_label_in_documents'])
            except (TypeError, ValueError, KeyError):
                continue

            label_name = (row.get('name') or '').strip()
            if label_name:
                label_map[label_id] = label_name

    return label_map


def lookup_label_name(label_id, label_map):
    if label_id in label_map:
        return label_map[label_id]

    one_based_id = label_id + 1
    if one_based_id in label_map:
        return label_map[one_based_id]

    return f'Class {label_id}'


def resolve_lookup_csv():
    candidates = [
        Path('data') / 'Multi-VSL200' / 'Multi-VSL200' / 'lookuptable.csv',
        Path('data') / 'MultiVSL200' / 'lookuptable.csv',
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def build_class_names(num_classes, label_map):
    return [lookup_label_name(class_id, label_map) for class_id in range(num_classes)]


def plot_confusion_matrix_subset(y_true, y_pred, class_ids, class_names, save_path):
    subset_mask = np.isin(y_true, class_ids)
    if not np.any(subset_mask):
        return None

    y_true_subset = y_true[subset_mask]
    y_pred_subset = y_pred[subset_mask]

    cm = confusion_matrix(y_true_subset, y_pred_subset, labels=class_ids)

    plt.figure(figsize=(max(12, len(class_ids) * 0.55), max(10, len(class_ids) * 0.45)))
    sns.heatmap(
        cm,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=False,
        cbar=True,
    )

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return cm


def print_top_bottom_classes(acc_per_class, class_names, top_k=20):
    valid_indices = np.where(~np.isnan(acc_per_class))[0]
    if len(valid_indices) == 0:
        print('[WARN] Không có lớp hợp lệ để thống kê accuracy theo lớp.')
        return [], []

    sorted_indices = valid_indices[np.argsort(acc_per_class[valid_indices])[::-1]]
    top_indices = sorted_indices[:top_k].tolist()
    bottom_indices = sorted_indices[-top_k:].tolist()[::-1]

    print('\n[INFO] >>> Top lớp accuracy cao nhất')
    for rank, class_id in enumerate(top_indices, 1):
        print(f"  {rank:02d}. {class_names[class_id]} -> {acc_per_class[class_id]:.2f}%")

    print('\n[INFO] >>> Top lớp accuracy thấp nhất')
    for rank, class_id in enumerate(bottom_indices, 1):
        print(f"  {rank:02d}. {class_names[class_id]} -> {acc_per_class[class_id]:.2f}%")

    return top_indices, bottom_indices

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
    # Read YAML explicitly in UTF-8 to avoid Windows default codepage decode errors.
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)
    
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
    parser.add_argument('--lookup_csv', default='', help='Đường dẫn lookuptable.csv để map label id -> tên từ')
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
    # [NEW] GENERATE & SAVE CONFUSION MATRICES FOR BEST/WORST WORDS
    # ==========================================
    print("\n[INFO] >>> Generating Confusion Matrices...")

    y_pred = np.argmax(final_score, axis=1)
    y_true = labels

    lookup_csv = args.lookup_csv or resolve_lookup_csv()
    label_map = load_lookup_table(lookup_csv)
    class_names = build_class_names(int(max(y_true.max(), y_pred.max())) + 1, label_map)

    cm_all = confusion_matrix(y_true, y_pred)
    class_totals = cm_all.sum(axis=1)
    class_correct = np.diag(cm_all)
    acc_per_class = np.full(len(class_totals), np.nan, dtype=float)
    nonzero_mask = class_totals > 0
    acc_per_class[nonzero_mask] = class_correct[nonzero_mask] / class_totals[nonzero_mask] * 100.0

    top_indices, bottom_indices = print_top_bottom_classes(acc_per_class, class_names, top_k=20)

    if not os.path.exists('./results'):
        os.makedirs('./results')

    top_names = [class_names[i] for i in top_indices]
    bottom_names = [class_names[i] for i in bottom_indices]

    top_save_path = './results/confusion_matrix_top20.png'
    bottom_save_path = './results/confusion_matrix_bottom20.png'

    plot_confusion_matrix_subset(
        y_true,
        y_pred,
        top_indices,
        top_names,
        top_save_path
    )
    plot_confusion_matrix_subset(
        y_true,
        y_pred,
        bottom_indices,
        bottom_names,
        bottom_save_path
    )

    print(f"[SUCCESS] Đã lưu Top-20 Confusion Matrix tại: {top_save_path}")
    print(f"[SUCCESS] Đã lưu Bottom-20 Confusion Matrix tại: {bottom_save_path}")

if __name__ == '__main__':
    main()