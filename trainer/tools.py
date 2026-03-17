import os
import yaml
import random
import torch
import numpy as np
import re

# ================================
# LOAD CONFIG & UTILS
# ================================
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, save_path):
    ensure_dir(os.path.dirname(save_path))
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, save_path)
    print(f"[INFO] Saved checkpoint -> {save_path}")

def load_checkpoint(model, optimizer, load_path, device="cuda"):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {load_path}")
    ckpt = torch.load(load_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[INFO] Loaded checkpoint from {load_path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.value = 0; self.sum = 0; self.count = 0; self.avg = 0
    def update(self, val, n=1): self.value = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().contiguous()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        acc = correct_k.mul_(100.0 / batch_size)
        results.append(acc)
    return results

# =========================================================
# TRANSFER LOAD (GREEDY MATCH + CLONE STRATEGY)
# =========================================================
def load_transfer_weights(model, weights_path, device):
    print(f"\n[INFO] >>> Bắt đầu Transfer Learning từ: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"[ERROR] Không tìm thấy file: {weights_path}")
        return model

    try:
        pretrained_dict = torch.load(weights_path, map_location=device)
    except Exception as e:
        print(f"[ERROR] Lỗi load file: {e}")
        return model
    
    # Chuẩn hóa format dict
    if isinstance(pretrained_dict, dict):
        if 'model' in pretrained_dict: pretrained_dict = pretrained_dict['model']
        elif 'model_state_dict' in pretrained_dict: pretrained_dict = pretrained_dict['model_state_dict']
        elif 'state_dict' in pretrained_dict: pretrained_dict = pretrained_dict['state_dict']

    model_dict = model.state_dict()
    new_state_dict = {}
    
    # Check biến môi trường
    is_bone_target = os.environ.get('TARGET_STREAM') == 'BONE'
    is_vel_target = os.environ.get('TARGET_STREAM') == 'VELOCITY'
    
    if is_bone_target:
        print(">>> [MODE] CLONE & EVOLVE: Joint(0-2) -> BONE(6-8)")
    elif is_vel_target:
        print(">>> [MODE] CLONE & EVOLVE: Joint(0-2) -> VELOCITY(3-5)")
    else:
        if "bone" in weights_path.lower():
            print(">>> [MODE] Load NTU Bone -> Kênh 6-8")
            is_bone_target = True
        elif "vel" in weights_path.lower():
            print(">>> [MODE] Load NTU Velocity -> Kênh 3-5")
            is_vel_target = True
        else:
            print(">>> [MODE] Load Normal (Joint -> Joint)")

    # --- HÀM XỬ LÝ GÁN WEIGHT (FIXED) ---
    def try_assign(source_v, target_v):
        # 1. DIRECT MATCH (Khớp hoàn toàn)
        if source_v.shape == target_v.shape:
            return source_v

        # Kiểm tra an toàn: Phải là Tensor 4 chiều mới check channel
        if len(source_v.shape) != 4 or len(target_v.shape) != 4:
            return None

        # [FIX CRITICAL BUG] Bắt buộc số Output Channels (dim 0) và Kernel Size (dim 2,3) phải khớp
        # Nếu không khớp output (ví dụ 8 vs 64) thì tuyệt đối không được copy
        if source_v.shape[0] != target_v.shape[0]: 
            return None
        if source_v.shape[2:] != target_v.shape[2:]:
            return None

        # 2. CLONE STRATEGY (9 -> 9)
        if source_v.shape[1] == 9 and target_v.shape[1] == 9:
            new_w = target_v.clone()
            knowledge = source_v[:, 0:3, :, :]
            if is_bone_target: new_w[:, 6:9, :, :] = knowledge
            elif is_vel_target: new_w[:, 3:6, :, :] = knowledge
            else: new_w = source_v
            return new_w

        # 3. PARTIAL LOAD (3 -> 9)
        elif source_v.shape[1] == 3 and target_v.shape[1] == 9:
            new_w = target_v.clone()
            if is_bone_target: new_w[:, 6:9, :, :] = source_v
            elif is_vel_target: new_w[:, 3:6, :, :] = source_v
            else: new_w[:, 0:3, :, :] = source_v
            return new_w
        
        return None
    # ----------------------------------------------

    # Danh sách các Layer ID cần quét
    layer_ids = ['l1.', 'l2.', 'l3.', 'l4.', 'l5.', 'l6.', 'l7.', 'l8.', 'l9.', 'l10.']
    layer_ids.extend(['data_bn.', 'fc.'])

    matched_keys_local = set()
    loaded_count = 0

    for lid in layer_ids:
        src_keys = [k for k in pretrained_dict.keys() if lid in k.replace('module.', '')]
        tgt_keys = [k for k in model_dict.keys() if lid in k]
        
        for s_key in src_keys:
            s_val = pretrained_dict[s_key]
            s_name_clean = s_key.replace('module.', '')
            s_suffix = s_name_clean.split('.')[-1]

            # A. Regex Mapping
            mappings = [
                (r'\.convs\.0\.', '.conv1.'), (r'\.convs\.1\.', '.conv2.'),
                (r'\.convs\.2\.', '.conv3.'), (r'\.convs\.3\.', '.conv4.'),
                (r'\.branches\.2\.0\.', '.conv.'), (r'\.branches\.2\.1\.', '.bn.'),
                (r'\.branches\.0\.0\.', '.conv.'), (r'\.branches\.0\.1\.', '.bn.'),
            ]
            mapped_name = s_name_clean
            for ptrn, repl in mappings:
                mapped_name = re.sub(ptrn, repl, mapped_name)
            
            if mapped_name in model_dict and mapped_name in tgt_keys:
                t_val = model_dict[mapped_name]
                assigned_w = try_assign(s_val, t_val)
                if assigned_w is not None:
                    new_state_dict[mapped_name] = assigned_w
                    matched_keys_local.add(mapped_name)
                    loaded_count += 1
                    continue

            # B. Greedy Shape Match
            for t_key in tgt_keys:
                if t_key in matched_keys_local: continue
                t_val = model_dict[t_key]
                t_suffix = t_key.split('.')[-1]

                if s_suffix == t_suffix:
                    assigned_w = try_assign(s_val, t_val)
                    if assigned_w is not None:
                        new_state_dict[t_key] = assigned_w
                        matched_keys_local.add(t_key)
                        loaded_count += 1
                        break 

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    
    print("-" * 50)
    print(f"[SUCCESS] Đã nạp thành công: {loaded_count} layers")
    print("-" * 50)

    return model