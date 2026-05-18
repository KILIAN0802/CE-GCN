import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# Import Model và DataLoader
from models.Hybrid_GCN import HybridGCN
from dataset.reader import FeatureReader

# ==========================================
# CẤU HÌNH THÔNG SỐ (HYPERPARAMETERS)
# ==========================================
class Config:
    # Model
    num_class = 200
    num_point = 50
    num_person = 1
    base_channel = 64
    
    # Dataset
    train_data_path = './data/fused_features/train_fused_features'
    train_label_path = './data/fused_features/train_fused_features/train_label.csv'
    val_data_path = './data/fused_features/val_fused_features'
    val_label_path = './data/fused_features/val_fused_features/val_label.csv'
    
    # Training
    epochs = 60
    batch_size = 32
    num_workers = 4
    learning_rate = 0.01
    weight_decay = 0.0004
    label_smoothing = 0.1
    warmup_epochs = 5
    
    # Checkpoint
    save_dir = './results/train_basic'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# LEARNING RATE SCHEDULER WARMUP
# ==========================================
def adjust_learning_rate(optimizer, epoch, cfg):
    """ Warmup learning rate strategy """
    if epoch <= cfg.warmup_epochs:
        # Tăng tuyến tính trong giai đoạn warmup
        lr = cfg.learning_rate * (epoch / cfg.warmup_epochs)
    else:
        # Cosine Annealing sau warmup
        progress = (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs)
        lr = cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# ==========================================
# TRAINING SCRIPT
# ==========================================
def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print(f"=== INITIALIZING HYBRID GCN TRAINING ===")
    print(f"Device: {cfg.device}")
    
    # 1. Khởi tạo Dataset
    print("[1] Loading Data...")
    train_dataset = FeatureReader(cfg.train_data_path, cfg.train_label_path, num_classes=cfg.num_class, window_size=64)
    val_dataset = FeatureReader(cfg.val_data_path, cfg.val_label_path, num_classes=cfg.num_class, window_size=64)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    # 2. Khởi tạo Model
    print("[2] Initializing HybridGCN Model...")
    model = HybridGCN(
        num_class=cfg.num_class, 
        num_point=cfg.num_point, 
        num_person=cfg.num_person, 
        base_channel=cfg.base_channel
    ).to(cfg.device)
    
    # 3. Loss & Optimizer (Label Smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    best_acc = 0.0
    
    # 4. Vòng lặp huấn luyện
    print("[3] Starting Training Loop...")
    for epoch in range(1, cfg.epochs + 1):
        # Điều chỉnh Learning Rate
        current_lr = adjust_learning_rate(optimizer, epoch, cfg)
        
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Train]", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Ổn định Gradient
            optimizer.step()
            
            # Thống kê
            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'lr': current_lr})
            
        train_acc = train_correct / total_samples * 100
        train_loss = train_loss / total_samples
        
        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += inputs.size(0)
                
        val_acc = val_correct / val_total * 100
        val_loss = val_loss / val_total
        
        print(f"Epoch {epoch:03d}/{cfg.epochs} | LR: {current_lr:.5f} | Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}%")
        
        # Lưu Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(cfg.save_dir, 'best_hybrid_gcn.pth')
            torch.save(model.state_dict(), save_path)
            print(f" -> Saved Best Model with Acc: {best_acc:.2f}% at {save_path}")

    print(f"=== TRAINING COMPLETED! BEST VAL ACC: {best_acc:.2f}% ===")

if __name__ == '__main__':
    main()
