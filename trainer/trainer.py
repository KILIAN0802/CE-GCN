import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys, os
import importlib
import numpy as np

# Add path to import other modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import tools
from trainer.tools import (
    load_config,
    set_seed,
    AverageMeter,
    accuracy,
    save_checkpoint,
    load_transfer_weights 
)

# ============================
# UTILS & HELPERS
# ============================
def import_class(name):
    try:
        module_name, class_name = name.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        klass = getattr(mod, class_name)
        return klass
    except Exception as e:
        raise ImportError(f"Cannot load class '{name}': {e}")

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def mixup_data(x, y, alpha=0.2, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================
# TRAIN ONE EPOCH
# ============================================
def train_one_epoch(model, criterion, optimizer, loader, device, epoch, num_epochs, use_mixup=True):
    model.train()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter() 

    # Thanh tiến trình (Progress Bar)
    pbar = tqdm(enumerate(loader), total=len(loader), 
                ncols=120, unit="batch", colour='green',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    for batch_idx, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # --- Mixup Logic ---
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2, device=device)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            targets_a = labels

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
        optimizer.step()

        # Tính Accuracy (Top-1 & Top-5)
        acc1, acc5 = accuracy(outputs, targets_a, topk=(1, 5))
        
        loss_meter.update(loss.item(), inputs.size(0))
        acc1_meter.update(acc1.item(), inputs.size(0))
        acc5_meter.update(acc5.item(), inputs.size(0))

        # Hiển thị trên thanh process bar
        pbar.set_postfix({
            'loss': f"{loss_meter.avg:.4f}", 
            'top1': f"{acc1_meter.avg:.2f}%",
            'top5': f"{acc5_meter.avg:.2f}%"
        })

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg

# ============================================
# VALIDATION
# ============================================
@torch.no_grad()
def validate(model, criterion, loader, device, epoch, num_epochs, phase='VAL'):
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter() 

    print(f"\n>>> [{phase}] Epoch {epoch}/{num_epochs}")
    pbar = tqdm(loader, total=len(loader), 
                ncols=120, unit="batch", colour='blue',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        loss_meter.update(loss.item(), inputs.size(0))
        acc1_meter.update(acc1.item(), inputs.size(0))
        acc5_meter.update(acc5.item(), inputs.size(0))

        pbar.set_postfix({
            'loss': f"{loss_meter.avg:.4f}", 
            'top1': f"{acc1_meter.avg:.2f}%",
            'top5': f"{acc5_meter.avg:.2f}%"
        })

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg

# ============================================
# MAIN FUNCTION
# ============================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/CTRGCN.yaml')
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--weights', default=None, help='load weights for test')
    args = parser.parse_args()

    # 1. Config & Log
    config = load_config(args.config)
    work_dir = config.get("work_dir", config["train"].get("log_dir", "./results"))
    os.makedirs(work_dir, exist_ok=True)
    
    if args.phase == 'train':
        log_file = os.path.join(work_dir, "train_log.txt")
        sys.stdout = Logger(log_file)
        print(f"[INFO] Log saving to {log_file}")

    # 2. Setup Device & Seed
    set_seed(config["train"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 3. Data Loader Helper
    def get_loader(feeder_name, feeder_args, batch_size, num_workers, shuffle):
        Feeder = import_class(feeder_name)
        dataset = Feeder(**feeder_args)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, drop_last=shuffle, pin_memory=True)

    # 4. Initialize Data Loaders
    if args.phase == 'train':
        print("[INFO] Loading Train & Val Data...")
        train_loader = get_loader(config['train_feeder'], config['train_feeder_args'], 
                                  config['train']['batch_size'], config['train']['num_workers'], True)
        val_loader = get_loader(config['val_feeder'], config['val_feeder_args'], 
                                config['train']['batch_size'], config['train']['num_workers'], False)
    else:
        print("[INFO] Loading Test Data...")
        test_loader = get_loader(config['test_feeder'], config['test_feeder_args'], 
                                 config['test']['test_batch_size'], config['train']['num_workers'], False)

    # 5. Build Model
    print("[INFO] Building Model...")
    Model = import_class(config['model'])
    model = Model(**config['model_args']).to(device)

    # [TRANSFER LEARNING LOGIC]
    if 'pretrained_path' in config and config['pretrained_path']:
        print(f"[INFO] Loading Transfer Weights from: {config['pretrained_path']}")
        model = load_transfer_weights(model, config['pretrained_path'], device)

    # [LOAD CHECKPOINT LOGIC]
    if args.weights:
        print(f"[INFO] Loading weights from {args.weights}")
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

    # [PHASE TEST]
    if args.phase == 'test':
        criterion = nn.CrossEntropyLoss()
        loss, top1, top5 = validate(model, criterion, test_loader, device, 1, 1, phase='TEST')
        print(f"\n" + "="*40)
        print(f" TEST RESULTS")
        print(f" Loss  : {loss:.4f}")
        print(f" Top-1 : {top1:.2f}%")
        print(f" Top-5 : {top5:.2f}%")
        print("="*40 + "\n")
        return

    # 6. Optimizer & Scheduler
    lbl_smooth = config.get('loss_args', {}).get('label_smoothing', 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=lbl_smooth)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["base_lr"],
                                  weight_decay=config["train"]["weight_decay"])
    
    total_epochs = config["train"]["num_epoch"]
    warmup_epochs = config["train"]["warm_up_epoch"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

    writer = SummaryWriter(work_dir)
    best_acc = 0.0
    use_mixup = config.get("use_mixup", True)

    # 7. TRAINING LOOP
    print("\n" + "="*50)
    print(f" START TRAINING ({total_epochs} Epochs)")
    print("="*50)

    for epoch in range(1, total_epochs + 1):
        # Warmup LR
        if epoch <= warmup_epochs:
            lr_scale = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = config["train"]["base_lr"] * lr_scale
        
        # Train 1 Epoch
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch, total_epochs, use_mixup
        )

        # Validation
        if epoch % config['train']['eval_interval'] == 0:
            val_loss, val_acc1, val_acc5 = validate(model, criterion, val_loader, device, epoch, total_epochs)
            
            # Save Best Model
            if val_acc1 > best_acc:
                best_acc = val_acc1
                save_checkpoint(model, optimizer, epoch, os.path.join(work_dir, "best_model.pth"))
                
                # [DISPLAY TOP-5 CLEARLY]
                print(f" -> NEW BEST MODEL SAVED!")
                print(f"    Top-1: {best_acc:.2f}%")
                print(f"    Top-5: {val_acc5:.2f}%")

            # Logging TensorBoard
            writer.add_scalars("Loss", {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars("Acc/Top1", {'train': train_acc1, 'val': val_acc1}, epoch)
            writer.add_scalars("Acc/Top5", {'train': train_acc5, 'val': val_acc5}, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        if epoch > warmup_epochs:
            scheduler.step()

        # Save Regular Checkpoint
        if epoch % config['train']['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(work_dir, f"checkpoint_epoch_{epoch}.pth"))

    print(f"\n[DONE] Training Finished. Best Validation Top-1: {best_acc:.2f}%")
    writer.close()

if __name__ == "__main__":
    main()