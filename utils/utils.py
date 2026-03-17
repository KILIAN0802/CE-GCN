import os
import yaml
import random
import numpy as np
import torch

# =======================================================
#  SEED — REPRODUCIBILITY
# =======================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Fix cuDNN behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =======================================================
#  DIRECTORY UTILS
# =======================================================
def ensure_dir(path):
    """Create directory if not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# =======================================================
#  YAML READER
# =======================================================
def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =======================================================
#  NUMPY SOFTMAX
# =======================================================
def softmax_numpy(x):
    """Softmax for numpy logits."""
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# =======================================================
#  METRICS
# =======================================================
class AverageMeter:
    """Tracks average, sum, count of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Compute Top-k accuracy."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t().contiguous()

    correct = pred.eq(target.view(1, -1))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        acc = correct_k * (100.0 / batch_size)
        results.append(acc)

    return results


# =======================================================
#  CHECKPOINT SAVE / LOAD
# =======================================================
def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model checkpoint (.pth)."""
    ensure_dir(os.path.dirname(save_path))

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(state, save_path)
    print(f"[OK] Saved checkpoint → {save_path}")


def load_checkpoint(model, ckpt_path, optimizer=None, device="cpu"):
    """Load checkpoint into model."""
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"], strict=True)

    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    return ckpt.get("epoch", 0)


# =======================================================
#  COUNT PARAMETERS
# =======================================================
def count_parameters(model):
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =======================================================
#  SAFE PRINT (multiprocessing)
# =======================================================
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs, flush=True)
    except Exception:
        pass
