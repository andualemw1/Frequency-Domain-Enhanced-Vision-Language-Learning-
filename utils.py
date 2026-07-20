
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """
    Guarantees deterministic behavior across runs by locking all underlying random frameworks.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Random seed fixed globally at: {seed}")


class ProgressTracker:
    """
    Tracks, logs, and automatically exports visual convergence curves for training diagnostics.
    """
    def __init__(self, save_dir="./checkpoints"):
        self.save_dir = save_dir
        self.history = {"train_loss": [], "learning_rates": [], "temperatures": []}
        os.makedirs(save_dir, exist_ok=True)

    def update(self, avg_loss, current_lr, current_temp):
        self.history["train_loss"].append(avg_loss)
        self.history["learning_rates"].append(current_lr)
        self.history["temperatures"].append(current_temp)
        self._plot_metrics()

    def _plot_metrics(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        plt.figure(figsize=(12, 4))
        
        # Subplot 1: Convergence Curve
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], marker='o', color='crimson', label='Train Loss')
        plt.title('Contrastive Convergence Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Subplot 2: Temperature Scaling Parameter Performance
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["temperatures"], marker='s', color='darkblue', label='Learnable Temp')
        plt.title('Logit Temperature Profile')
        plt.xlabel('Epoch')
        plt.ylabel('Softmax Scale Factor')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_metrics.png"), dpi=150)
        plt.close()


def verify_batch_alignment(img_batch, text_batch):
    """
    Safety net to verify that vision features and text arrays map identically before backpropagation.
    Catches shape exceptions gracefully caused by truncated tail batches.
    """
    img_size = img_batch.shape[0]
    text_size = text_batch["input_ids"].shape[0]
    
    if img_size != text_size:
        raise ValueError(
            f"Dimension Mismatch Detected! Image Batch Size ({img_size}) does not match "
            f"Text Batch Size ({text_size}). Check your DataLoader configuration settings."
        )
    return True

