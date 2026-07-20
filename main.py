
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import utils
from train import train_one_epoch  # Using the standardized external module

# Corrected Data and Model Imports
from data.data_loader import get_multimodal_dataloader
from models.levit import LeViT_256
from transformers import AutoModel

from models.projector import ImageProjector, TextProjector
from models.multimodal import MultimodalCLIP

def main():
    # --- Configurable Hyperparameters ---
    EPOCHS = 10
    LR = 5e-5
    WEIGHT_DECAY = 1e-2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "/home/andualemwelabo.tulu/multi-modal/medical_vlm/checkpoints/LeViT-256-13b5763e.pth"
    SAVE_DIR = "./checkpoints"
    
    # Initialize seeds and metrics tracking
    utils.set_seed(42)
    os.makedirs(SAVE_DIR, exist_ok=True)
    tracker = utils.ProgressTracker(save_dir=SAVE_DIR)

    print(f"Using device: {DEVICE}")

    # --- 1. Model Initialization ---
    text_encoder = AutoModel.from_pretrained("medicalai/ClinicalBERT")
    img_encoder = LeViT_256()

    # Load pre-trained vision weights safely
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading LeViT weight from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        img_encoder.load_state_dict(checkpoint["model"])
    else:
        print("Warning: Vision checkpoint not found! Initializing randomly.")

    img_projector = ImageProjector()
    text_projector = TextProjector()

    # Construct the complete network layout
    model = MultimodalCLIP(img_encoder, text_encoder, img_projector, text_projector)
    model.to(DEVICE)

    # --- 2. Optimization Config ---
    # Encoders get a lower learning rate to preserve features, fresh projectors get the full scale
    optimizer_grouped_parameters = [
        {"params": model.img_encoder.parameters(), "lr": LR * 0.1},  
        {"params": model.text_encoder.parameters(), "lr": LR * 0.1}, 
        {"params": model.img_projector.parameters(), "lr": LR},       
        {"params": model.text_projector.parameters(), "lr": LR},      
        {"params": [model.logit_scale], "lr": LR}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=WEIGHT_DECAY)
    
    # --- 3. Unified Dataloader Initialization ---
    CSV_PATH = "/home/andualemwelabo.tulu/multi-modal/medical_vlm/data/iu-chest-x-rays-cleaned/versions/1/cleaned_dataset.csv"
    IMG_DIR = "/home/andualemwelabo.tulu/multi-modal/medical_vlm/data/iu-chest-x-rays-cleaned/versions/1/resized_images/256"

    # Note: Keep batch size at 8 if constrained by memory, but bump to 16/32 if VRAM allows
    train_dataloader = get_multimodal_dataloader(
        csv_file=CSV_PATH,
        root_dir=IMG_DIR,
        batch_size=8, 
        shuffle=True,
        num_workers=4
    )
    
    # Calculate total steps for scheduler mapping
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # --- 4. Main Training Loop ---
    print("Beginning multi-modal contrastive training...")
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        # Execute training pass using the external unified train engine
        avg_train_loss = train_one_epoch(
            model=model, 
            train_dataloader=train_dataloader, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            device=DEVICE,
            log_interval=10
        )
        print(f"Epoch {epoch} complete | Avg Train Loss: {avg_train_loss:.4f}")

        # Capture metrics data and update convergence chart
        current_lr = optimizer.param_groups[2]["lr"]  
        current_temp = model.logit_scale.exp().item()
        tracker.update(avg_train_loss, current_lr, current_temp)
        
        # Save structural weights metadata block
        checkpoint_meta = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }
        
        # Save chronological checkpoint
        torch.save(checkpoint_meta, os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}.pth"))
        
        # Track and update the absolute best weights configuration
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(checkpoint_meta, os.path.join(SAVE_DIR, "best_multimodal_model.pth"))
            print(" New personal best! Model saved successfully.")

if __name__ == "__main__":
    main()

