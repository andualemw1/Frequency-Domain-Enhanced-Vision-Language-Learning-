
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel

# Your structural imports
from levit import LeViT_256
from projector import ImageProjector, TextProjector
from models.multimodal_cli import MultimodalCLIP
from train import train_one_epoch

def run_sanity_check():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running check on device: {DEVICE}")

    # 1. Initialize a lightweight setup
    text_encoder = AutoModel.from_pretrained("medicalai/ClinicalBERT")
    img_encoder = LeViT_256()
    img_projector = ImageProjector()
    text_projector = TextProjector()

    model = MultimodalCLIP(img_encoder, text_encoder, img_projector, text_projector)
    model.to(DEVICE)

    # 2. Setup Optimizer with a high LR so we see rapid updates
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=5)

    # 3. Create a tiny Mock Dataset (Batch Size = 4, Sequence Length = 128)
    # This replaces your real dataloaders for a 5-second trial
    mock_images = [torch.randn(4, 3, 224, 224) for _ in range(3)]
    mock_texts = [
        {
            "input_ids": torch.randint(0, 2000, (4, 128)),
            "attention_mask": torch.ones(4, 128, dtype=torch.long)
        }
        for _ in range(3)
    ]

    print("\n--- Starting Sanity Check Loops ---")
    
    # Run for 2 quick epochs just to verify gradients work
    for epoch in range(1, 3):
        model.train()
        print(f"\nChecking Epoch {epoch}...")
        
        # Manually loop over our tiny mock lists instead of full dataloaders
        for batch_idx, (img_batch, text_batch) in enumerate(zip(mock_images, mock_texts)):
            img_batch = img_batch.to(DEVICE)
            input_ids = text_batch["input_ids"].to(DEVICE)
            attention_mask = text_batch["attention_mask"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Pass
            similarity = model(img_batch, input_ids, attention_mask)
            
            # Compute Loss manually for the check
            batch_size = similarity.shape[0]
            labels = torch.arange(batch_size, device=DEVICE)
            loss_i = torch.nn.functional.cross_entropy(similarity, labels)
            loss_t = torch.nn.functional.cross_entropy(similarity.T, labels)
            loss = (loss_i + loss_t) / 2
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            print(f"  Batch {batch_idx} -> Loss: {loss.item():.4f}")
            
    print("\n Success: Forward and backward optimization passes completed without errors!")

if __name__ == "__main__":
    run_sanity_check()

