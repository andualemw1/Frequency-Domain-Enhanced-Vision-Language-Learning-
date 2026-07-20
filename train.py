
import torch
import torch.nn.functional as F

def compute_contrastive_loss(similarity, device):
    """
    Computes bidirectional cross-entropy loss over a similarity matrix.
    Args:
        similarity (Tensor): Matrix of shapes [Batch_Size, Batch_Size] representing 
                             scaled dot-product similarities.
        device (str/torch.device): Target hardware ('cuda' or 'cpu').
    """
    batch_size = similarity.shape[0]
    # In-batch targets: diagonal positions are matching image-text pairs
    labels = torch.arange(batch_size, device=device)
    
    # Image-to-Text loss (rows evaluate across columns)
    loss_i = F.cross_entropy(similarity, labels)
    
    # Text-to-Image loss (columns evaluate across rows)
    loss_t = F.cross_entropy(similarity.T, labels)
    
    # Symmetric balance
    return (loss_i + loss_t) / 2


def train_one_epoch(model, img_dataloader, text_dataloader, optimizer, scheduler, device, log_interval=10):
    """
    Runs a single forward, backward, and optimization cycle across the aligned datasets.
    """
    model.train()
    running_loss = 0.0
    steps = 0
    
    # Use python's zip function to iterate jointly across your loaders.
    # Note: Assumes img_loader and text_loader yield identically ordered/shuffled pair batches.
    for batch_idx, (img_batch, text_batch) in enumerate(zip(img_dataloader, text_dataloader)):
        
        # 1. Clean device mapping
        img_batch = img_batch.to(device)
        input_ids = text_batch["input_ids"].to(device)
        attention_mask = text_batch["attention_mask"].to(device)
        
        # 2. Reset gradients
        optimizer.zero_grad()
        
        # 3. Model inference through our MultimodalCLIP wrapper
        similarity = model(img_batch, input_ids, attention_mask)
        
        # 4. Compute loss
        loss = compute_contrastive_loss(similarity, device)
        
        # 5. Backpropagation
        loss.backward()
        
        # Transformer safety net: Prevents sudden exploding gradients from blowing up weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 6. Weight adjustments
        optimizer.step()
        scheduler.step()
        
        # Logging aggregation
        running_loss += loss.item()
        steps += 1
        
        if batch_idx % log_interval == 0:
            current_temp = model.logit_scale.exp().item()
            print(f"  [Batch {batch_idx:03d}] Current Loss: {loss.item():.4f} | Scale Temp: {current_temp:.2f}")
            
    epoch_avg_loss = running_loss / max(steps, 1)
    return epoch_avg_loss


@torch.no_grad()
def validate_epoch(model, img_dataloader, text_dataloader, device):
    """
    Evaluates loss over a separate validation split without computing gradients.
    """
    model.eval()
    running_loss = 0.0
    steps = 0
    
    for img_batch, text_batch in zip(img_dataloader, text_dataloader):
        img_batch = img_batch.to(device)
        input_ids = text_batch["input_ids"].to(device)
        attention_mask = text_batch["attention_mask"].to(device)
        
        similarity = model(img_batch, input_ids, attention_mask)
        loss = compute_contrastive_loss(similarity, device)
        
        running_loss += loss.item()
        steps += 1
        
    val_avg_loss = running_loss / max(steps, 1)
    return val_avg_loss


