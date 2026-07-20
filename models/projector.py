
import torch
import torch.nn as nn

class MultimodalProjector(nn.Module):
    def __init__(self, image_dim: int = 512, text_dim: int = 768, shared_dim: int = 768):
        super().__init__()
        
        # Image Projector: Maps [B, image_dim] -> [B, shared_dim]
        self.image_projector = nn.Sequential(
            nn.Linear(image_dim, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim) # Stabilizes contrastive training
        )
        
        # Text Projector: Maps [B, text_dim] -> [B, shared_dim]
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim)
        )

    def pool_text_features(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs mean pooling over valid (non-padding) tokens.
        Inputs:
            last_hidden_state: [B, seq_len, text_dim]
            attention_mask: [B, seq_len]
        Output:
            text_embedding: [B, text_dim]
        """
        # Expand mask to [B, seq_len, 1] to match feature dimensions
        mask_expanded = attention_mask.unsqueeze(-1).float()
        
        # Sum up features across the token dimension (ignoring padding)
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        
        # Clamp the mask sum to prevent division by zero on completely empty sequences
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def forward(self, image_features: torch.Tensor, text_hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """
        Inputs:
            image_features: [B, image_dim] (From LeViT)
            text_hidden_states: [B, seq_len, text_dim] (From ClinicalBERT)
            attention_mask: [B, seq_len]
        Outputs:
            projected_image: [B, shared_dim] (L2 Normalized)
            projected_text: [B, shared_dim] (L2 Normalized)
        """
        # 1. Pool the text tokens into a single sequence vector
        pooled_text = self.pool_text_features(text_hidden_states, attention_mask)
        
        # 2. Project both into the shared latent space
        projected_image = self.image_projector(image_features)
        projected_text = self.text_projector(pooled_text)
        
        # 3. L2 Normalize the outputs (Crucial step for CLIP-style cosine similarity loss)
        projected_image = nn.functional.normalize(projected_image, p=2, dim=-1)
        projected_text = nn.functional.normalize(projected_text, p=2, dim=-1)
        
        return projected_image, projected_text
    
