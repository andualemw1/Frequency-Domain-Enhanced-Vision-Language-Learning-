
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from levit import LeViT_256
from projector import ImageProjector, TextProjector
from models.multimodal_cli import MultimodalCLIP

@torch.no_grad()
def generate_report_from_candidates(image_tensor, candidate_sentences, model, tokenizer, device, top_k=3):
    """
    Generates a report by selecting the top-K text fragments that match the target image.
    """
    model.eval()
    
    # 1. Process and project the image embedding
    image_tensor = image_tensor.to(device)
    # Handle single image batching [C, H, W] -> [1, C, H, W]
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
        
    img_feats = model.img_encoder.forward_features(image_tensor)
    img_emb = model.img_projector(img_feats)
    img_emb = F.normalize(img_emb, dim=-1) # [1, Dim]

    # 2. Tokenize and encode all candidate sentences
    encoded_inputs = tokenizer(
        candidate_sentences, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    ).to(device)
    
    text_outputs = model.text_encoder(
        input_ids=encoded_inputs["input_ids"], 
        attention_mask=encoded_inputs["attention_mask"]
    )
    
    # Masked mean pooling
    expanded_mask = encoded_inputs["attention_mask"].unsqueeze(-1)
    text_feats = (text_outputs.last_hidden_state * expanded_mask).sum(dim=1)
    text_feats = text_feats / expanded_mask.sum(dim=1)
    
    text_embs = model.text_projector(text_feats)
    text_embs = F.normalize(text_embs, dim=-1) # [Num_Candidates, Dim]

    # 3. Compute cosine similarity scores
    # We omit logit_scale here since we just want raw relative ranking distributions
    scores = (img_emb @ text_embs.T).squeeze(0) # [Num_Candidates]
    probabilities = F.softmax(scores * 100, dim=-1) # Scale up slightly for contrast

    # 4. Rank and gather findings
    top_scores, top_indices = torch.topk(probabilities, k=min(top_k, len(candidate_sentences)))
    
    # Construct final summary report
    constructed_report = []
    print("\n--- Model Inference Rankings ---")
    for i in range(len(top_indices)):
        idx = top_indices[i].item()
        prob = top_scores[i].item()
        sentence = candidate_sentences[idx]
        print(f"Confidence: {prob*100:.2f}% | Finding: {sentence}")
        
        # If the confidence is high enough, add it to the report
        if prob > 0.15: 
            constructed_report.append(sentence)
            
    return " ".join(constructed_report)


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_FILE = "./checkpoints/best_multimodal_model.pth"
    
    # Initialize framework components
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    text_encoder = AutoModel.from_pretrained("medicalai/ClinicalBERT")
    img_encoder = LeViT_256()
    img_projector = ImageProjector()
    text_projector = TextProjector()
    
    # Construct wrapper architecture and load weights
    model = MultimodalCLIP(img_encoder, text_encoder, img_projector, text_projector)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    print("Multi-modal model weights loaded successfully for inference.")

    # --- Setup Clinical Findings Knowledge Base ---
    # In production, this would be a list of hundreds of standard sentences from old reports.
    clinical_knowledge_base = [
        "The lungs are clear without focal consolidation, effusion, or pneumothorax.",
        "Cardiomegaly is present with mild enlargement of the cardiac silhouette.",
        "Severe right-sided pleural effusion is noted with associated compressive atelectasis.",
        "Acute displaced fracture identified along the posterior aspect of the lateral rib.",
        "Degenerative changes and osteophyte formations are visible along the thoracic spine.",
        "A suspicious focal opacity is noted within the right upper lobe, recommending follow-up CT."
    ]

    # Dummy tensor simulating a single incoming validation image [Batch=1, Channels=3, H=224, W=224]
    mock_image = torch.randn(3, 224, 224)

    # Execute inference report selection
    final_report = generate_report_from_candidates(
        mock_image, clinical_knowledge_base, model, tokenizer, DEVICE, top_k=2
    )
    
    print("\n================ GENERATED REPORT ================")
    print(final_report)
    print("==================================================")

if __name__ == "__main__":
    main()
