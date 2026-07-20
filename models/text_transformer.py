
# pip install transformers torch

import torch
from transformers import AutoTokenizer, AutoModel

model_name = "medicalai/ClinicalBERT"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")

# text encoder
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

model.eval()


