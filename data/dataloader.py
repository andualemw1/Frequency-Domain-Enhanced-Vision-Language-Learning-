
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoTokenizer

# Disable parallel tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define default ImageNet transformations for LeViT
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

class MedicalMultimodalDataset(Dataset):
    def __init__(self, csv_file, root_dir, tokenizer, max_length=128, transform=None):
        """
        Args:
            csv_file (str): Path to the cleaned_dataset.csv file.
            root_dir (str): Folder containing the resized image files.
            tokenizer (AutoTokenizer): Pre-trained tokenizer.
            max_length (int): Text truncation length.
            transform (callable, optional): Transform operations for images.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform if transform is not None else default_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Fetch text data safely
        report = str(row["org_caption"])
        tokens = self.tokenizer(
            report,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        text_data = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }

        # 2. Fetch the matched image file via image_id mapping
        image_id = str(row["image_id"])
        
        # Add an extension fallback if the CSV doesn't explicitly store it
        if not image_id.lower().endswith((".png", ".jpg", ".jpeg")):
            img_filename = f"{image_id}.png"
        else:
            img_filename = image_id
            
        img_path = os.path.join(self.root_dir, img_filename)
        
        # Double check alternative extensions if files are mixed format
        if not os.path.exists(img_path):
            for ext in [".jpg", ".jpeg", ".png"]:
                alt_path = os.path.join(self.root_dir, f"{image_id}{ext}")
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break

        # Load and convert image
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not locate matched image at: {img_path}")

        if self.transform is not None:
            image = self.transform(image)

        # Return a paired tuple matching your training loop signature
        return image, text_data


def get_multimodal_dataloader(csv_file, root_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    Helper function to instantly build the unified data stream.
    """
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    
    dataset = MedicalMultimodalDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        tokenizer=tokenizer
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True # Strongly recommended for contrastive loss to keep shapes even
    )
    
    return dataloader


