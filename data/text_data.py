
import pandas as pd
from torch.utils.data import Dataset

class TextDataset(Dataset):

    def __init__(self, csv_file, tokenizer, max_length=128):

        self.df = pd.read_csv(csv_file)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        report = str(self.df.iloc[idx]["org_caption"])

        tokens = self.tokenizer(
            report,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "report": report,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }
    
