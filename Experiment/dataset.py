"""
PyTorch Dataset for compound-figure image-caption pairs.
Returns (pixel_values, input_ids, attention_mask) ready for CLIPModel.
"""

import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PanelDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, processor, max_text_len: int = 77):
        self.df         = pd.read_csv(csv_path).reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.processor  = processor
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.images_dir / row["image_filename"]).convert("RGB")
        text  = str(row["subcaption"])

        enc = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
        )
        return {
            "pixel_values":  enc["pixel_values"].squeeze(0),
            "input_ids":     enc["input_ids"].squeeze(0),
            "attention_mask":enc["attention_mask"].squeeze(0),
        }
