"""
Panel detection dataset.
Reads images + YOLO labels (class xc yc w h, normalised).
Supports batch_size > 1 via processor-level padding in collate_fn.

Folder layout:
    root/
    ├── images/   ← .jpg / .png
    └── labels/   ← one .txt per image  "class xc yc w h"
"""

import os
import random
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image


class PanelDataset(Dataset):
    """
    __getitem__ processes each image individually (resize + normalise).
    collate_fn pads tensors to the max size in the batch and builds pixel_mask.
    """

    def __init__(self, root: str, processor, augment: bool = False,
                 max_samples: Optional[int] = None):
        self.images_dir = os.path.join(root, "images")
        self.labels_dir = os.path.join(root, "labels")
        self.processor  = processor
        self.augment    = augment
        self._jitter    = T.ColorJitter(brightness=0.3, contrast=0.3,
                                        saturation=0.2, hue=0.05)

        stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if max_samples:
            stems = stems[:max_samples]
        self.stems = stems

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem     = self.stems[idx]
        img_path = self._find_image(stem)
        lbl_path = os.path.join(self.labels_dir, stem + ".txt")
        if img_path is None or not os.path.exists(lbl_path):
            return None

        pil            = Image.open(img_path).convert("RGB")
        boxes, classes = self._read_labels(lbl_path)
        if not boxes:
            return None

        boxes   = torch.tensor(boxes,   dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)

        if self.augment:
            if random.random() < 0.5:
                pil = TF.hflip(pil)
                boxes[:, 0] = 1.0 - boxes[:, 0]
            pil = self._jitter(pil)

        # process individually — no padding here; collate_fn pads the batch
        encoding     = self.processor(images=pil, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)   # (3, H, W)

        return {
            "pixel_values": pixel_values,
            "labels":       {"class_labels": classes, "boxes": boxes},
            "stem":         stem,
        }

    def _find_image(self, stem: str) -> Optional[str]:
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(self.images_dir, stem + ext)
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def _read_labels(path: str) -> Tuple[List[List[float]], List[int]]:
        boxes, classes = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                boxes.append([
                    max(0.0, min(1.0, xc)),
                    max(0.0, min(1.0, yc)),
                    max(0.01, min(1.0, bw)),
                    max(0.01, min(1.0, bh)),
                ])
                classes.append(cls)
        return boxes, classes


def collate_fn(batch):
    """
    1. Drops None samples.
    2. Pads pixel_values to the max H×W in the batch (bottom-right padding).
    3. Builds pixel_mask  : 1 = real pixel, 0 = padding.
    4. Returns batched tensors + list of label dicts.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    pixel_values_list = [b["pixel_values"] for b in batch]
    labels            = [b["labels"]       for b in batch]

    max_h = max(pv.shape[1] for pv in pixel_values_list)
    max_w = max(pv.shape[2] for pv in pixel_values_list)

    padded_pixels = []
    pixel_masks   = []
    for pv in pixel_values_list:
        _, h, w  = pv.shape
        padded   = F.pad(pv, (0, max_w - w, 0, max_h - h))
        mask     = torch.zeros(max_h, max_w, dtype=torch.long)
        mask[:h, :w] = 1
        padded_pixels.append(padded)
        pixel_masks.append(mask)

    return {
        "pixel_values": torch.stack(padded_pixels),
        "pixel_mask":   torch.stack(pixel_masks),
        "labels":       labels,
    }