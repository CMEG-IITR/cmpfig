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
from PIL import Image, ImageFilter


class PanelDataset(Dataset):

    def __init__(self, root: str, processor, augment: bool = False,
                 max_samples: Optional[int] = None):
        self.images_dir = os.path.join(root, "images")
        self.labels_dir = os.path.join(root, "labels")
        self.processor  = processor
        self.augment    = augment

        # stronger color jitter than before
        self._jitter = T.ColorJitter(
            brightness=0.5,   # was 0.3
            contrast=0.5,     # was 0.3
            saturation=0.4,   # was 0.2
            hue=0.1,          # was 0.05
        )

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
            pil, boxes, classes = self._augment(pil, boxes, classes)

        encoding     = self.processor(images=pil, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels":       {"class_labels": classes, "boxes": boxes},
            "stem":         stem,
        }

    def _augment(self, pil: Image.Image, boxes: torch.Tensor, classes: torch.Tensor):
        """All augmentations that keep YOLO boxes valid."""

        # 1. Horizontal flip (boxes: xc flips)
        if random.random() < 0.5:
            pil = TF.hflip(pil)
            boxes[:, 0] = 1.0 - boxes[:, 0]

        # 2. Vertical flip — panels can appear upside down in scans
        if random.random() < 0.2:
            pil = TF.vflip(pil)
            boxes[:, 1] = 1.0 - boxes[:, 1]

        # 3. Random crop + resize (zoom in effect)
        #    Crop between 70–100% of the image, then resize back
        if random.random() < 0.5:
            pil, boxes, classes = self._random_crop(pil, boxes, classes, min_scale=0.7)

        # 4. Color jitter
        if random.random() < 0.8:
            pil = self._jitter(pil)

        # 5. Random grayscale (simulates B&W figures)
        if random.random() < 0.2:
            pil = TF.to_grayscale(pil, num_output_channels=3)

        # 6. Gaussian blur (simulates low-res scans)
        if random.random() < 0.2:
            radius = random.uniform(0.5, 1.5)
            pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))

        # 7. Random rotation (small — panels are mostly upright)
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            pil   = TF.rotate(pil, angle, fill=0)

        return pil, boxes, classes

    @staticmethod
    def _random_crop(pil: Image.Image, boxes: torch.Tensor, classes: torch.Tensor,
                     min_scale: float = 0.7):
        """
        Crop a random sub-region (min_scale to 1.0 of W and H),
        keep only boxes whose centre falls inside the crop,
        remap box coordinates to the new frame, resize back.
        classes is kept in sync with boxes.
        """
        W, H = pil.size
        scale_w = random.uniform(min_scale, 1.0)
        scale_h = random.uniform(min_scale, 1.0)
        crop_w  = int(W * scale_w)
        crop_h  = int(H * scale_h)

        left = random.randint(0, W - crop_w)
        top  = random.randint(0, H - crop_h)

        # Normalised crop boundaries
        x_min_n = left / W
        y_min_n = top  / H
        x_max_n = (left + crop_w) / W
        y_max_n = (top  + crop_h) / H

        # Keep boxes whose centre is inside the crop
        cx, cy = boxes[:, 0], boxes[:, 1]
        keep   = (cx >= x_min_n) & (cx <= x_max_n) & \
                 (cy >= y_min_n) & (cy <= y_max_n)

        if keep.sum() == 0:
            return pil, boxes, classes

        boxes   = boxes[keep]
        classes = classes[keep]

        # Remap to new coordinate frame
        boxes[:, 0] = (boxes[:, 0] - x_min_n) / (x_max_n - x_min_n)
        boxes[:, 1] = (boxes[:, 1] - y_min_n) / (y_max_n - y_min_n)
        boxes[:, 2] = boxes[:, 2] / scale_w
        boxes[:, 3] = boxes[:, 3] / scale_h

        boxes = boxes.clamp(0.0, 1.0)
        boxes[:, 2] = boxes[:, 2].clamp(min=0.01)
        boxes[:, 3] = boxes[:, 3].clamp(min=0.01)

        pil = TF.crop(pil, top, left, crop_h, crop_w)
        pil = pil.resize((W, H), Image.BILINEAR)

        return pil, boxes, classes

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