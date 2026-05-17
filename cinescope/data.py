import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


ATTRIBUTE_NAMES: List[str] = [
    "composition",
    "lighting",
    "color",
    "motion",
    "narrative",
]


class StoryboardDataset(Dataset):
    """JSONL-based storyboard dataset."""

    def __init__(self, data_root: str, ann_file: str, image_size: int = 224):
        self.data_root = Path(data_root)
        self.ann_file = Path(ann_file)
        self.items = []

        with self.ann_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        image_path = self.data_root / item["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        score = torch.tensor(float(item["score"]), dtype=torch.float32)
        attrs = [float(item["attributes"][name]) for name in ATTRIBUTE_NAMES]
        attrs = torch.tensor(attrs, dtype=torch.float32)
        return {"image": image, "score": score, "attributes": attrs, "path": str(image_path)}
