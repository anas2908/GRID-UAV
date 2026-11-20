import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from grid.actions import action_text, prompt_text
from grid.alignment import TEACHER_IMAGE_SIZE
from grid.model import preprocess_views

CAMERAS = ("frontcamera", "downcamera")
MAX_TEXT_TOKENS = 2048


def load_manifest(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def view_paths(data_root, sample):
    folder = Path(data_root) / sample["traj_rel_dir"]
    return [folder / camera / sample["img_name"] for camera in CAMERAS]


def teacher_views(images):
    views = [torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).float().div_(255) for image in images]
    views = F.interpolate(torch.stack(views), size=(TEACHER_IMAGE_SIZE, TEACHER_IMAGE_SIZE),
                          mode="bicubic", align_corners=False, antialias=True)
    return views.clamp_(0, 1).contiguous()


class DepthCache:
    def __init__(self, path, num_samples):
        self.array = np.load(path, mmap_mode="r")
        if self.array.shape[0] != num_samples or self.array.shape[1] != len(CAMERAS):
            raise ValueError(f"depth cache {path} has shape {self.array.shape}, expected [{num_samples},2,64,64]")

    def __getitem__(self, index):
        return np.array(self.array[index], dtype=np.float16)


class NavigationDataset(Dataset):
    def __init__(self, manifest, data_root, tokenizer, processor, depth_cache):
        self.samples = load_manifest(manifest)
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.image_processor = processor.image_processor
        self.depth = DepthCache(depth_cache, len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        images = []
        for path in view_paths(self.data_root, sample):
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        land = sample["is_last_step"] or sample["is_penultimate"]
        prompt = prompt_text(sample["instruction"])
        item = {
            "pixel_values": preprocess_views(images, self.image_processor),
            "text": prompt + action_text(sample["label"], land) + (self.tokenizer.eos_token or "</s>"),
            "prompt": prompt,
            "depth": torch.from_numpy(self.depth[index]),
            "teacher_views": teacher_views(images),
        }
        return item


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        tokens = self.tokenizer([f["text"] for f in features], return_tensors="pt", padding="longest",
                                truncation=True, max_length=MAX_TEXT_TOKENS)
        labels = tokens.input_ids.clone()
        for row, feature in enumerate(features):
            prompt = self.tokenizer(feature["prompt"], return_tensors="pt", add_special_tokens=True).input_ids[0]
            labels[row, :len(prompt)] = -100
        labels[tokens.attention_mask == 0] = -100
        batch = {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]).to(torch.bfloat16),
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "labels": labels,
        }
        depth = torch.stack([f["depth"] for f in features]).float()
        batch["depth"] = depth
        batch["depth_valid"] = torch.ones(depth.shape[:2], dtype=torch.bool)
        batch["teacher_views"] = torch.stack([f["teacher_views"] for f in features])
        return batch
