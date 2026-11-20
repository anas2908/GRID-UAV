from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image

from grid.actions import navigation_prompt, object_description, parse_action
from grid.depth import AerialDepth
from grid.model import OPENVLA, OPENVLA_REVISION, build_policy, preprocess_views


def to_pil(image, bgr=False):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if bgr:
        array = array[..., ::-1]
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


class GRIDPolicy:
    def __init__(self, weights_dir, base_model=OPENVLA, revision=OPENVLA_REVISION, device="cuda", max_new_tokens=20):
        weights_dir = Path(weights_dir)
        self.device = torch.device(device)
        model, self.tokenizer, processor = build_policy(base_model, revision=revision)
        self.image_processor = processor.image_processor
        self.model = PeftModel.from_pretrained(model, str(weights_dir / "policy"))
        self.model.to(self.device).eval()
        self.depth = AerialDepth(weights_dir / "depth" / "aerial_depth.pt", device=self.device)
        self.max_new_tokens = max_new_tokens

    def prepare(self, front, down, prompt):
        views = [to_pil(front), to_pil(down)]
        for view in views:
            if view.size != (256, 256):
                raise ValueError("GRID expects 256x256 front and down images")
        inputs = dict(self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=False))
        inputs["pixel_values"] = preprocess_views(views, self.image_processor).unsqueeze(0).to(self.model.dtype)
        depth = self.depth.predict(views)
        inputs["depth"] = torch.from_numpy(depth.astype(np.float32))[None]
        inputs["depth_valid"] = torch.ones((1, 2), dtype=torch.bool)
        return {key: value.to(self.device) for key, value in inputs.items()}

    @torch.inference_mode()
    def generate(self, inputs):
        ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  eos_token_id=[self.tokenizer.eos_token_id])
        return self.tokenizer.decode(ids[0], skip_special_tokens=False)

    def act(self, front, down, instruction, hint=""):
        prompt = navigation_prompt(hint, object_description(instruction))
        text = self.generate(self.prepare(front, down, prompt))
        action = parse_action(text)
        action["text"] = text.split("Action:")[-1].replace("</s>", "").replace("<pad>", "").strip()
        return action
