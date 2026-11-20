import inspect

import torch
from torch import nn
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from grid.geometry import TOKENS_PER_VIEW, depth_to_patch_xyz

NUM_VIEWS = 2
VISUAL_TOKENS = NUM_VIEWS * TOKENS_PER_VIEW
FEATURE_DIM = 2176
OPENVLA = "openvla/openvla-7b"
OPENVLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class Ego3DEmbedding(nn.Module):
    def __init__(self, in_channels=12, out_channels=FEATURE_DIM, n_freqs=8, scale_m=40.0):
        super().__init__()
        self.scale_m = scale_m
        self.register_buffer("freq_bands", 2 ** torch.linspace(0, n_freqs - 1, n_freqs), persistent=False)
        self.register_buffer("center", torch.zeros(in_channels), persistent=False)
        self.position_embedding_head = nn.Sequential(
            nn.Linear(in_channels * (2 * n_freqs + 1), out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter, gain=0.01)
        nn.init.zeros_(self.position_embedding_head[-1].weight)
        nn.init.zeros_(self.position_embedding_head[-1].bias)

    @torch.no_grad()
    def encode(self, xyz):
        xyz = (xyz.float() - self.center.float()) / self.scale_m
        phases = xyz.unsqueeze(-1) * self.freq_bands.float()
        encoded = torch.cat((xyz.unsqueeze(-1), torch.sin(phases), torch.cos(phases)), dim=-1)
        return encoded.reshape(*xyz.shape[:2], -1)

    def forward(self, xyz):
        encoded = self.encode(xyz)
        return self.position_embedding_head(encoded.to(self.position_embedding_head[0].weight.dtype))


class ViewProjector(nn.Module):
    def __init__(self, rgb_projector):
        super().__init__()
        self.rgb_path = rgb_projector
        self.ego3d = Ego3DEmbedding()
        self.camera_embedding = nn.Parameter(torch.zeros(NUM_VIEWS, FEATURE_DIM))
        self.depth = None
        self.depth_valid = None

    def forward(self, features):
        if self.depth is None:
            raise ValueError("GRID needs depth maps for every multimodal forward pass")
        xyz = depth_to_patch_xyz(self.depth, self.depth_valid)
        position = self.ego3d(xyz.to(features.device)).to(features.dtype)
        valid = self.depth_valid.to(features.device).repeat_interleave(TOKENS_PER_VIEW, dim=1)[..., None]
        camera = self.camera_embedding.repeat_interleave(TOKENS_PER_VIEW, dim=0).to(features.dtype)
        return self.rgb_path(features + position * valid + camera[None])


def install_views(model):
    projector = ViewProjector(model.projector)
    reference = next(model.projector.parameters())
    projector.to(device=reference.device, dtype=reference.dtype)
    model.projector = projector
    encode = model.vision_backbone.forward

    def two_view_forward(pixel_values):
        batch = pixel_values.shape[0]
        features = encode(pixel_values.reshape(batch * NUM_VIEWS, *pixel_values.shape[2:]))
        return features.reshape(batch, NUM_VIEWS * features.shape[1], features.shape[2])

    model.vision_backbone.forward = two_view_forward
    install_depth_inputs(model)
    return model


def install_depth_inputs(model):
    projector = model.projector
    forward = model.forward
    prepare = model.prepare_inputs_for_generation
    signature = inspect.signature(forward)

    def geometry_forward(*args, depth=None, depth_valid=None, **kwargs):
        bound = signature.bind_partial(*args, **kwargs).arguments
        multimodal = bound.get("pixel_values") is not None and bound.get("past_key_values") is None
        if multimodal and depth is None:
            raise ValueError("GRID needs depth maps for every multimodal forward pass")
        if multimodal and depth_valid is None:
            depth_valid = torch.ones(depth.shape[:2], dtype=torch.bool, device=depth.device)
        # PEFT's modules_to_save deep-copies the projector, so set depth on every live ViewProjector copy.
        projectors = [m for m in model.modules() if isinstance(m, ViewProjector)] or [projector]
        for p in projectors:
            p.depth, p.depth_valid = (depth, depth_valid) if multimodal else (None, None)
        try:
            return forward(*args, **kwargs)
        finally:
            for p in projectors:
                p.depth = p.depth_valid = None

    def geometry_prepare(input_ids=None, past_key_values=None, inputs_embeds=None, pixel_values=None,
                         attention_mask=None, depth=None, depth_valid=None, **kwargs):
        prepared = prepare(input_ids=input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                           pixel_values=pixel_values, attention_mask=attention_mask, **kwargs)
        prepared.update(depth=depth, depth_valid=depth_valid)
        return prepared

    model.forward = geometry_forward
    model.prepare_inputs_for_generation = geometry_prepare
    return model


def preprocess_views(images, image_processor):
    tensors = [image_processor(images=image, return_tensors="pt")["pixel_values"][0] for image in images]
    return torch.stack(tensors)


def build_policy(model_path, device_map=None, revision=None):
    options = {"trust_remote_code": True, "revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_path, **options)
    processor = AutoProcessor.from_pretrained(model_path, **options)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        **options,
    )
    model.resize_token_embeddings(len(tokenizer))
    install_views(model)
    return model, tokenizer, processor
