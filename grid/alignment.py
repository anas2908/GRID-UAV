import torch
from torch import nn
from torch.nn import functional as F

TEACHER_IMAGE_SIZE = 518
TEACHER_PATCH_SIZE = 14
TEACHER_DIM = 2048
STUDENT_DIM = 4096
TOKEN_GRID = (16, 16)
VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


class AlignmentHead(nn.Module):
    def __init__(self, student_dim=STUDENT_DIM, teacher_dim=TEACHER_DIM):
        super().__init__()
        self.vlm_norm = nn.LayerNorm(student_dim, eps=1e-5)
        self.fc1 = nn.Linear(student_dim, teacher_dim)
        self.fc2 = nn.Linear(teacher_dim, teacher_dim)
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            x = F.layer_norm(features.float(), self.vlm_norm.normalized_shape,
                             self.vlm_norm.weight.float(), self.vlm_norm.bias.float(), self.vlm_norm.eps)
            x = F.linear(x, self.fc1.weight.float(), self.fc1.bias.float())
            return F.linear(F.gelu(x), self.fc2.weight.float(), self.fc2.bias.float())


class VGGTTeacher(nn.Module):
    def __init__(self, checkpoint=None, device="cuda", uv_scale=0.1):
        super().__init__()
        from vggt.heads.utils import create_uv_grid, position_grid_to_embed
        from vggt.models.aggregator import Aggregator

        self.aggregator = Aggregator(img_size=TEACHER_IMAGE_SIZE, patch_size=TEACHER_PATCH_SIZE,
                                     embed_dim=1024, cached_layer_indices=(23,))
        if checkpoint is None:
            state = torch.hub.load_state_dict_from_url(VGGT_URL, map_location="cpu")
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = {k[len("aggregator."):]: v for k, v in state.items() if k.startswith("aggregator.")}
        self.aggregator.load_state_dict(state, strict=True)
        device = torch.device(device)
        self.aggregator.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        self.aggregator.requires_grad_(False).eval()
        side = TEACHER_IMAGE_SIZE // TEACHER_PATCH_SIZE
        grid = create_uv_grid(side, side, aspect_ratio=1.0, dtype=torch.float32, device=device)
        self.register_buffer("uv", position_grid_to_embed(grid, TEACHER_DIM), persistent=False)
        self.uv_scale = uv_scale

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, views):
        targets = []
        for sample in views.split(1):
            with torch.autocast(device_type=views.device.type, dtype=torch.bfloat16,
                                enabled=views.device.type == "cuda"):
                layers, patch_start = self.aggregator(sample)
            patches = layers[-1][:, :, patch_start:].float()
            batch, num_views, _, dim = patches.shape
            side = TEACHER_IMAGE_SIZE // TEACHER_PATCH_SIZE
            grids = patches.reshape(batch * num_views, side, side, dim).permute(0, 3, 1, 2)
            grids = grids + self.uv_scale * self.uv.to(grids).permute(2, 0, 1)[None]
            pooled = F.interpolate(grids, size=TOKEN_GRID, mode="bilinear", align_corners=True)
            targets.append(pooled.permute(0, 2, 3, 1).reshape(batch, -1, dim))
        return torch.cat(targets).detach()


def visual_hidden_states(outputs, layer, num_tokens):
    return outputs.hidden_states[layer][:, 1:num_tokens + 1, :]


def alignment_loss(student, teacher, eps=1e-6):
    with torch.autocast(device_type=student.device.type, enabled=False):
        student = F.normalize(student.float(), dim=-1, eps=eps)
        teacher = F.normalize(teacher.detach().float(), dim=-1, eps=eps)
        return (1.0 - (student * teacher).sum(dim=-1)).mean()
