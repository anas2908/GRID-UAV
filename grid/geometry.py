import torch
import torch.nn.functional as F

IMAGE_SIZE = 256
DEPTH_SIZE = 64
PATCH_GRID = 16
TOKENS_PER_VIEW = PATCH_GRID * PATCH_GRID
SAMPLES_PER_PATCH = 4
MAX_DEPTH_M = 80.0


def patch_sample_pixels(device=None):
    axis = torch.arange(2 * PATCH_GRID, device=device, dtype=torch.float32) * 8 + 4
    vv, uu = torch.meshgrid(axis, axis, indexing="ij")
    uv = torch.stack((uu, vv), -1).reshape(PATCH_GRID, 2, PATCH_GRID, 2, 2)
    return uv.permute(0, 2, 1, 3, 4).reshape(TOKENS_PER_VIEW, SAMPLES_PER_PATCH, 2)


def camera_to_body(points, view):
    x, y, z = points.unbind(-1)
    if view == 0:
        return torch.stack((z + 1, x, y), -1)
    if view == 1:
        return torch.stack((-y, x, z), -1)
    raise ValueError("view must be 0 (front) or 1 (down)")


@torch.no_grad()
def depth_to_patch_xyz(depth, valid):
    batch, views = depth.shape[:2]
    valid = valid.to(depth.device)
    safe = torch.where(valid[..., None, None], depth.float(), 0.0)
    ranges = F.avg_pool2d(safe.reshape(batch * views, 1, DEPTH_SIZE, DEPTH_SIZE), 2)
    ranges = ranges.reshape(batch, views, PATCH_GRID, 2, PATCH_GRID, 2)
    ranges = ranges.permute(0, 1, 2, 4, 3, 5).reshape(batch, views, TOKENS_PER_VIEW, SAMPLES_PER_PATCH)
    uv = patch_sample_pixels(device=depth.device)
    half = IMAGE_SIZE / 2
    rays = torch.cat(((uv - half) / half, torch.ones_like(uv[..., :1])), -1)
    points = ranges[..., None] * F.normalize(rays, dim=-1)[None, None]
    body = torch.stack([camera_to_body(points[:, i], i) for i in range(views)], 1)
    body = body * valid[..., None, None, None]
    return body.reshape(batch, views * TOKENS_PER_VIEW, 3 * SAMPLES_PER_PATCH)
