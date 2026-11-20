import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from grid.depth import build_depth_model, load_depth_state, preprocess_view

CAMERAS = ("frontcamera", "downcamera")
SPLIT_SALT = "grid-depth-split-v1"
FLIP_SALT = "grid-depth-flip-v1"
DEPTH_RANGE_M = 100.0
MAX_DEPTH_M = 80.0
CAP_CODE = 204
OUTPUT_SIZE = 256
VKITTI_URL = ("https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/"
              "070e97e4b80e317ec1d03c19927304f4091a180b/depth_anything_v2_metric_vkitti_vitl.pth")


def trajectory_split(samples, calibration=0.10, test=0.10):
    by_scene = defaultdict(set)
    for sample in samples:
        by_scene[sample["traj_rel_dir"].split("/", 1)[0]].add(sample["traj_rel_dir"])
    role = {}
    for scene in sorted(by_scene):
        ranked = sorted(by_scene[scene], key=lambda t: hashlib.sha256(f"{SPLIT_SALT}\0{t}".encode()).hexdigest())
        n_cal = max(1, round(len(ranked) * calibration))
        n_test = max(1, round(len(ranked) * test))
        for rank, trajectory in enumerate(ranked):
            role[trajectory] = "calibration" if rank < n_cal else "test" if rank < n_cal + n_test else "train"
    return [role[s["traj_rel_dir"]] for s in samples]


def flip_draw(index, seed, epoch):
    payload = f"{FLIP_SALT}\0{seed}\0{epoch}\0{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(2 ** 64)


class AerialDepthDataset(Dataset):
    def __init__(self, samples, indices, rgb_root, depth_root, flip=0.0, seed=0):
        self.samples, self.indices = samples, indices
        self.rgb_root, self.depth_root = Path(rgb_root), Path(depth_root)
        self.flip, self.seed, self.epoch = flip, seed, 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        sample = self.samples[index]
        folder_rgb = self.rgb_root / sample["traj_rel_dir"]
        folder_depth = self.depth_root / sample["traj_rel_dir"]
        views, depths = [], []
        for camera in CAMERAS:
            with Image.open(folder_rgb / camera / sample["img_name"]) as image:
                views.append(image.convert("RGB"))
            with Image.open(folder_depth / f"{camera}_depth" / sample["img_name"]) as image:
                depths.append(np.array(image.convert("L"), dtype=np.uint8))
        depth = np.stack(depths)
        if self.flip and flip_draw(index, self.seed, self.epoch) < self.flip:
            views = [view.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for view in views]
            depth = np.flip(depth, axis=2).copy()
        return torch.stack([preprocess_view(view) for view in views]), torch.from_numpy(depth)


def predict(model, views):
    batch = views.shape[0]
    depth = model(views.flatten(0, 1))
    if depth.ndim == 4:
        depth = depth[:, 0]
    depth = F.interpolate(depth[:, None].float(), size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=True)
    return depth.reshape(batch, len(CAMERAS), OUTPUT_SIZE, OUTPUT_SIZE).clamp(0.0, MAX_DEPTH_M)


def silog_loss(prediction, target_code):
    code = target_code.float()
    target = (code + 0.5) * (DEPTH_RANGE_M / 255.0)
    valid = target_code < CAP_CODE
    difference = torch.log(target[valid].clamp_min(1e-3)) - torch.log(prediction.float()[valid].clamp_min(1e-3))
    if difference.numel() == 0:
        return prediction.sum() * 0.0
    return torch.sqrt((difference.square().mean() - 0.5 * difference.mean().square()).clamp_min(1e-12))


@torch.no_grad()
def evaluate(model, loader, device):
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for views, code in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = predict(model, views.to(device))
        code = code.to(device)
        valid = code < CAP_CODE
        target = ((code.float() + 0.5) * (DEPTH_RANGE_M / 255.0))[valid]
        pred = prediction.float()[valid].clamp_min(1e-3)
        ratio = torch.maximum(pred / target, target / pred)
        totals += torch.stack([((pred - target).abs() / target).sum().double(), (ratio < 1.25).sum().double(),
                               (pred - target).square().sum().double(), torch.tensor(float(valid.sum()), device=device, dtype=torch.float64)])
    if dist.is_initialized():
        dist.all_reduce(totals)
    count = totals[3].clamp_min(1)
    return {"abs_rel": float(totals[0] / count), "d1": float(totals[1] / count), "rmse_m": float(torch.sqrt(totals[2] / count))}


def cosine_schedule(optimizer, total_steps, warmup_ratio):
    warmup = round(total_steps * warmup_ratio)

    def scale(step):
        if warmup and step < warmup:
            return max(1e-8, step / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def build_optimizer(model, head_lr, encoder_lr, weight_decay):
    head = list(model.depth_head.parameters())
    head_ids = {id(p) for p in head}
    encoder = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW([{"params": head, "lr": head_lr}, {"params": encoder, "lr": encoder_lr}],
                             weight_decay=weight_decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rgb-root", required=True)
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--global-batch", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    accumulation = args.global_batch // (world * args.micro_batch)
    if accumulation < 1 or accumulation * world * args.micro_batch != args.global_batch:
        raise SystemExit("--global-batch must be a multiple of GPUs x --micro-batch")

    samples = json.loads(Path(args.manifest).read_text())
    roles = trajectory_split(samples)
    split = {name: [i for i, r in enumerate(roles) if r == name] for name in ("train", "calibration", "test")}
    train_set = AerialDepthDataset(samples, split["train"], args.rgb_root, args.depth_root, flip=0.5, seed=args.seed)
    eval_sets = {name: AerialDepthDataset(samples, split[name], args.rgb_root, args.depth_root)
                 for name in ("calibration", "test")}
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(train_set, batch_size=args.micro_batch, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, drop_last=False)
    eval_loaders = {name: DataLoader(data, batch_size=args.micro_batch, num_workers=args.workers,
                                     sampler=DistributedSampler(data, num_replicas=world, rank=rank, shuffle=False))
                    for name, data in eval_sets.items()}

    model = build_depth_model()
    if args.init:
        state = load_depth_state(args.init)
    else:
        state = torch.hub.load_state_dict_from_url(VKITTI_URL, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    ddp = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    steps_per_epoch = math.ceil(len(loader) / accumulation)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print(json.dumps({"train": len(split["train"]), "calibration": len(split["calibration"]),
                          "test": len(split["test"]), "steps_per_epoch": steps_per_epoch}), flush=True)

    optimizer = scheduler = None
    step = 0
    for epoch in range(args.epochs):
        if epoch in (0, 1):
            optimizer = build_optimizer(model, args.head_lr, args.encoder_lr, args.weight_decay)
            stage_epochs = 1 if epoch == 0 else args.epochs - 1
            scheduler = cosine_schedule(optimizer, steps_per_epoch * stage_epochs, args.warmup_ratio)
        sampler.set_epoch(epoch)
        train_set.epoch = epoch
        ddp.train()
        optimizer.zero_grad(set_to_none=True)
        for micro, (views, code) in enumerate(loader):
            last = micro == len(loader) - 1
            boundary = (micro + 1) % accumulation == 0 or last
            group = accumulation if not last or (micro + 1) % accumulation == 0 else (micro + 1) % accumulation
            views, code = views.to(device, non_blocking=True), code.to(device, non_blocking=True)
            context = ddp.no_sync() if not boundary else torch.enable_grad()
            with context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = silog_loss(predict(ddp, views), code)
                (loss / group).backward()
            if not boundary:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and step % 100 == 0:
                print(f"epoch {epoch} step {step} silog {float(loss):.4f}", flush=True)
        ddp.eval()
        metrics = evaluate(model, eval_loaders["calibration"], device)
        if rank == 0:
            print(json.dumps({"epoch": epoch, "calibration": metrics}), flush=True)
            torch.save(model.state_dict(), output / f"aerial_depth_epoch{epoch}.pt")
    metrics = evaluate(model, eval_loaders["test"], device)
    if rank == 0:
        print(json.dumps({"test": metrics}), flush=True)
        torch.save(model.state_dict(), output / "aerial_depth.pt")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
