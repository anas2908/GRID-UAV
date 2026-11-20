import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from grid.data import load_manifest, view_paths
from grid.depth import AerialDepth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    samples = load_manifest(args.manifest)
    output = Path(args.output)
    if rank == 0 and not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(output, mode="w+", dtype=np.float16, shape=(len(samples), 2, 64, 64)).flush()
    if world_size > 1:
        dist.barrier()
    cache = np.load(output, mmap_mode="r+")
    model = AerialDepth(args.checkpoint, device=f"cuda:{local_rank}")

    indices = list(range(rank, len(samples), world_size))
    for start in range(0, len(indices), args.batch_size):
        batch = indices[start:start + args.batch_size]
        views = []
        for index in batch:
            for path in view_paths(args.data_root, samples[index]):
                with Image.open(path) as image:
                    views.append(image.convert("RGB"))
        depth = model.predict(views).reshape(len(batch), 2, 64, 64)
        cache[batch] = depth
        if rank == 0 and (start // args.batch_size) % 200 == 0:
            print(f"{start * world_size}/{len(samples)}", flush=True)
    cache.flush()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
