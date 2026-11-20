import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from grid.actions import navigation_prompt, parse_action
from grid.model import OPENVLA, OPENVLA_REVISION
from grid.policy import GRIDPolicy

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "assets" / "example"
DESCRIPTION = ("The red car with white stripes and a sunroof is parked on a cracked asphalt surface in an urban area "
               "between closely packed buildings featuring arched windows and fire escapes; it's located near a corner "
               "with streets extending in multiple directions amongst mid-rise buildings under a clear sky.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "checkpoints" / "grid"))
    parser.add_argument("--base-model", default=OPENVLA)
    parser.add_argument("--revision", default=OPENVLA_REVISION)
    parser.add_argument("--front", default=str(EXAMPLE / "front.png"))
    parser.add_argument("--down", default=str(EXAMPLE / "down.png"))
    parser.add_argument("--hint", default="forward-right")
    parser.add_argument("--description", default=DESCRIPTION)
    parser.add_argument("--save-depth", default=None)
    args = parser.parse_args()

    policy = GRIDPolicy(args.weights, base_model=args.base_model, revision=args.revision)
    front = Image.open(args.front).convert("RGB")
    down = Image.open(args.down).convert("RGB")
    hint = f"{args.hint.strip()} " if args.hint.strip() else ""
    inputs = policy.prepare(front, down, navigation_prompt(hint, args.description))
    text = policy.generate(inputs)
    action = parse_action(text)

    depth = inputs["depth"][0].float().cpu().numpy()
    print(f"output:  {text.split('Action:')[-1].replace('</s>', '').strip()}")
    print(f"action:  forward {action['fwd']:.2f} m, down {action['down']:.2f} m, yaw {action['yaw']:.3f} rad, "
          f"land {action['land']}")
    print(f"depth:   front median {np.median(depth[0]):.1f} m, down median {np.median(depth[1]):.1f} m")
    if args.save_depth:
        np.save(args.save_depth, depth)


if __name__ == "__main__":
    main()
