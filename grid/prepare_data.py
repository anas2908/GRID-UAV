import argparse
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from grid.data import CAMERAS, load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archives", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--with-depth", action="store_true")
    args = parser.parse_args()

    folders = list(CAMERAS) + ([f"{camera}_depth" for camera in CAMERAS] if args.with_depth else [])
    members = defaultdict(set)
    for sample in load_manifest(args.manifest):
        trajectory = sample["traj_rel_dir"]
        for folder in folders:
            members[trajectory.split("/", 1)[0]].add(f"{trajectory}/{folder}/{sample['img_name']}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for scene, names in sorted(members.items()):
        archive = Path(args.archives) / f"{scene}.zip"
        missing = [name for name in names if not (output / name).exists()]
        print(f"{scene}: {len(names)} files, {len(missing)} to extract", flush=True)
        if not missing:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as listing:
            listing.write("\n".join(sorted(missing)) + "\n")
            listing.flush()
            subprocess.run(["7z", "x", "-y", f"-o{output}", str(archive), f"@{listing.name}"], check=True,
                           stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
