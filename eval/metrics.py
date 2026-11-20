import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("seen_valset", "unseen_object_valset", "unseen_map_valset")
DIFFICULTIES = ("full", "easy", "hard")
PATTERNS = {
    "count": r"Total Count:\s*([\d.]+)",
    "sr": r"Success Rate \(SR\):\s*([\d.]+)",
    "osr": r"Oracle Success Rate \(OSR\):\s*([\d.]+)",
    "ne": r"Average Normalized Error \(NE\):\s*([\d.]+)",
    "spl": r"Average Success Path Length \(SPL\):\s*([\d.]+)",
}


def map_metrics(harness, results, category, scene):
    output = subprocess.run(
        [sys.executable, str(harness / "utils" / "metric.py"), "--root_dir", str(results),
         "--analysis_list", f"{category}/{scene}", "--path_type_list", *DIFFICULTIES, "--map_filter", scene],
        cwd=harness, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True,
    ).stdout
    values = {key: [float(v) for v in re.findall(pattern, output)] for key, pattern in PATTERNS.items()}
    return [{key: values[key][i] if i < len(values[key]) else 0.0 for key in PATTERNS} for i in range(len(DIFFICULTIES))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--harness", default=str(ROOT / "third_party" / "AeroVLA"))
    args = parser.parse_args()
    results, harness = Path(args.results).resolve(), Path(args.harness).resolve()

    rows = []
    for category in CATEGORIES:
        for scene_dir in sorted((results / category).glob("*")):
            if scene_dir.is_dir():
                for difficulty, metrics in zip(DIFFICULTIES, map_metrics(harness, results, category, scene_dir.name)):
                    rows.append({"category": category, "map": scene_dir.name, "difficulty": difficulty, **metrics})
    with open(results / "metrics_per_map.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["category", "map", "difficulty", *PATTERNS])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'split':<22}{'subset':<7}{'N':>6}{'SR':>8}{'OSR':>8}{'NE':>9}{'SPL':>8}")
    summary = []
    for category in CATEGORIES:
        for difficulty in DIFFICULTIES:
            selected = [r for r in rows if r["category"] == category and r["difficulty"] == difficulty and r["count"] > 0]
            count = sum(r["count"] for r in selected)
            if not count:
                continue
            mean = {key: sum(r["count"] * r[key] for r in selected) / count for key in ("sr", "osr", "ne", "spl")}
            summary.append({"category": category, "difficulty": difficulty, "count": count, **mean})
            print(f"{category:<22}{difficulty:<7}{int(count):>6}{mean['sr']:>8.2f}{mean['osr']:>8.2f}"
                  f"{mean['ne']:>9.2f}{mean['spl']:>8.2f}")
    with open(results / "metrics_summary.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["category", "difficulty", "count", "sr", "osr", "ne", "spl"])
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
