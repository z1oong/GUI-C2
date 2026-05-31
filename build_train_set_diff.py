import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


POINT_RE = re.compile(r'<points\s+x1="([0-9.]+)"\s+y1="([0-9.]+)"')
OS_ATLAS_MARKER = "/os-atlas/"
PATH_FIXUPS = {
    "web/web_domain/seeclick_web_images": "web/web_domain/seeclick_web_images/seeclick_web_imgs",
    "desktop/desktop_domain/linux_image": "desktop/desktop_domain/linux_images",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert train_set.json into train-set format with difficulty annotations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("train_set.json"),
        help="Path to the raw 8-trial training set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/train_set_diff.json"),
        help="Path to the converted difficulty-annotated training set.",
    )
    return parser.parse_args()


def parse_points(raw_outputs):
    points = []
    misses = 0
    for raw in raw_outputs:
        match = POINT_RE.search(raw or "")
        if match:
            points.append((float(match.group(1)) / 1000.0, float(match.group(2)) / 1000.0))
        else:
            misses += 1
    return points, misses


def distance_to_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = 0.0 if x1 <= x <= x2 else min(abs(x - x1), abs(x - x2))
    dy = 0.0 if y1 <= y <= y2 else min(abs(y - y1), abs(y - y2))
    return math.hypot(dx, dy)


def rms_dispersion(points):
    if not points:
        return 1.0
    center_x = sum(x for x, _ in points) / len(points)
    center_y = sum(y for _, y in points) / len(points)
    variance = sum((x - center_x) ** 2 + (y - center_y) ** 2 for x, y in points) / len(points)
    return math.sqrt(variance)


def clamp01(value):
    return max(0.0, min(1.0, value))


def difficulty_raw_score(sample):
    bbox = sample["bbox"]
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    area = width * height
    short_side = min(width, height)

    points, misses = parse_points(sample["raw_outputs"])
    total_trials = max(len(sample["grounding_trials"]), len(sample["raw_outputs"]), 1)
    correct_ratio = sample.get("correct_count", 0) / total_trials
    miss_ratio = misses / total_trials

    if points:
        mean_distance = sum(distance_to_bbox(point, bbox) for point in points) / len(points)
    else:
        mean_distance = 1.0

    size_by_short_side = 1.0 - min(1.0, short_side / 0.12)
    size_by_area = 1.0 - min(1.0, math.sqrt(area) / 0.22)
    distance_term = min(1.0, mean_distance / 0.20)
    dispersion_term = min(1.0, rms_dispersion(points) / 0.25)

    return clamp01(
        0.40 * (1.0 - correct_ratio)
        + 0.20 * distance_term
        + 0.15 * size_by_short_side
        + 0.10 * size_by_area
        + 0.10 * dispersion_term
        + 0.05 * miss_ratio
    )


def percentile_ranks(values):
    if not values:
        return []
    if len(values) == 1:
        return [0.5]

    ordered = sorted(range(len(values)), key=lambda index: values[index])
    percentiles = [0.0] * len(values)
    scale = len(values) - 1
    for rank, index in enumerate(ordered):
        percentiles[index] = rank / scale
    return percentiles





def normalize_image_path(source_file, img_filename):
    normalized = source_file.replace("\\", "/")
    if OS_ATLAS_MARKER in normalized:
        prefix, suffix = normalized.split(OS_ATLAS_MARKER, 1)
        del prefix
        relative_dir = PATH_FIXUPS.get(suffix.strip("/"), suffix.strip("/"))
        return f"{relative_dir}/{img_filename}"
    raise ValueError(f"Unsupported source_file path: {source_file}")


def bbox_to_solution(bbox):
    return [round(float(value) * 1000.0, 1) for value in bbox]


def main():
    args = parse_args()
    samples = json.loads(args.input.read_text())

    raw_scores = [difficulty_raw_score(sample) for sample in samples]
    indices_by_platform = defaultdict(list)
    for index, sample in enumerate(samples):
        platform = sample.get("platform", "unknown")
        indices_by_platform[platform].append(index)

    normalized_scores = [0.0] * len(samples)
    for global_indices in indices_by_platform.values():
        platform_percentiles = percentile_ranks([raw_scores[index] for index in global_indices])
        for local_index, global_index in enumerate(global_indices):
            normalized_scores[global_index] = platform_percentiles[local_index]

    converted = []
    for sample, normalized_score in zip(samples, normalized_scores):
        converted.append(
            {
                "image": normalize_image_path(sample["source_file"], sample["img_filename"]),
                "problem": sample["instruction"],
                "solution": bbox_to_solution(sample["bbox"]),
                "rate": 1,
                "difficulty_score": round(normalized_score, 6),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2, ensure_ascii=False))

    print(f"Wrote {len(converted)} samples to {args.output}")


if __name__ == "__main__":
    main()