import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

DEFAULT_ROOT = "/kaggle/input/fakeavceleb/FakeAVCeleb"
DEFAULT_OUT = "/kaggle/working/fakeav_video_frames"


def _label_from_top_dir(top_dir):
    name = top_dir.lower()
    if "fakevideo" in name:
        return "fake"
    if "realvideo" in name:
        return "real"
    return None


def _group_id_from_parts(parts):
    lower = [part.lower() for part in parts]
    gender_idx = None
    if "men" in lower:
        gender_idx = lower.index("men")
    elif "women" in lower:
        gender_idx = lower.index("women")

    if gender_idx is not None and gender_idx + 1 < len(parts):
        return "/".join(parts[: gender_idx + 2])

    return "/".join(parts[:-1])


def scan_videos(root_dir):
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    items = []
    for video_path in root_path.rglob("*"):
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTS:
            continue

        rel_parts = video_path.relative_to(root_path).parts
        if len(rel_parts) < 2:
            continue

        label = _label_from_top_dir(rel_parts[0])
        if label is None:
            continue

        group_id = _group_id_from_parts(rel_parts)
        items.append({
            "path": str(video_path),
            "label": label,
            "group_id": group_id,
        })

    if not items:
        raise RuntimeError("No videos found. Check dataset path and structure.")

    return items


def _split_groups(groups, train_split, val_split):
    n_train = int(len(groups) * train_split)
    n_val = int(len(groups) * val_split)
    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train + n_val]
    test_groups = groups[n_train + n_val :]
    return train_groups, val_groups, test_groups


def split_by_group(items, train_split, val_split, seed):
    rng = random.Random(seed)

    group_map = {}
    group_label = {}
    for item in items:
        group_map.setdefault(item["group_id"], []).append(item)
        group_label.setdefault(item["group_id"], item["label"])

    real_groups = [gid for gid, label in group_label.items() if label == "real"]
    fake_groups = [gid for gid, label in group_label.items() if label == "fake"]

    rng.shuffle(real_groups)
    rng.shuffle(fake_groups)

    real_train, real_val, real_test = _split_groups(real_groups, train_split, val_split)
    fake_train, fake_val, fake_test = _split_groups(fake_groups, train_split, val_split)

    def expand(groups):
        expanded = []
        for gid in groups:
            expanded.extend(group_map[gid])
        return expanded

    train_items = expand(real_train + fake_train)
    val_items = expand(real_val + fake_val)
    test_items = expand(real_test + fake_test)

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)

    return train_items, val_items, test_items


def summarize_items(header, items):
    real_count = sum(1 for item in items if item["label"] == "real")
    fake_count = sum(1 for item in items if item["label"] == "fake")
    print(f"{header}: {real_count} real, {fake_count} fake")


def _sample_per_class(items, max_per_class, seed):
    if max_per_class is None or max_per_class <= 0:
        return items

    rng = random.Random(seed)
    real_items = [item for item in items if item["label"] == "real"]
    fake_items = [item for item in items if item["label"] == "fake"]

    if real_items:
        real_items = rng.sample(real_items, min(max_per_class, len(real_items)))
    if fake_items:
        fake_items = rng.sample(fake_items, min(max_per_class, len(fake_items)))

    sampled = real_items + fake_items
    rng.shuffle(sampled)
    return sampled


def _frame_indices(frame_count, frames_per_video):
    if frame_count <= 0:
        return []
    if frame_count <= frames_per_video:
        return list(range(frame_count))
    step = frame_count / float(frames_per_video)
    return [int(step * i) for i in range(frames_per_video)]


def _resolve_frames_per_video(frame_count, fps, frames_per_video, frames_per_second):
    if frames_per_second and fps > 0:
        duration = frame_count / fps if frame_count > 0 else 0
        if duration > 0:
            return max(1, int(round(duration * frames_per_second)))
    return frames_per_video


def extract_frames(video_path, output_dir, frames_per_video, frames_per_second):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    target_frames = _resolve_frames_per_video(frame_count, fps, frames_per_video, frames_per_second)
    if frame_count <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        frame_count = len(frames)
        indices = _frame_indices(frame_count, target_frames)
        saved = 0
        for idx, frame_idx in enumerate(indices):
            frame = frames[frame_idx]
            out_path = output_dir / f"{Path(video_path).stem}_f{idx:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        return saved

    indices = _frame_indices(frame_count, target_frames)
    saved = 0
    for idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = output_dir / f"{Path(video_path).stem}_f{idx:03d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved


def process_split(items, split_dir, out_root, frames_per_video, frames_per_second, max_per_class, seed):
    items = _sample_per_class(items, max_per_class, seed)
    if not items:
        return 0

    processed = 0
    for item in items:
        video_path = item["path"]
        label_dir = "real" if item["label"] == "real" else "fake"
        output_dir = Path(out_root) / split_dir / label_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        processed += extract_frames(video_path, output_dir, frames_per_video, frames_per_second)

    return processed


def main():
    parser = argparse.ArgumentParser(description="Extract frames from FakeAVCeleb videos")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--frames-per-second", type=float, default=2.0)
    parser.add_argument("--max-videos-per-class", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    if not np.isclose(args.train_split + args.val_split + args.test_split, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    items = scan_videos(args.root)
    summarize_items("Total", items)

    train_items, val_items, test_items = split_by_group(
        items,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
    )

    summarize_items("Train split", train_items)
    summarize_items("Validation split", val_items)
    summarize_items("Test split", test_items)

    if args.summary_only:
        return

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Extracting train split frames...")
    process_split(
        train_items,
        "train",
        out_root,
        args.frames_per_video,
        args.frames_per_second,
        args.max_videos_per_class,
        args.seed,
    )
    print("Extracting validation split frames...")
    process_split(
        val_items,
        "validation",
        out_root,
        args.frames_per_video,
        args.frames_per_second,
        args.max_videos_per_class,
        args.seed + 1,
    )
    print("Extracting test split frames...")
    process_split(
        test_items,
        "test",
        out_root,
        args.frames_per_video,
        args.frames_per_second,
        args.max_videos_per_class,
        args.seed + 2,
    )

    print("Frame extraction complete.")
    print(f"Output dir: {out_root}")


if __name__ == "__main__":
    main()
