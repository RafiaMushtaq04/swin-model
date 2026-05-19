import argparse
import os
import random
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
from PIL import Image

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

DEFAULT_ROOT = "/kaggle/input/fakeavceleb/FakeAVCeleb"
DEFAULT_OUT = "/kaggle/working/fakeav_audio_png"

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 20
FMAX = 8000


def _label_from_top_dir(top_dir):
    name = top_dir.lower()
    if "fakeaudio" in name:
        return "fake"
    if "realaudio" in name:
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


def _extract_audio(video_path, wav_path, sample_rate):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        wav_path,
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}")


def _log_mel(audio_path):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def _save_png(log_mel, output_path):
    log_mel = np.clip(log_mel, -80.0, 0.0)
    log_mel = (log_mel + 80.0) / 80.0
    log_mel = (log_mel * 255.0).astype(np.uint8)
    image = Image.fromarray(log_mel)
    image = image.convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


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


def process_split(items, split_dir, out_root, max_per_class, seed):
    items = _sample_per_class(items, max_per_class, seed)
    if not items:
        return 0

    processed = 0
    for item in items:
        video_path = item["path"]
        label_dir = "real" if item["label"] == "real" else "fake"
        output_path = Path(out_root) / split_dir / label_dir / f"{Path(video_path).stem}.png"
        if output_path.exists():
            processed += 1
            continue

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = str(Path(temp_dir) / "audio.wav")
            _extract_audio(video_path, wav_path, SAMPLE_RATE)
            log_mel = _log_mel(wav_path)
            _save_png(log_mel, output_path)
            processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser(description="Preprocess FakeAVCeleb audio to log-mel PNGs")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
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

    print("Preprocessing train split...")
    process_split(train_items, "train", out_root, args.max_videos_per_class, args.seed)
    print("Preprocessing validation split...")
    process_split(val_items, "validation", out_root, args.max_videos_per_class, args.seed + 1)
    print("Preprocessing test split...")
    process_split(test_items, "test", out_root, args.max_videos_per_class, args.seed + 2)

    print("Preprocessing complete.")
    print(f"Output dir: {out_root}")


if __name__ == "__main__":
    main()
