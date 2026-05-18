import argparse
import os
from pathlib import Path
import random

import librosa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


DEFAULT_ROOT = "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA"
DEFAULT_OUT = "/kaggle/working/asvspoof2019_la_png"

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 20
FMAX = 8000


def read_protocol(protocol_path):
    entries = []
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id = parts[1]
            label = parts[-1].lower()
            if label not in {"bonafide", "spoof"}:
                continue
            entries.append((file_id, label))
    return entries


def summarize_protocol(protocol_path):
    entries = read_protocol(protocol_path)
    real_count = sum(1 for _, label in entries if label == "bonafide")
    fake_count = sum(1 for _, label in entries if label == "spoof")
    return real_count, fake_count


def find_audio_path(root_dir, split_dir, file_id):
    for ext in (".flac", ".wav"):
        candidate = Path(root_dir) / split_dir / "flac" / f"{file_id}{ext}"
        if candidate.exists():
            return candidate
        candidate = Path(root_dir) / split_dir / f"{file_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def log_mel_spectrogram(audio_path):
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


def save_png(log_mel, output_path):
    log_mel = np.clip(log_mel, -80.0, 0.0)
    log_mel = (log_mel + 80.0) / 80.0
    log_mel = (log_mel * 255.0).astype(np.uint8)
    image = Image.fromarray(log_mel)
    image = image.convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def process_split(root_dir, split_dir, protocol_path, out_dir, subset_per_class=None, seed=3):
    entries = read_protocol(protocol_path)
    if not entries:
        raise RuntimeError(f"No protocol entries found: {protocol_path}")

    rng = random.Random(seed)
    real = [e for e in entries if e[1] == "bonafide"]
    fake = [e for e in entries if e[1] == "spoof"]

    if subset_per_class is not None and subset_per_class > 0:
        real = rng.sample(real, min(subset_per_class, len(real)))
        fake = rng.sample(fake, min(subset_per_class, len(fake)))

    selected = real + fake
    rng.shuffle(selected)

    for file_id, label in selected:
        audio_path = find_audio_path(root_dir, split_dir, file_id)
        if audio_path is None:
            continue
        log_mel = log_mel_spectrogram(audio_path)
        label_dir = "real" if label == "bonafide" else "fake"
        output_path = Path(out_dir) / label_dir / f"{file_id}.png"
        save_png(log_mel, output_path)


def preview_samples(out_dir, per_class=3, seed=3):
    rng = random.Random(seed)
    fig, axes = plt.subplots(2, per_class, figsize=(3 * per_class, 6))

    for row, label_dir in enumerate(["real", "fake"]):
        class_dir = Path(out_dir) / label_dir
        candidates = list(class_dir.glob("*.png"))
        if not candidates:
            continue
        picks = rng.sample(candidates, min(per_class, len(candidates)))
        for col, path in enumerate(picks):
            try:
                img = Image.open(path)
                axes[row, col].imshow(img)
                axes[row, col].axis("off")
            except Exception:
                axes[row, col].axis("off")

    axes[0, 0].set_title("real")
    axes[1, 0].set_title("fake")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "preview_grid.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ASVspoof2019 LA into PNG log-mel")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--train-per-class", type=int, default=10000)
    parser.add_argument("--dev-per-class", type=int, default=2000)
    parser.add_argument("--eval-per-class", type=int, default=2000)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    root_dir = args.root
    out_dir = args.out

    train_protocol = Path(root_dir) / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_protocol = Path(root_dir) / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_protocol = Path(root_dir) / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"

    if not train_protocol.exists():
        raise FileNotFoundError(f"Missing train protocol: {train_protocol}")
    if not dev_protocol.exists():
        raise FileNotFoundError(f"Missing dev protocol: {dev_protocol}")
    if not eval_protocol.exists():
        raise FileNotFoundError(f"Missing eval protocol: {eval_protocol}")

    train_real, train_fake = summarize_protocol(train_protocol)
    dev_real, dev_fake = summarize_protocol(dev_protocol)
    eval_real, eval_fake = summarize_protocol(eval_protocol)
    print("Protocol summary (bonafide vs spoof):")
    print(f"- train: {train_real} real, {train_fake} fake")
    print(f"- dev:   {dev_real} real, {dev_fake} fake")
    print(f"- eval:  {eval_real} real, {eval_fake} fake")

    if args.summary_only:
        return

    print("Preprocessing train split...")
    process_split(
        root_dir,
        "ASVspoof2019_LA_train",
        train_protocol,
        Path(out_dir) / "train",
        subset_per_class=args.train_per_class,
        seed=args.seed,
    )

    print("Preprocessing validation split...")
    process_split(
        root_dir,
        "ASVspoof2019_LA_dev",
        dev_protocol,
        Path(out_dir) / "validation",
        subset_per_class=args.dev_per_class,
        seed=args.seed + 1,
    )

    print("Preprocessing eval split...")
    process_split(
        root_dir,
        "ASVspoof2019_LA_eval",
        eval_protocol,
        Path(out_dir) / "test",
        subset_per_class=args.eval_per_class,
        seed=args.seed + 2,
    )

    if args.preview:
        print("Saving preview grid...")
        preview_samples(Path(out_dir) / "train", per_class=3, seed=args.seed)

    print("Preprocessing complete.")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
