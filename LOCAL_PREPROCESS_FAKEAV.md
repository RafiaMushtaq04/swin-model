# FakeAVCeleb Audio Preprocessing (Local PC)

This guide is for preprocessing FakeAVCeleb audio into log-mel PNGs on a Windows PC.

## 1) Install prerequisites

- Git for Windows
- Python 3.10 or 3.11
- FFmpeg (required for audio extraction)

### FFmpeg setup (Windows)
1. Download FFmpeg release zip from https://ffmpeg.org/download.html
2. Extract to a folder, for example: C:\Tools\ffmpeg
3. Add to PATH:
   - Windows search: "Environment Variables" -> "Edit the system environment variables"
   - Click "Environment Variables" -> "Path" -> "New"
   - Add: C:\Tools\ffmpeg\bin
4. Open a new PowerShell and run:
   - `ffmpeg -version`

## 2) Get the code

Open PowerShell:

```
cd D:\fyp

git clone -b ide-fakeav https://github.com/RafiaMushtaq04/swin-model.git
cd swin-model
```

## 3) Create a Python environment

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Put the dataset on disk

Unzip or copy the dataset so it looks like this:

```
D:\datasets\FakeAVCeleb\
  FakeVideo-FakeAudio\
  FakeVideo-RealAudio\
  RealVideo-FakeAudio\
  RealVideo-RealAudio\
```

## 5) Run preprocessing

Basic run (all splits):

```
python ForensiCore-fakeav-audio-preprocess.py \
  --root D:\datasets\FakeAVCeleb \
  --out D:\datasets\fakeav_audio_png \
  --train-split 0.8 --val-split 0.1 --test-split 0.1
```

Split-by-split (safer for lower-spec PCs):

```
python ForensiCore-fakeav-audio-preprocess.py \
  --root D:\datasets\FakeAVCeleb \
  --out D:\datasets\fakeav_audio_png \
  --train-split 0.8 --val-split 0.1 --test-split 0.1 \
  --split train

python ForensiCore-fakeav-audio-preprocess.py \
  --root D:\datasets\FakeAVCeleb \
  --out D:\datasets\fakeav_audio_png \
  --train-split 0.8 --val-split 0.1 --test-split 0.1 \
  --split validation

python ForensiCore-fakeav-audio-preprocess.py \
  --root D:\datasets\FakeAVCeleb \
  --out D:\datasets\fakeav_audio_png \
  --train-split 0.8 --val-split 0.1 --test-split 0.1 \
  --split test
```

Chunked run (if full dataset is too slow):

```
python ForensiCore-fakeav-audio-preprocess.py \
  --root D:\datasets\FakeAVCeleb \
  --out D:\datasets\fakeav_audio_png \
  --train-split 0.8 --val-split 0.1 --test-split 0.1 \
  --split train --max-videos-per-class 200
```

Then increase `--max-videos-per-class` in later runs (400, 600, ...).

## 6) Output structure

The output keeps class labels and splits:

```
D:\datasets\fakeav_audio_png\
  train\real\*.png
  train\fake\*.png
  validation\real\*.png
  validation\fake\*.png
  test\real\*.png
  test\fake\*.png
```

## Troubleshooting

- If you see "ffmpeg failed", verify `ffmpeg -version` works in PowerShell.
- If you see "No videos found", check the dataset folder structure and `--root` path.
- If it is slow, run split-by-split and use `--max-videos-per-class`.
