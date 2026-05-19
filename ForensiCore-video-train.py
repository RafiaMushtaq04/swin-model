import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img

from ForensiCore import build_video_model

MODEL_NAME = "ForensiCore-FakeAV-Video"
NUM_CLASSES = 1
PATCH_SIZE = (3, 3)
EMBED_DIM = 64
NUM_HEADS = 8
WINDOW_SIZE = 2
SHIFT_SIZE = 1
NUM_MLP = 256
QKV_BIAS = True
DROPOUT_RATE = 0.15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 2e-4
LABEL_SMOOTHING = 0.0
BATCH_SIZE = 32
EPOCHS = 20
RANDOM_STATE = 3
AUTOTUNE = tf.data.AUTOTUNE

AUGMENT_TRAIN = True
AUGMENT_PROB = 0.6
BRIGHTNESS_DELTA = 0.1
CONTRAST_RANGE = (0.8, 1.2)
JPEG_QUALITY_RANGE = (70, 100)
BLUR_PROB = 0.2

INPUT_SIZE_MODE = "fixed"
FIXED_INPUT_SIZE = (224, 224)
SIZE_SCAN_MAX_SAMPLES = 3000

DEFAULT_DATA_ROOT = "/kaggle/working/fakeav_video_frames"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
LABELS = {"real": 0, "fake": 1}


def collect_labeled_paths(root_dir, label_map, image_exts):
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Warning: dataset root not found -> {root_dir}")
        return [], []

    file_paths = []
    labels = []

    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in image_exts:
            continue

        matched_label = None
        for folder_name, label in label_map.items():
            if path.parent.name.lower() == folder_name.lower():
                matched_label = label
                break

        if matched_label is not None:
            file_paths.append(str(path))
            labels.append(matched_label)

    return file_paths, labels


def print_binary_counts(header, labels):
    labels = np.asarray(labels)
    real_count = int(np.sum(labels == 0))
    fake_count = int(np.sum(labels == 1))
    print(f"{header}: {real_count} is real and {fake_count} is fake")


def print_stage_banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_run_config(input_shape):
    print_stage_banner("Run Configuration")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"input_shape: {input_shape}")
    print(f"NUM_CLASSES: {NUM_CLASSES}")
    print(f"PATCH_SIZE: {PATCH_SIZE}")
    print(f"EMBED_DIM: {EMBED_DIM}, NUM_HEADS: {NUM_HEADS}, NUM_MLP: {NUM_MLP}")
    print(f"WINDOW_SIZE: {WINDOW_SIZE}, SHIFT_SIZE: {SHIFT_SIZE}")
    print(f"DROPOUT_RATE: {DROPOUT_RATE}, LEARNING_RATE: {LEARNING_RATE}, WEIGHT_DECAY: {WEIGHT_DECAY}")
    print(f"BATCH_SIZE: {BATCH_SIZE}, EPOCHS: {EPOCHS}, RANDOM_STATE: {RANDOM_STATE}")
    print(f"INPUT_SIZE_MODE: {INPUT_SIZE_MODE}, FIXED_INPUT_SIZE: {FIXED_INPUT_SIZE}")


def scan_image_sizes(paths, max_samples=3000, seed=3):
    if len(paths) == 0:
        return Counter()

    rng = np.random.default_rng(seed)
    sample_count = min(max_samples, len(paths))
    sampled_idx = rng.choice(len(paths), size=sample_count, replace=False)

    size_counter = Counter()
    for idx in sampled_idx:
        try:
            with load_img(paths[idx]) as image:
                width, height = image.size
            size_counter[(height, width)] += 1
        except Exception:
            continue

    return size_counter


def resolve_input_shape(paths):
    size_counter = scan_image_sizes(paths, max_samples=SIZE_SCAN_MAX_SAMPLES, seed=RANDOM_STATE)
    print_stage_banner("Dataset Image Size Scan")
    if not size_counter:
        print("No image sizes could be scanned. Falling back to FIXED_INPUT_SIZE.")
        return (FIXED_INPUT_SIZE[0], FIXED_INPUT_SIZE[1], 3)

    top_sizes = size_counter.most_common(5)
    print("Top image sizes (height, width) from sampled files:")
    for (h, w), count in top_sizes:
        print(f"- {(h, w)} -> {count} samples")

    most_common_size = top_sizes[0][0]
    if INPUT_SIZE_MODE.lower() == "auto":
        chosen_size = most_common_size
        print(f"Auto mode: selected most common size {chosen_size}")
    else:
        chosen_size = FIXED_INPUT_SIZE
        print(f"Fixed mode: forcing resize to {chosen_size}")
        if chosen_size != most_common_size:
            print(f"Note: most common dataset size is {most_common_size}, but fixed size is {chosen_size}")

    return (chosen_size[0], chosen_size[1], 3)


def compute_class_weight(labels):
    labels = np.asarray(labels)
    real_count = int(np.sum(labels == 0))
    fake_count = int(np.sum(labels == 1))
    total = real_count + fake_count
    if real_count == 0 or fake_count == 0:
        raise ValueError("Cannot compute class weights: one class has zero samples.")

    weight_for_real = total / (2.0 * real_count)
    weight_for_fake = total / (2.0 * fake_count)
    return {0: weight_for_real, 1: weight_for_fake}


def build_balanced_eval_dataset(paths, labels, target_shape, batch_size, seed=3):
    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels, dtype=np.int32)

    real_idx = np.where(labels == 0)[0]
    fake_idx = np.where(labels == 1)[0]
    if len(real_idx) == 0 or len(fake_idx) == 0:
        raise ValueError("Balanced evaluation requires both classes in the split.")

    rng = np.random.default_rng(seed)
    per_class = min(len(real_idx), len(fake_idx))
    real_sel = rng.choice(real_idx, size=per_class, replace=False)
    fake_sel = rng.choice(fake_idx, size=per_class, replace=False)
    balanced_idx = np.concatenate([real_sel, fake_sel])
    rng.shuffle(balanced_idx)

    balanced_paths = paths[balanced_idx].tolist()
    balanced_labels = labels[balanced_idx]

    dataset = build_tf_dataset(
        balanced_paths,
        balanced_labels,
        target_shape,
        batch_size,
        shuffle=False,
    )
    return dataset, len(balanced_paths)


def build_tf_dataset(paths, labels, target_shape, batch_size, shuffle=False, seed=3):
    target_height, target_width = target_shape[:2]
    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _apply_blur(image):
        return tf.nn.avg_pool2d(image[None, ...], ksize=3, strides=1, padding="SAME")[0]

    def _decode_and_preprocess(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [target_height, target_width])
        image = tf.cast(image, tf.float32) / 255.0
        if shuffle and AUGMENT_TRAIN:
            if tf.random.uniform(()) < AUGMENT_PROB:
                image = tf.image.random_flip_left_right(image)
            if tf.random.uniform(()) < AUGMENT_PROB:
                image = tf.image.random_brightness(image, max_delta=BRIGHTNESS_DELTA)
            if tf.random.uniform(()) < AUGMENT_PROB:
                image = tf.image.random_contrast(image, CONTRAST_RANGE[0], CONTRAST_RANGE[1])
            if tf.random.uniform(()) < AUGMENT_PROB:
                image_uint8 = tf.cast(tf.clip_by_value(image * 255.0, 0.0, 255.0), tf.uint8)
                image_uint8 = tf.image.random_jpeg_quality(
                    image_uint8, JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1]
                )
                image = tf.cast(image_uint8, tf.float32) / 255.0
            if tf.random.uniform(()) < BLUR_PROB:
                image = _apply_blur(image)

        image = tf.image.per_image_standardization(image)
        if NUM_CLASSES == 1:
            label = tf.cast(label, tf.float32)
        else:
            label = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
        return image, label

    dataset = dataset.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def run_video_pipeline(data_root=DEFAULT_DATA_ROOT):
    train_paths, train_labels = collect_labeled_paths(
        Path(data_root) / "train", LABELS, IMAGE_EXTS
    )
    val_paths, val_labels = collect_labeled_paths(
        Path(data_root) / "validation", LABELS, IMAGE_EXTS
    )
    test_paths, test_labels = collect_labeled_paths(
        Path(data_root) / "test", LABELS, IMAGE_EXTS
    )

    if len(train_paths) == 0:
        raise ValueError("No images found. Check dataset roots and folder mappings.")

    input_shape = resolve_input_shape(train_paths)
    print_run_config(input_shape)

    print_binary_counts("Train split", train_labels)
    print_binary_counts("Validation split", val_labels)
    print_binary_counts("Test split", test_labels)

    class_weight = compute_class_weight(train_labels)
    print(f"Class weights: {class_weight}")

    print_stage_banner("Building tf.data Pipelines")
    train_ds = build_tf_dataset(
        train_paths,
        train_labels,
        input_shape,
        BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_STATE,
    )

    val_ds, val_count = build_balanced_eval_dataset(
        val_paths,
        val_labels,
        input_shape,
        BATCH_SIZE,
        seed=RANDOM_STATE + 1,
    )

    test_ds = build_tf_dataset(
        test_paths,
        test_labels,
        input_shape,
        BATCH_SIZE,
        shuffle=False,
    )

    train_steps = (len(train_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    val_batches = (val_count + BATCH_SIZE - 1) // BATCH_SIZE

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=max(train_steps * EPOCHS, 1),
        alpha=0.1,
    )

    model = build_video_model(
        input_shape=input_shape,
        num_classes=NUM_CLASSES,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        shift_size=SHIFT_SIZE,
        num_mlp=NUM_MLP,
        qkv_bias=QKV_BIAS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        label_smoothing=LABEL_SMOOTHING,
        output_activation="sigmoid",
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=WEIGHT_DECAY,
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        verbose=1,
        restore_best_weights=True,
    )
    model_checkpoint = ModelCheckpoint(
        f"{MODEL_NAME}-best.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    print_stage_banner("Training Started")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_batches,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        class_weight=class_weight,
    )

    print_stage_banner("Training Summary")
    print(f"Best val_auc: {np.max(history.history['val_auc']):.4f}")

    print_stage_banner("Test Evaluation")
    test_results = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    print(f"Test precision: {test_results[2]:.4f}")
    print(f"Test recall: {test_results[3]:.4f}")
    print(f"Test AUC: {test_results[4]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train on FakeAVCeleb video frames")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    run_video_pipeline(args.data_root)


if __name__ == "__main__":
    main()
