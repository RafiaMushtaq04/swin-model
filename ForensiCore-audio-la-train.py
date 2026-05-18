import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from ForensiCore import build_audio_model

MODEL_NAME = "ForensiCore-Audio-LA"
NUM_CLASSES = 1
PATCH_SIZE = (3, 3)
EMBED_DIM = 64
NUM_HEADS = 4
WINDOW_SIZE = 2
NUM_MLP = 256
DROPOUT_RATE = 0.2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 2e-4
BATCH_SIZE = 16
EPOCHS = 20
RANDOM_STATE = 3
AUTOTUNE = tf.data.AUTOTUNE

DEFAULT_DATA_ROOT = "/kaggle/working/asvspoof2019_la_png"


def print_stage_banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def collect_labeled_paths(root_dir):
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Directory not found: {root_dir}")

    file_paths = []
    labels = []

    for path in root_path.rglob("*.png"):
        if not path.is_file():
            continue
        if path.parent.name.lower() == "real":
            label = 0
        elif path.parent.name.lower() == "fake":
            label = 1
        else:
            continue
        file_paths.append(str(path))
        labels.append(label)

    if len(file_paths) == 0:
        raise ValueError(f"No labeled images found in: {root_dir}")

    return file_paths, np.asarray(labels)


def print_binary_counts(header, labels):
    labels = np.asarray(labels)
    real_count = int(np.sum(labels == 0))
    fake_count = int(np.sum(labels == 1))
    print(f"{header}: {real_count} is real and {fake_count} is fake")


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

    def _decode_and_preprocess(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_png(image_bytes, channels=3)
        image = tf.image.resize(image, [target_height, target_width])
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.float32)
        return image, label

    dataset = dataset.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def build_balanced_train_dataset(paths, labels, target_shape, batch_size, seed=3):
    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels, dtype=np.int32)

    real_mask = labels == 0
    fake_mask = labels == 1
    if not np.any(real_mask) or not np.any(fake_mask):
        raise ValueError("Balanced sampling requires both classes to be present.")

    def build_unbatched(sub_paths, sub_labels, sub_seed):
        target_height, target_width = target_shape[:2]
        dataset = tf.data.Dataset.from_tensor_slices((sub_paths, sub_labels))
        dataset = dataset.shuffle(
            buffer_size=len(sub_paths),
            seed=sub_seed,
            reshuffle_each_iteration=True,
        )

        def _decode(path, label):
            image_bytes = tf.io.read_file(path)
            image = tf.io.decode_png(image_bytes, channels=3)
            image = tf.image.resize(image, [target_height, target_width])
            image = tf.cast(image, tf.float32) / 255.0
            label = tf.cast(label, tf.float32)
            return image, label

        return dataset.map(_decode, num_parallel_calls=AUTOTUNE)

    real_ds = build_unbatched(paths[real_mask].tolist(), labels[real_mask], seed)
    fake_ds = build_unbatched(paths[fake_mask].tolist(), labels[fake_mask], seed + 1)

    balanced = tf.data.Dataset.sample_from_datasets(
        [real_ds, fake_ds],
        weights=[0.5, 0.5],
        seed=seed,
    )
    return balanced.batch(batch_size).prefetch(AUTOTUNE)


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


def run_audio_pipeline(data_root):
    train_paths, train_labels = collect_labeled_paths(Path(data_root) / "train")
    val_paths, val_labels = collect_labeled_paths(Path(data_root) / "validation")
    test_root = Path(data_root) / "test"
    test_paths = []
    test_labels = np.asarray([])
    if test_root.exists():
        test_paths, test_labels = collect_labeled_paths(test_root)

    input_shape = (224, 224, 3)

    print_binary_counts("Train (before balancing)", train_labels)
    print_binary_counts("Validation (before balancing)", val_labels)
    if len(test_paths) > 0:
        print_binary_counts("Test (before balancing)", test_labels)

    print_stage_banner("Building tf.data Pipelines")
    train_ds = build_balanced_train_dataset(
        train_paths,
        train_labels,
        input_shape,
        BATCH_SIZE,
        seed=RANDOM_STATE,
    ).repeat()
    val_ds, val_count = build_balanced_eval_dataset(
        val_paths,
        val_labels,
        input_shape,
        BATCH_SIZE,
        seed=RANDOM_STATE + 1,
    )

    steps_per_epoch = max((2 * min(np.sum(train_labels == 0), np.sum(train_labels == 1))) // BATCH_SIZE, 1)
    val_batches = (val_count + BATCH_SIZE - 1) // BATCH_SIZE

    model = build_audio_model(
        input_shape=input_shape,
        num_classes=NUM_CLASSES,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        num_mlp=NUM_MLP,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        label_smoothing=0.0,
        output_activation="sigmoid",
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
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
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_batches,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
    )

    print_stage_banner("Training Summary")
    print(f"Best val_auc: {np.max(history.history['val_auc']):.4f}")

    if len(test_paths) > 0:
        print_stage_banner("Test Evaluation")
        test_ds = build_tf_dataset(
            test_paths,
            test_labels,
            input_shape,
            BATCH_SIZE,
            shuffle=False,
        )
        test_results = model.evaluate(test_ds, verbose=1)
        print(f"Test loss: {test_results[0]:.4f}")
        print(f"Test accuracy: {test_results[1]:.4f}")
        print(f"Test precision: {test_results[2]:.4f}")
        print(f"Test recall: {test_results[3]:.4f}")
        print(f"Test AUC: {test_results[4]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train on ASVspoof2019 LA PNGs")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    run_audio_pipeline(args.data_root)


if __name__ == "__main__":
    main()
