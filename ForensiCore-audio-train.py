import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img

from ForensiCore import build_audio_model

MODEL_NAME = 'ForensiCore-Audio'
NUM_CLASSES = 1
PATCH_SIZE = (3, 3)
EMBED_DIM = 32
NUM_HEADS = 4
WINDOW_SIZE = 2
SHIFT_SIZE = 1
NUM_MLP = 256
QKV_BIAS = True
DROPOUT_RATE = 0.25
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.0
BATCH_SIZE = 16
EPOCHS = 20
RANDOM_STATE = 3
AUTOTUNE = tf.data.AUTOTUNE

DATA_ROOT = '/kaggle/input/datasets/bishertello/asvspoof-21-df-cqt/my_dataset'
TRAIN_DIR = str(Path(DATA_ROOT) / 'train')
VAL_DIR = str(Path(DATA_ROOT) / 'validation')
TEST_DIR = str(Path(DATA_ROOT) / 'test')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
LABELS = {'real': 0, 'fake': 1}
PRINT_EVERY = 100000
USE_SPEC_AUG = False
SPEC_AUG_PROB = 0.3
TIME_MASK_MAX = 12
FREQ_MASK_MAX = 12
USE_STANDARDIZATION = False
USE_BASELINE = True
BASELINE_EPOCHS = 5
BASELINE_SAMPLES_PER_CLASS = 2000
USE_FOCAL_LOSS = False
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Input size control
# - 'fixed': always resize to FIXED_INPUT_SIZE
# - 'auto': scan training images and select most common size
INPUT_SIZE_MODE = 'fixed'
FIXED_INPUT_SIZE = (224, 224)
SIZE_SCAN_MAX_SAMPLES = 3000


def print_stage_banner(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def print_binary_counts(header, labels):
    labels = np.asarray(labels)
    real_count = int(np.sum(labels == 0))
    fake_count = int(np.sum(labels == 1))
    print(f"{header}: {real_count} is real and {fake_count} is fake")


def print_label_samples(paths, labels, sample_count=3, seed=3):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    for class_id, class_name in [(0, 'real'), (1, 'fake')]:
        class_idx = np.where(labels == class_id)[0]
        if len(class_idx) == 0:
            print(f"Label sample ({class_name}): none found")
            continue

        picked = rng.choice(class_idx, size=min(sample_count, len(class_idx)), replace=False)
        print(f"Label sample ({class_name}):")
        for idx in picked:
            print(f"- {paths[idx]}")


def plot_sample_grid(paths, labels, output_path, seed=3, per_class=3):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    fig, axes = plt.subplots(2, per_class, figsize=(3 * per_class, 6))
    for row, class_id in enumerate([0, 1]):
        class_idx = np.where(labels == class_id)[0]
        if len(class_idx) == 0:
            continue
        picked = rng.choice(class_idx, size=min(per_class, len(class_idx)), replace=False)
        for col, idx in enumerate(picked):
            try:
                img = load_img(paths[idx])
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
            except Exception:
                axes[row, col].axis('off')

    axes[0, 0].set_title('real')
    axes[1, 0].set_title('fake')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def collect_labeled_paths(root_dir, label_map, image_exts):
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Directory not found: {root_dir}")

    file_paths = []
    labels = []

    scanned = 0
    for path in root_path.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in image_exts:
            continue

        path_parts_lower = {part.lower() for part in path.parts}
        matched_label = None
        for folder_name, label in label_map.items():
            if folder_name.lower() in path_parts_lower:
                matched_label = label
                break

        if matched_label is not None:
            file_paths.append(str(path))
            labels.append(matched_label)

        scanned += 1
        if scanned % PRINT_EVERY == 0:
            print(f"Scanned {scanned:,} files in {root_dir}...")

    if len(file_paths) == 0:
        raise ValueError(f"No labeled images found in: {root_dir}")

    return file_paths, np.asarray(labels)


def scan_image_sizes(paths, max_samples=3000, seed=3):
    if len(paths) == 0:
        return {}

    rng = np.random.default_rng(seed)
    sample_count = min(max_samples, len(paths))
    sampled_idx = rng.choice(len(paths), size=sample_count, replace=False)

    size_counter = {}
    for idx in sampled_idx:
        try:
            with load_img(paths[idx]) as image:
                width, height = image.size
            key = (height, width)
            size_counter[key] = size_counter.get(key, 0) + 1
        except Exception:
            continue

    return size_counter


def resolve_input_shape(paths):
    size_counter = scan_image_sizes(paths, max_samples=SIZE_SCAN_MAX_SAMPLES, seed=RANDOM_STATE)
    print_stage_banner('Dataset Image Size Scan')

    if not size_counter:
        print('No image sizes scanned successfully. Falling back to fixed size.')
        return (FIXED_INPUT_SIZE[0], FIXED_INPUT_SIZE[1], 3)

    top_sizes = sorted(size_counter.items(), key=lambda item: item[1], reverse=True)[:5]
    print('Top image sizes (height, width) from sampled files:')
    for (h, w), count in top_sizes:
        print(f"- {(h, w)} -> {count} samples")

    most_common_size = top_sizes[0][0]
    if INPUT_SIZE_MODE.lower() == 'auto':
        chosen_size = most_common_size
        print(f"Auto mode: selected most common size {chosen_size}")
    else:
        chosen_size = FIXED_INPUT_SIZE
        print(f"Fixed mode: forcing resize to {chosen_size}")
        if chosen_size != most_common_size:
            print(f"Note: most common dataset size is {most_common_size}, but fixed size is {chosen_size}")

    return (chosen_size[0], chosen_size[1], 3)


def _apply_spec_augment(image):
    if tf.random.uniform(()) > SPEC_AUG_PROB:
        return image

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    t = tf.random.uniform((), 0, TIME_MASK_MAX + 1, dtype=tf.int32)
    f = tf.random.uniform((), 0, FREQ_MASK_MAX + 1, dtype=tf.int32)

    t0 = tf.random.uniform((), 0, tf.maximum(1, width - t + 1), dtype=tf.int32)
    f0 = tf.random.uniform((), 0, tf.maximum(1, height - f + 1), dtype=tf.int32)

    time_mask = tf.concat(
        [
            tf.ones([height, t0, 1], dtype=image.dtype),
            tf.zeros([height, t, 1], dtype=image.dtype),
            tf.ones([height, tf.maximum(0, width - t0 - t), 1], dtype=image.dtype),
        ],
        axis=1,
    )
    freq_mask = tf.concat(
        [
            tf.ones([f0, width, 1], dtype=image.dtype),
            tf.zeros([f, width, 1], dtype=image.dtype),
            tf.ones([tf.maximum(0, height - f0 - f), width, 1], dtype=image.dtype),
        ],
        axis=0,
    )

    return image * time_mask * freq_mask


def build_tf_dataset(paths, labels, target_shape, batch_size, shuffle=False, seed=3, augment=False):
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
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [target_height, target_width])
        image = tf.cast(image, tf.float32) / 255.0
        if USE_STANDARDIZATION:
            image = tf.image.per_image_standardization(image)
        if augment and USE_SPEC_AUG:
            image = _apply_spec_augment(image)
        if NUM_CLASSES == 1:
            label = tf.cast(label, tf.float32)
        else:
            label = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
        return image, label

    dataset = dataset.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def build_tf_dataset_unbatched(paths, labels, target_shape, shuffle=False, seed=3, augment=False):
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
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [target_height, target_width])
        image = tf.cast(image, tf.float32) / 255.0
        if USE_STANDARDIZATION:
            image = tf.image.per_image_standardization(image)
        if augment and USE_SPEC_AUG:
            image = _apply_spec_augment(image)
        if NUM_CLASSES == 1:
            label = tf.cast(label, tf.float32)
        else:
            label = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
        return image, label

    return dataset.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)


def build_balanced_train_dataset(paths, labels, target_shape, batch_size, seed=3):
    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels, dtype=np.int32)

    real_mask = labels == 0
    fake_mask = labels == 1
    if not np.any(real_mask) or not np.any(fake_mask):
        raise ValueError('Balanced sampling requires both classes to be present.')

    real_ds = build_tf_dataset_unbatched(
        paths[real_mask].tolist(),
        labels[real_mask],
        target_shape,
        shuffle=True,
        seed=seed,
        augment=True,
    )
    fake_ds = build_tf_dataset_unbatched(
        paths[fake_mask].tolist(),
        labels[fake_mask],
        target_shape,
        shuffle=True,
        seed=seed + 1,
        augment=True,
    )

    balanced = tf.data.Dataset.sample_from_datasets(
        [real_ds, fake_ds],
        weights=[0.5, 0.5],
        seed=seed,
    )
    return balanced.batch(batch_size).prefetch(AUTOTUNE)


def inspect_batch_balance(dataset, num_batches=3):
    for idx, (_, labels) in enumerate(dataset.take(num_batches)):
        mean_label = tf.reduce_mean(labels)
        print(f"Batch {idx + 1} label mean: {float(mean_label):.4f}")


def compute_steps_per_epoch(labels, batch_size):
    labels = np.asarray(labels)
    real_count = int(np.sum(labels == 0))
    fake_count = int(np.sum(labels == 1))
    per_class = min(real_count, fake_count)
    return max((2 * per_class) // batch_size, 1)


def build_balanced_eval_dataset(paths, labels, target_shape, batch_size, seed=3):
    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels, dtype=np.int32)

    real_idx = np.where(labels == 0)[0]
    fake_idx = np.where(labels == 1)[0]
    if len(real_idx) == 0 or len(fake_idx) == 0:
        raise ValueError('Balanced evaluation requires both classes in the split.')

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


def _sample_per_class(paths, labels, per_class, seed=3):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    paths = np.asarray(paths)

    real_idx = np.where(labels == 0)[0]
    fake_idx = np.where(labels == 1)[0]
    if len(real_idx) == 0 or len(fake_idx) == 0:
        return paths.tolist(), labels

    real_sel = rng.choice(real_idx, size=min(per_class, len(real_idx)), replace=False)
    fake_sel = rng.choice(fake_idx, size=min(per_class, len(fake_idx)), replace=False)
    idx = np.concatenate([real_sel, fake_sel])
    rng.shuffle(idx)
    return paths[idx].tolist(), labels[idx]


def build_baseline_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ],
    )
    return model


def run_audio_pipeline():
    print_stage_banner('Collecting Spectrogram Paths')
    train_paths, train_labels = collect_labeled_paths(TRAIN_DIR, LABELS, IMAGE_EXTS)
    val_paths, val_labels = collect_labeled_paths(VAL_DIR, LABELS, IMAGE_EXTS)
    test_paths, test_labels = collect_labeled_paths(TEST_DIR, LABELS, IMAGE_EXTS)

    input_shape = resolve_input_shape(train_paths)

    print_binary_counts('Train (before balancing)', train_labels)
    print_binary_counts('Validation (before balancing)', val_labels)
    print_binary_counts('Test (before balancing)', test_labels)
    print_label_samples(np.asarray(train_paths), train_labels, sample_count=3, seed=RANDOM_STATE)
    plot_sample_grid(train_paths, train_labels, f"{MODEL_NAME}-samples.png", seed=RANDOM_STATE)

    print_stage_banner('Building tf.data Pipelines')
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
    test_ds = build_tf_dataset(test_paths, test_labels, input_shape, BATCH_SIZE)

    print_stage_banner('Balanced Batch Sanity Check')
    inspect_batch_balance(train_ds, num_batches=3)

    steps_per_epoch = compute_steps_per_epoch(train_labels, BATCH_SIZE)
    train_batches = steps_per_epoch
    val_batches = (val_count + BATCH_SIZE - 1) // BATCH_SIZE
    test_batches = (len(test_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Streaming dataset configured with input_shape: {input_shape}")
    print(f"Train batches: {train_batches}, Validation batches: {val_batches}, Test batches: {test_batches}")

    if USE_BASELINE:
        print_stage_banner('Baseline Sanity Run')
        train_paths_small, train_labels_small = _sample_per_class(
            train_paths, train_labels, BASELINE_SAMPLES_PER_CLASS, seed=RANDOM_STATE
        )
        val_paths_small, val_labels_small = _sample_per_class(
            val_paths, val_labels, BASELINE_SAMPLES_PER_CLASS // 2, seed=RANDOM_STATE + 1
        )

        train_ds_small = build_tf_dataset(
            train_paths_small,
            train_labels_small,
            input_shape,
            BATCH_SIZE,
            shuffle=True,
            seed=RANDOM_STATE,
            augment=False,
        )
        val_ds_small = build_tf_dataset(
            val_paths_small,
            val_labels_small,
            input_shape,
            BATCH_SIZE,
        )

        baseline_model = build_baseline_model(input_shape)
        baseline_model.summary()
        baseline_model.fit(
            train_ds_small,
            epochs=BASELINE_EPOCHS,
            validation_data=val_ds_small,
        )

    model = build_audio_model(
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
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        label_smoothing=LABEL_SMOOTHING,
        output_activation='sigmoid',
    )

    if USE_FOCAL_LOSS:
        loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
            gamma=FOCAL_GAMMA,
            alpha=FOCAL_ALPHA,
        )
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy()

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        ),
        loss=loss_fn,
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ],
    )

    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=5,
        verbose=1,
        restore_best_weights=True,
    )
    model_checkpoint = ModelCheckpoint(
        f"{MODEL_NAME}-best.keras",
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    print_stage_banner('Training Started')
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_batches,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
    )

    best_epoch = int(np.argmax(history.history['val_accuracy']) + 1)
    best_val_acc = float(np.max(history.history['val_accuracy']))
    best_val_loss = float(np.min(history.history['val_loss']))

    print_stage_banner('Training Summary')
    print(f"Best epoch: {best_epoch}")
    print(f"Best val_accuracy: {best_val_acc:.4f}")
    print(f"Best val_loss: {best_val_loss:.4f}")

    test_results = model.evaluate(test_ds, verbose=1)
    test_loss = test_results[0]
    test_acc = test_results[1]
    test_auc = test_results[4] if len(test_results) > 4 else None
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    if test_auc is not None:
        print(f"Test AUC: {test_auc:.4f}")

    print_stage_banner('Label Samples (Repeat)')
    print_label_samples(np.asarray(train_paths), train_labels, sample_count=3, seed=RANDOM_STATE)

    final_model_path = f"{MODEL_NAME}.keras"
    model.save(final_model_path)
    print(f"model saved: {final_model_path}")

    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.savefig(f'{MODEL_NAME}-acc.png')

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.savefig(f'{MODEL_NAME}-loss.png')


def main():
    run_audio_pipeline()


if __name__ == '__main__':
    main()
