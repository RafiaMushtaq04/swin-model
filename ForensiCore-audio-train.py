import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img

from ForensiCore import build_audio_model

MODEL_NAME = 'ForensiCore-Audio'
NUM_CLASSES = 1
PATCH_SIZE = (3, 3)
EMBED_DIM = 64
NUM_HEADS = 8
WINDOW_SIZE = 2
SHIFT_SIZE = 1
NUM_MLP = 256
QKV_BIAS = True
DROPOUT_RATE = 0.03
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
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
PRINT_EVERY = 5000

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


def compute_class_weight(labels):
    labels = np.asarray(labels)
    real_count = int(np.sum(labels == 0))
    fake_count = int(np.sum(labels == 1))
    total = real_count + fake_count
    if real_count == 0 or fake_count == 0:
        raise ValueError('Cannot compute class weights: one class has zero samples.')

    weight_for_real = total / (2.0 * real_count)
    weight_for_fake = total / (2.0 * fake_count)
    return {0: weight_for_real, 1: weight_for_fake}


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
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [target_height, target_width])
        image = tf.cast(image, tf.float32) / 255.0
        if NUM_CLASSES == 1:
            label = tf.cast(label, tf.float32)
        else:
            label = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
        return image, label

    dataset = dataset.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def run_audio_pipeline():
    print_stage_banner('Collecting Spectrogram Paths')
    train_paths, train_labels = collect_labeled_paths(TRAIN_DIR, LABELS, IMAGE_EXTS)
    val_paths, val_labels = collect_labeled_paths(VAL_DIR, LABELS, IMAGE_EXTS)
    test_paths, test_labels = collect_labeled_paths(TEST_DIR, LABELS, IMAGE_EXTS)

    input_shape = resolve_input_shape(train_paths)

    print_binary_counts('Train (before balancing)', train_labels)
    print_binary_counts('Validation (before balancing)', val_labels)
    print_binary_counts('Test (before balancing)', test_labels)

    class_weight = compute_class_weight(train_labels)
    print(f"Class weights: {class_weight}")

    print_stage_banner('Building tf.data Pipelines')
    train_ds = build_tf_dataset(
        train_paths,
        train_labels,
        input_shape,
        BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_STATE,
    )
    val_ds = build_tf_dataset(val_paths, val_labels, input_shape, BATCH_SIZE)
    test_ds = build_tf_dataset(test_paths, test_labels, input_shape, BATCH_SIZE)

    train_batches = (len(train_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    val_batches = (len(val_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    test_batches = (len(test_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Streaming dataset configured with input_shape: {input_shape}")
    print(f"Train batches: {train_batches}, Validation batches: {val_batches}, Test batches: {test_batches}")

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

    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=5,
        verbose=1,
        restore_best_weights=True,
    )
    model_checkpoint = ModelCheckpoint(
        f"{MODEL_NAME}-best.keras",
        monitor='val_accuracy',
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
        validation_data=val_ds,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        class_weight=class_weight,
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
