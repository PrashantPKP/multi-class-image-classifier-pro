import argparse
from pathlib import Path
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras


def find_data_dir():
    candidates = [
        Path.cwd() / "traning_data" / "train",
        Path.cwd() / "train",
        Path.cwd().parent / "traning_data" / "train",
        Path.cwd().parent / "train",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find training data directory. Expected a 'traning_data/train' folder.")


def build_model(num_classes: int):
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.layers.Input(shape=(224, 224, 3))
    x = keras.layers.RandomFlip("horizontal")(inputs)
    x = keras.layers.RandomRotation(0.1)(x)
    x = keras.layers.RandomZoom(0.1)(x)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main(args):
    data_dir = find_data_dir()
    print("Using training data from:", data_dir)

    batch_size = args.batch_size
    img_size = (224, 224)
    import random

    class_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    class_names = [d.name for d in class_dirs]

    models_dir = Path.cwd() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    class_names_path = models_dir / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print("Saved class mapping to:", class_names_path)

    train_files = []
    train_labels = []
    val_files = []
    val_labels = []
    rng = random.Random(123)
    for idx, d in enumerate(class_dirs):
        class_files = []
        for f in d.glob('*'):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                class_files.append(str(f))

        rng.shuffle(class_files)
        if len(class_files) <= 1:
            train_files.extend(class_files)
            train_labels.extend([idx] * len(class_files))
            continue

        split_index = max(1, int(len(class_files) * 0.2))
        val_subset = class_files[:split_index]
        train_subset = class_files[split_index:]

        val_files.extend(val_subset)
        val_labels.extend([idx] * len(val_subset))
        train_files.extend(train_subset)
        train_labels.extend([idx] * len(train_subset))

    num_total = len(train_files) + len(val_files)

    print(f"Found {num_total} files belonging to {len(class_names)} classes.")
    print(f"Training files: {len(train_files)} | Validation files: {len(val_files)}")

    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    def decode_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)
        return img

    def process_path(path, label):
        return decode_img(path), tf.cast(label, tf.int32)

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.map(lambda p, l: process_path(p, l), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(lambda p, l: process_path(p, l), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    # Prepare or resume model
    out_dir = Path.cwd() / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)

    if args.resume and output_path.exists():
        print("Resuming training from existing model:", output_path)
        model = keras.models.load_model(output_path)
        # sanity-check output shape
        if model.output_shape[-1] != num_classes:
            raise RuntimeError(f"Model output shape ({model.output_shape[-1]}) does not match dataset classes ({num_classes}).")
        # use a smaller learning rate when resuming
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("Loaded and compiled model for resume.\n")
        initial_epochs = 0
        model.summary()
    else:
        model = build_model(num_classes)
        model.summary()

    labels_array = np.array(train_labels)
    class_counts = np.bincount(labels_array)
    max_count = max(class_counts.max(), 1)
    class_weights = {idx: float(max_count / count) if count else 1.0 for idx, count in enumerate(class_counts)}
    print("Class weights:", class_weights)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_path),
            save_best_only=True,
            monitor="val_accuracy",
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    # If we built a fresh model, follow the freeze->fine-tune schedule; if resuming, just continue training
    if not (args.resume and output_path.exists()):
        # initial frozen training
        initial_epochs = min(4, args.epochs)
        if initial_epochs > 0:
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=initial_epochs,
                callbacks=callbacks,
                class_weight=class_weights,
            )

        # unfreeze and fine-tune
        try:
            base_model = model.get_layer("mobilenetv2_1.00_224")
            base_model.trainable = True
            for layer in base_model.layers[:-20]:
                layer.trainable = False
        except ValueError:
            # layer name may differ across TF versions; attempt best-effort unfreeze
            for layer in model.layers:
                if isinstance(layer, keras.Model):
                    for l in layer.layers[-20:]:
                        l.trainable = True

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    # Continue training for the requested total epochs (if resuming, this will be additional epochs)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        initial_epoch=0,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # backup existing model if requested
    final_path = output_path
    if args.backup and final_path.exists():
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = final_path.with_name(f"intel_image_classification_model.{stamp}.keras")
        final_path.replace(backup_path)
        print("Backed up previous model to:", backup_path)

    if final_path.exists():
        print("Best model saved to:", final_path)
    else:
        model.save(final_path)
        print("Saved trained model to:", final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-class image classifier on local data")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing model file instead of creating a new one")
    parser.add_argument("--output", type=str, default="models/intel_image_classification_model.keras", help="Path to save the trained model")
    parser.add_argument("--backup", action="store_true", help="Backup existing model before overwriting (adds timestamp)")
    args = parser.parse_args()
    main(args)
