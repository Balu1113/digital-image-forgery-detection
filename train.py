"""End-to-end training pipeline for the ELA forgery detector.

Example
-------
    python train.py --real-dir data/CASIA2/Au --fake-dir data/CASIA2/Tp \
        --epochs 20 --batch-size 32 --model-out image_real_or_fake.keras

The script converts every image to an Error Level Analysis image, splits the
result into train/validation sets, trains the fusion network from
:mod:`ifdmodel` and writes the resulting model plus a training-curve plot.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model_utils import image_to_ela  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Common folder names used by public forgery datasets.
REAL_ALIASES = {"au", "real", "authentic", "original", "pristine"}
FAKE_ALIASES = {"tp", "tampered", "doctored", "fake", "forged", "manipulated"}


def _list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_dataset_dirs(data_dir=None, real_dir=None, fake_dir=None):
    """Work out the real/fake source directories from CLI options."""
    if real_dir and fake_dir:
        return Path(real_dir), Path(fake_dir)

    if not data_dir:
        raise ValueError("Provide --real-dir and --fake-dir (or --data-dir).")

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    children = {p.name.lower(): p for p in data_dir.iterdir() if p.is_dir()}
    real = next((children[a] for a in REAL_ALIASES if a in children), None)
    fake = next((children[a] for a in FAKE_ALIASES if a in children), None)

    if real is None or fake is None:
        raise ValueError(
            f"Could not find real/fake sub-folders inside {data_dir}. "
            f"Expected names like {sorted(REAL_ALIASES)} and {sorted(FAKE_ALIASES)}."
        )
    return real, fake


def build_ela_dataset(real_dir, fake_dir, work_dir, quality, val_split, seed, limit=None):
    """Convert sources to ELA and lay them out in an ImageDataGenerator tree."""
    random.seed(seed)
    work_dir = Path(work_dir)
    splits = {
        "train": {"Real": [], "Doctored": []},
        "val": {"Real": [], "Doctored": []},
    }

    for label, source in (("Real", real_dir), ("Doctored", fake_dir)):
        files = _list_images(source)
        if limit:
            files = files[:limit]
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_split))
        split_map = [(f, "val") for f in files[:n_val]] + [(f, "train") for f in files[n_val:]]
        for path, split in split_map:
            splits[split][label].append(path)

    for split, labels in splits.items():
        for label, paths in labels.items():
            out_dir = work_dir / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for path in paths:
                ela = image_to_ela(path, quality=quality)
                ela.convert("RGB").save(out_dir / f"{path.stem}.jpg", "JPEG")

    counts = {split: {label: len(paths) for label, paths in labels.items()} for split, labels in splits.items()}
    return work_dir, counts


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history.get("accuracy", []), label="train")
    axes[0].plot(history.history.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history.get("loss", []), label="train")
    axes[1].plot(history.history.get("val_loss", []), label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the ELA image forgery detector.")
    parser.add_argument("--data-dir", help="Root folder containing real/fake sub-folders.")
    parser.add_argument("--real-dir", help="Folder with authentic images.")
    parser.add_argument("--fake-dir", help="Folder with tampered images.")
    parser.add_argument("--work-dir", default="ela_dataset", help="Scratch folder for ELA images.")
    parser.add_argument("--model-out", default="image_real_or_fake.keras", help="Output model path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.3)
    parser.add_argument("--ela-quality", type=int, default=90)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Cap images per class (debugging).")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet weights.")
    parser.add_argument("--clean", action="store_true", help="Delete the work dir before converting.")
    args = parser.parse_args(argv)

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    from ifdmodel import build_fusion_model

    real_dir, fake_dir = resolve_dataset_dirs(args.data_dir, args.real_dir, args.fake_dir)
    print(f"Real images : {real_dir}")
    print(f"Fake images : {fake_dir}")

    if args.clean and Path(args.work_dir).exists():
        shutil.rmtree(args.work_dir)

    work_dir, counts = build_ela_dataset(
        real_dir,
        fake_dir,
        args.work_dir,
        quality=args.ela_quality,
        val_split=args.val_split,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps(counts, indent=2))

    datagen = ImageDataGenerator(rescale=None)
    target_size = (args.image_size, args.image_size)

    train_gen = datagen.flow_from_directory(
        work_dir / "train", target_size=target_size, batch_size=args.batch_size, class_mode="binary"
    )
    val_gen = datagen.flow_from_directory(
        work_dir / "val", target_size=target_size, batch_size=args.batch_size, class_mode="binary"
    )
    print("Class indices:", train_gen.class_indices)

    model = build_fusion_model(
        input_shape=(args.image_size, args.image_size, 3),
        learning_rate=args.learning_rate,
        pretrained=not args.no_pretrained,
    )

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        steps_per_epoch=max(1, len(train_gen)),
        validation_steps=max(1, len(val_gen)),
    )

    plot_history(history, "training_history.png")
    model.save(args.model_out)
    print(f"Saved model to {args.model_out}")

    results = model.evaluate(val_gen, verbose=1)
    print(f"Validation loss     : {results[0]:.4f}")
    print(f"Validation accuracy : {results[1]:.4f}")


if __name__ == "__main__":
    main()
