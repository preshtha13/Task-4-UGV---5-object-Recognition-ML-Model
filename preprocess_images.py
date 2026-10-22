
from pathlib import Path
from PIL import Image
import numpy as np

# Project directory
BASE = Path(__file__).parent
DATASET = BASE / "dataset"

# Five object classes (must match split_dataset.py)
CLASSES = [
    "apple",
    "cups",
    "oranges",
    "paper_bags",
    "water_bottle",
]

# Dataset configuration
IMAGE_SIZE = (160, 160)
SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_and_preprocess(split):
    """
    Load images for one split, resize them to 160x160,
    convert them to RGB, and normalize pixels to 0-1.

    Returns:
        X: Image array with shape (N, 160, 160, 3)
        y: Integer class labels with shape (N,)
    """
    if split not in SPLITS:
        raise ValueError(
            f"Invalid split: {split}. Choose from {SPLITS}"
        )

    images = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        class_folder = DATASET / split / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {class_folder}"
            )

        files = sorted(
            file for file in class_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        valid_count = 0

        for file in files:
            try:
                with Image.open(file) as img:
                    img = img.convert("RGB")
                    img = img.resize(
                        IMAGE_SIZE,
                        Image.Resampling.BILINEAR
                    )

                    # Convert to float32 and normalize to 0-1
                    array = np.asarray(
                        img, dtype=np.float32
                    ) / 255.0

                    images.append(array)
                    labels.append(label)
                    valid_count += 1

            except (OSError, ValueError) as error:
                print(f"Skipping {file}: {error}")

        print(
            f"{split} - {class_name}: "
            f"{valid_count} images processed"
        )

    if not images:
        raise ValueError(
            f"No valid images found in dataset/{split}"
        )

    X = np.stack(images)
    y = np.asarray(labels, dtype=np.int64)

    print(f"\n{split.upper()} RESULTS")
    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Pixel range:", X.min(), "to", X.max())

    return X, y


if __name__ == "__main__":
    # Create folder to store processed data
    PROCESSED = BASE / "processed_data"
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Process and save training, validation, and testing data
    for split_name in SPLITS:
        X, y = load_and_preprocess(split_name)

        # Save processed images and their labels
        np.save(PROCESSED / f"X_{split_name}.npy", X)
        np.save(PROCESSED / f"y_{split_name}.npy", y)

        print(f"Saved {split_name} images and labels\n")

    print("All preprocessing completed!")