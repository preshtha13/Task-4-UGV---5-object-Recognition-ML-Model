
from pathlib import Path
import random
import shutil

# Project folders
BASE = Path(__file__).parent
RAW_IMAGES = BASE / "raw_images"
DATASET = BASE / "dataset"

# Five object classes
CLASSES = [
    "apple",
    "cups",
    "oranges",
    "paper_bags",
    "water_bottle",
]

# Split proportions
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Reproducible random shuffle
random.seed(42)

SPLITS = {
    "train": TRAIN_RATIO,
    "val": VAL_RATIO,
    "test": TEST_RATIO,
}


def split_dataset():
    # Check the raw images folder
    if not RAW_IMAGES.exists():
        raise FileNotFoundError(
            f"Raw images folder not found: {RAW_IMAGES}"
        )

    # Remove previous generated splits to avoid duplicate or stale files
    for split in SPLITS:
        split_path = DATASET / split
        if split_path.exists():
            shutil.rmtree(split_path)

    total_images = 0

    for class_name in CLASSES:
        class_folder = RAW_IMAGES / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_folder}"
            )

        images = [
            image for image in class_folder.iterdir()
            if image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(images) != 2000:
            raise ValueError(
                f"{class_name}: expected 2000 images, "
                f"but found {len(images)}"
            )

        random.shuffle(images)

        # Calculate split sizes
        train_end = 1400
        val_end = 1700

        split_images = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        # Copy images into their respective split folders
        for split_name, selected_images in split_images.items():
            output_folder = DATASET / split_name / class_name
            output_folder.mkdir(parents=True, exist_ok=True)

            for image in selected_images:
                shutil.copy2(image, output_folder / image.name)

            print(
                f"{class_name} - {split_name}: "
                f"{len(selected_images)} images"
            )

        total_images += len(images)

    print("\nDataset splitting completed successfully!")
    print(f"Total raw images: {total_images}")
    print("Training images: 7000")
    print("Validation images: 1500")
    print("Testing images: 1500")
    print(f"Output folder: {DATASET}")


if __name__ == "__main__":
    split_dataset()
