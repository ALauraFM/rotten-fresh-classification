import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

def split_validation_to_test(data_dir: str, test_size=0.5, random_state=42):
    """
    Splits the validation set into validation and test sets.

    Args:
        data_dir (str): Path to the dataset directory (containing train/ and valid/)
        test_size (float): Proportion of validation data to move to test (default 0.5)
        random_state (int): Random state for reproducibility
    """
    data_path = Path(data_dir)

    # Paths
    valid_dir = data_path / "valid"
    test_dir = data_path / "test"

    # Create test directory structure
    if test_dir.exists():
        print(f"Test directory {test_dir} already exists. Skipping creation.")
    else:
        test_dir.mkdir(parents=True)
        print(f"Created test directory: {test_dir}")

    # Get all class subdirectories in valid
    classes = [d for d in valid_dir.iterdir() if d.is_dir()]

    for class_dir in classes:
        class_name = class_dir.name
        test_class_dir = test_dir / class_name
        test_class_dir.mkdir(exist_ok=True)

        # Get all images in this class
        images = list(class_dir.glob("*"))
        if not images:
            continue

        # Split into val and test
        val_images, test_images = train_test_split(images, test_size=test_size, random_state=random_state)

        # Move test images
        for img in test_images:
            shutil.move(str(img), str(test_class_dir / img.name))

        print(f"Class {class_name}: Moved {len(test_images)} images to test, kept {len(val_images)} in valid")

    print("Validation split completed.")

if __name__ == "__main__":
    # Load environment variables from .env
    load_dotenv()
    
    # Get data directory from .env
    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        raise ValueError("DATA_DIR not found in .env file")
    
    split_validation_to_test(data_dir)