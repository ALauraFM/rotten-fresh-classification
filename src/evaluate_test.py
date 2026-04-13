import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import get_test_dataloader
from src.model import build_model


def evaluate_test(model_path="best_model.pth", batch_size=32):
    load_dotenv()

    data_dir = os.getenv("DATA_DIR")
    test_dir = os.getenv("TEST_DIR")
    if not (data_dir or test_dir):
        raise ValueError("DATA_DIR or TEST_DIR must be set in .env file")
    
    root_path = Path(__file__).resolve().parent.parent
    model_path = root_path / model_path
    test_loader, classes = get_test_dataloader(data_dir=data_dir, test_dir=test_dir, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "test_classification_report.txt", "w") as f:
        f.write(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Test Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_test.png", dpi=150)
    plt.close()

    print("Test evaluation finished.")
    print(report)
    print(f"Saved report to {output_dir / 'test_classification_report.txt'}")
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix_test.png'}")


if __name__ == "__main__":
    evaluate_test()