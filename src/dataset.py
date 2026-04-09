from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

def get_transforms(img_size=224):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


def get_dataloaders(data_dir: str, batch_size=32):
    data_path = Path(data_dir)

    train_tf, val_tf = get_transforms()

    train_dataset = datasets.ImageFolder(data_path / "train", transform=train_tf)
    val_dataset   = datasets.ImageFolder(data_path / "valid", transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes