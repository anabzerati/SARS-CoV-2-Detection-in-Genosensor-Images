import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    data_dir: str,
    transform_train: transforms.Compose,
    transform_eval: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Load train, validation, and test datasets from directories in ImageFolder format.

    The expected folder structure is:
        data_dir/
        ├── train/
        │   ├── class1/
        │   ├── class2/
        ├── val/
        │   ├── class1/
        │   ├── class2/
        └── test/
            ├── class1/
            ├── class2/

    Args:
        data_dir (str):
            Base path to the dataset containing `train`, `val`, and `test` folders.
        transform_train (transforms.Compose):
            Transformations applied to the training images (e.g., augmentation + normalization).
        transform_eval (transforms.Compose):
            Transformations applied to validation/test images (e.g., resize + normalization only).
        batch_size (int, optional):
            Number of samples per batch (default: 32).
        num_workers (int, optional):
            Number of worker threads for data loading (default: 4).
        
    Returns:
        tuple:
            A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
            - class_names (list): List of class names (inferred from folder names).

    Raises:
        FileNotFoundError:
            If any of the expected subdirectories (`train`, `val`, or `test`) are missing.
    """

    # dataset paths 
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # verify if required directories exist 
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    # ImageFolder datasets 
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_eval)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_eval)

    class_names = train_dataset.classes

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"\nDataset loaded from {data_dir}")
    print(f"Classes ({len(class_names)}): {class_names}")

    return train_loader, val_loader, test_loader, class_names
