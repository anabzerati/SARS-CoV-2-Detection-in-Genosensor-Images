import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import transforms

from cnn import train, test
from src.data import get_dataloaders
from src.augmentations import augmentations  

# configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
num_epochs = 300
learning_rate = 1e-4
patience = 10

weights_dir = "./model_weights"
os.makedirs(weights_dir, exist_ok=True)
os.makedirs("results", exist_ok=True)

# validation/test transform (no augmentation)
transform_eval = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# model configurations
models_to_train = [
    "resnet50",
    "resnet101",
    "densenet121",
    "convnext_tiny"
]

# main experiment loop
def run_experiment(model_name, data_dir, aug_name, aug_transform):
    """
    Trains, validates, and tests a model with a given data augmentation setup.
    """

    print(f"----- Training model: {model_name} | Augmentation: {aug_name} -----")

    # get data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        transform_train=aug_transform,
        transform_eval=transform_eval,
        batch_size=batch_size
    )
    num_classes = len(class_names)

    # create model
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    best_model_state, epoch_results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        patience=patience,
        device=device
    )

    # load best model for testing
    model.load_state_dict(best_model_state)
    print(f"\n ---- Testing final model: {model_name} ({aug_name}) -----")

    test(
        model=model,
        loader=test_loader,
        class_names=class_names,
        criterion=criterion,
        complete=True,
        device=device
    )

    # save training history
    results_file = f"results/{model_name}_{aug_name}_epochs.json"
    with open(results_file, "w") as f:
        json.dump(epoch_results, f, indent=4)
    print(f"Epoch results saved to {results_file}")

if __name__ == "__main__":
    data_dir = "../dataset/split_orig"

    for model_name in models_to_train:
        for aug_name, aug_transform in augmentations.items():
          run_experiment(model_name, data_dir, aug_name, aug_transform)
