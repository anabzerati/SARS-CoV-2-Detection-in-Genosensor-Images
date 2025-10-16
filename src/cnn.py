import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix

import os
from tqdm import tqdm
import numpy as np

from typing import Dict, Tuple, List, Optional

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    num_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
) -> Tuple[Dict, List[Dict]]:
    """
    Generic PyTorch training loop with early stopping and checkpointing.
    """

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    epoch_results = []

    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = optimizer or optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # loss and optimization
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            # one-hot-encoded label to index
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        # loss and accuracy on training data
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        
        # loss and accuracy on testing data
        avg_val_loss, val_accuracy = test(model, val_loader, criterion)
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # epoch summary
        epoch_results.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        })

        # loss analysis
        if avg_val_loss < best_val_loss: # loss improves = new best loss
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            # saving best model
            best_model_state = model.state_dict() 
            torch.save(best_model_state, f"best_model.pt")
            
            print(f"New best model saved")
        else:
            epochs_without_improvement += 1
            print(f"No improvement. Early stopping counter: {epochs_without_improvement}/{patience}")

        # early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # checkpoint after 30 epochs
        if (epoch + 1) % 30 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

            print(f"Saved checkpoint")

    return best_model_state, epoch_results

# tests model
def test(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: str,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
    complete: bool = False,
) -> Tuple[float, float]:
    """
    Evaluate model on given loader and optionally save classification results.
    """
    
    model.eval()
    all_preds, all_labels = [], []
    test_loss, test_correct, test_total = 0, 0, 0

    if criterion == None:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if labels.ndim > 1:
                labels = labels.argmax(dim=1)

            test_correct += predicted.eq(labels).sum().item()
            test_total += labels.size(0)

    avg_test_loss = test_loss / len(loader)
    test_accuracy = 100 * test_correct / test_total

    # prints and saves information
    if complete:
        print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%\n")

        classif_report = classification_report(all_labels, all_preds, digits=4, target_names=class_names)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        print("Classification Report:\n", classif_report)
        print("Confusion Matrix:\n", conf_matrix)

        os.makedirs('results', exist_ok=True)

        with open(f'results/classification_report.txt', 'w') as f:
            f.write(classif_report)

        np.savetxt(f'results/confusion_matrix.txt', conf_matrix, fmt='%d')

        print(f"\nSaved classification report and confusion matrix in 'results/' directory.")

    return avg_test_loss, test_accuracy


