import torch
from torch.utils.data import DataLoader
import torch.nn as nn


def trainNN(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, loss_fn, optimizer, epochs: int=10, device:torch.device=None, log_train:bool=False, log_test:bool=False):
    """
    Train the neural network model.
    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        loss_fn: Loss function to use for training.
        optimizer: Optimizer to use for training.
        epochs (int, optional): Number of epochs to train. Defaults to 10.
        device (torch.device, optional): Device to run the model on. Defaults to None.
        log_train (bool, optional): Whether to log training progress. Defaults to False.
        log_test (bool, optional): Whether to log testing progress. Defaults to False.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        correct_train, total_train, train_loss = evaluate_model(model, train_loader, loss_fn, device)
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)
        if log_train:
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {100 * correct_train / total_train:.2f}%"
            )
        
        model.eval()
        correct, total, test_loss = evaluate_model(model, test_loader, loss_fn, device)
        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(correct / total)
        if log_test:
            print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%")

    return model, train_losses, test_losses, train_accuracies, test_accuracies

def evaluate_model(model: nn.Module, test_loader: DataLoader, loss_fn, device: torch.device):
    """
    Evaluate the model on the test set.
    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        loss_fn: Loss function to use for evaluation.
        device (torch.device): Device to run the model on.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct,total,test_loss
