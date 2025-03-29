import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def trainNN(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, loss_fn, optimizer, epochs=10, device=None, log_train=False, log_test=False):
    """
    Train the neural network model.
    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int, optional): Number of epochs to train. Defaults to 10.
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
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(correct / total)
        if log_train:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        # Test the model
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        correct, total, test_loss = evaluate_model(model, test_loader, loss_fn, device)

        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(correct / total)
        if log_test:
            print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%")

    return model, train_losses, test_losses, train_accuracies, test_accuracies

def evaluate_model(model: nn.Module, test_loader: DataLoader, loss_fn, device):
    """
    Evaluate the model on the test set.
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
        
