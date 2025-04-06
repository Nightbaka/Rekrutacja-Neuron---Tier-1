import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def plot_images(images, labels):
    """
    Plot a list of images with their corresponding labels.
    Args:
        images (list): List of images to plot.
        labels (list): List of labels corresponding to the images.
    """
    fig, axes = plt.subplots(1, len(images), figsize=(15, 15))
    if len(images) <= 1:
        axes = [axes]
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    return fig, axes

def analyze_predictions(model: Module, test_loader: DataLoader, device: torch.device):
    """
    Predict samples using model, print classification report and plot wrongly predicted images.
    Args:
        model (torch.nn.Module): Trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): Device to run the model on.
    """
    model.eval()
    wrongly_predicted = []
    wrong_labels = []
    correct_labels = []
    predicted_labels = []
    y_true = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            mask = predicted != batch_y
            wrongly_predicted.extend(batch_X[mask].cpu().numpy())
            wrong_labels.extend(predicted[mask].cpu().numpy())
            correct_labels.extend(batch_y[mask].cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            y_true.extend(batch_y.cpu().numpy())

    class_map = {0: "X", 1: "Y", 2: "Z"}
    wrongly_predicted = np.array(wrongly_predicted)
    print(
        classification_report(
            predicted_labels, y_true, target_names=["X", "Y", "Z"], digits=3
        )
    )
    # Plot wrongly predicted images
    wrongly_predicted = wrongly_predicted.reshape(-1, 28, 28)
    labels = [
        f"Prd:{class_map[pred]}, True:{class_map[true]}"
        for pred, true in zip(wrong_labels, correct_labels)
    ]
    plot_images(wrongly_predicted, labels)
    plt.show()

def plot_losses_accs(train_loss, test_loss, train_acc, test_acc):
    """
    Plot training and testing losses and accuracies.
    Args:
        train_loss (list): List of training losses.
        test_loss (list): List of testing losses.
        train_acc (list): List of training accuracies.
        test_acc (list): List of testing accuracies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    axes[0].plot(train_loss, label="Train Loss")
    axes[0].plot(test_loss, label="Test Loss")
    axes[0].set_title("Losses")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plot accuracies
    axes[1].plot(train_acc, label="Train Accuracy")
    axes[1].plot(test_acc, label="Test Accuracy")
    axes[1].set_title("Accuracies")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
