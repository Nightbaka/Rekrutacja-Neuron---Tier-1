import matplotlib.pyplot as plt

def plot_images(images, labels):
    """
    Plot a list of images with their corresponding labels.
    Args:
        images (list): List of images to plot.
        labels (list): List of labels corresponding to the images.
    """
    fig, axes = plt.subplots(1, len(images), figsize=(15, 15))
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    return fig, axes
