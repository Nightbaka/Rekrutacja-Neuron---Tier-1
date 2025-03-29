import pandas as pd
import torch

def to_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int = 32) -> torch.utils.data.DataLoader:
    """
    Convert a pandas dataframes to a DataLoader.
    Args:
        X (pd.DataFrame): Features dataframe.
        y (pd.Series): Labels series.
        batch_size (int, optional): Batch size. Defaults to 32.
    """

    from torch.utils.data import DataLoader, TensorDataset
    import torch

    # Create a TensorDataset
    dataset = TensorDataset(X, y)

    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

def to_tensor(X: pd.DataFrame, y: pd.Series, shape = [-1, 1, 28, 28]) -> tuple:
    """
    Convert a pandas dataframes to PyTorch tensors. Reshapes X to the shape apropriate for a CNN.
    Args:
        X (pd.DataFrame): Features dataframe.
        y (pd.Series): Labels series.
    """
    import torch

    # Convert pandas DataFrame to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    # Prep X tensor for CNN
    X_tensor = X_tensor.reshape(shape)

    return X_tensor, y_tensor

def preprocess(df: pd.DataFrame, label_col= "label",batch_size: int = 32) -> tuple:
    """
    Preprocess the data.
    Args:
        path (str): Path to the data file.
        batch_size (int, optional): Batch size. Defaults to 32.
    """
    import pandas as pd
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_tensor, y_tensor = to_tensor(X, y)
    loader = to_loader(X_tensor, y_tensor, batch_size)

    return loader
    