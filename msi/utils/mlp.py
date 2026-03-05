import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for regression tasks.

    Parameters
    ----------
    input_dim : int
        Dimension of input vectors
    hidden_dims : List[int]
        List of hidden layer dimensions
    output_dim : int
        Dimension of output vectors
    dropout : float, optional
        Dropout probability. If None or 0, no dropout is applied
    use_layer_norm : bool, optional
        Whether to use layer normalization after each hidden layer
    activation : str, optional
        Activation function to use. Options: 'relu', 'tanh', 'gelu'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: Optional[float] = None,
        use_layer_norm: bool = False,
        activation: str = "relu",
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))

            layers.append(self.activation)

            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

        self.y_mean = None
        self.y_std = None

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        validation_split: float = 0.2,
        clip_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        patience: Optional[int] = None,
        verbose: bool = True,
        plot_history: bool = False,
        device: str = "cpu",
        standardize_labels: bool = True,
    ) -> dict:
        """
        Train the MLP on the provided data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, input_dim)
        y : np.ndarray
            Target data of shape (n_samples, output_dim)
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        num_epochs : int
            Number of training epochs
        validation_split : float
            Fraction of data to use for validation (0-1)
        clip_grad_norm : float, optional
            Maximum norm for gradient clipping. If None, no clipping is applied
        weight_decay : float
            L2 regularization coefficient
        patience : int, optional
            Early stopping patience. If None, no early stopping
        verbose : bool
            Whether to print training progress
        plot_history : bool
            Whether to plot training and validation loss curves
        device : str
            Device to train on ('cpu' or 'cuda')
        standardize_labels : bool
            Whether to standardize the labels (y) before training

        Returns
        -------
        history : dict
            Dictionary containing training history with keys:
            'train_loss', 'val_loss', 'epoch'
        """
        self.to(device)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # split into train and validation
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
        X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

        if standardize_labels:
            self.y_mean = torch.mean(y_train, dim=0)
            self.y_std = torch.std(y_train, dim=0)
            self.y_std[self.y_std == 0] = 1.0  # Avoid division by zero

            y_train = (y_train - self.y_mean) / self.y_std
            y_val = (y_val - self.y_mean) / self.y_std
        else:
            self.y_mean = None
            self.y_std = None

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # loss and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        history = {"train_loss": [], "val_loss": [], "epoch": []}

        # early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        # training
        pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
        for epoch in pbar:
            self.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)

                loss.backward()

                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)

                optimizer.step()

                train_losses.append(loss.item())

            # validation
            self.eval()
            with torch.no_grad():
                X_val_device = X_val.to(device)
                y_val_device = y_val.to(device)
                val_outputs = self(X_val_device)
                val_loss = loss_fn(val_outputs, y_val_device).item()

            # history
            train_loss = np.mean(train_losses)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["epoch"].append(epoch + 1)

            pbar.set_postfix({"train_loss": f"{train_loss:.6f}", "val_loss": f"{val_loss:.6f}"})

            # early stopping
            if patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        pbar.write(f"Early stopping at epoch {epoch + 1}")
                        break

        pbar.close()

        if plot_history:
            self._plot_history(history)

        return history

    def _plot_history(self, history: dict) -> None:
        """
        Plot training and validation loss curves.

        Parameters
        ----------
        history : dict
            Training history dictionary
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history["epoch"], history["train_loss"], label="Training Loss", linewidth=2)
        plt.plot(history["epoch"], history["val_loss"], label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss (MSE)", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def predict(self, X, device: str = "cpu"):
        """
        Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input data of shape (n_samples, input_dim)
        device : str
            Device to use for prediction

        Returns
        -------
        predictions : np.ndarray or torch.Tensor
            Predicted outputs of shape (n_samples, output_dim). Returns tensor if input is tensor,
            numpy array otherwise.
        """
        self.to(device)
        self.eval()

        is_tensor = isinstance(X, torch.Tensor)

        with torch.no_grad():
            if is_tensor:
                X_tensor = X.to(device)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

            predictions = self(X_tensor)

            if getattr(self, "y_mean", None) is not None:
                mean = self.y_mean.to(device)
                std = self.y_std.to(device)
                predictions = predictions * std + mean

            if not is_tensor:
                predictions = predictions.cpu().numpy()

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray, device: str = "cpu") -> float:
        """
        Compute the MSE score on the provided data.

        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray
            True targets
        device : str
            Device to use

        Returns
        -------
        mse : float
            Mean squared error
        """
        predictions = self.predict(X, device=device)
        mse = np.mean((predictions - y) ** 2)
        return mse
