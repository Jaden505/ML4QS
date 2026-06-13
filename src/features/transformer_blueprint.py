"""Blueprint: PyTorch Transformer for raw sensor time-series classification.

This scaffold loads **raw windowed data** (no handcrafted features) and trains
a Transformer encoder with multi-head self-attention.  It is intended as a
starting point for future deep-learning experiments; it is **not** part of the
classical ML pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------


class SensorTransformer(nn.Module):
    """Encoder-only Transformer for multi-channel sensor classification.

    Parameters
    ----------
    n_channels : int
        Number of sensor channels (e.g., 9).
    window_size : int
        Number of time-steps per window (e.g., 250 for 2.5 s at 100 Hz).
    d_model : int
        Embedding dimension (projection from raw channels).
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    dim_feedforward : int
        Hidden dimension of the FFN inside each encoder layer.
    num_classes : int
        Number of exercise classes.
    dropout : float
        Dropout rate.
    """
    def __init__(
        self,
        n_channels: int = 9,
        window_size: int = 250,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(n_channels, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)  # average over time dimension
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x : (batch, window_size, n_channels) → logits : (batch, num_classes)"""
        x = self.input_projection(x)  # (B, T, C) → (B, T, d_model)
        x = x + self.pos_encoding
        x = self.encoder(x)           # (B, T, d_model)
        x = x.transpose(1, 2)         # (B, d_model, T) for pooling
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        return self.classifier(x)      # (B, num_classes)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _accuracy(y_pred: Tensor, y_true: Tensor) -> float:
    return (y_pred.argmax(dim=1) == y_true).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def run_transformer(
    windows: np.ndarray,
    labels: np.ndarray,
    train_participants: Optional[list[str]] = None,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 3,
    seed: int = 42,
    device_str: str = "auto",
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """Train and evaluate the Transformer on raw sensor windows.

    Parameters
    ----------
    windows : np.ndarray, shape (N, window_size, n_channels)
    labels : np.ndarray, shape (N,)
    train_participants : list of str, optional
        Participants to use for training (others become test).
        If None, uses an 80/20 random split.

    Returns
    -------
    dict with keys: train_loss, train_acc, test_loss, test_acc, num_params
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _resolve_device(device_str)
    n_classes = len(np.unique(labels))

    # --- Prepare data ---
    if train_participants is not None:
        from src.data.windowing import create_windows
        from src.data.load_data import load_data
        # Re-create with participant metadata
        df = load_data()
        w, l, meta = create_windows(df)
        # Map participants to indices
        part_array = meta["participants"]
        train_mask = np.isin(part_array, train_participants)
        X_train, y_train = w[train_mask], l[train_mask]
        X_test, y_test = w[~train_mask], l[~train_mask]
    else:
        from sklearn.model_selection import train_test_split
        n = len(windows)
        idx = np.arange(n)
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
        X_train, y_train = windows[train_idx], labels[train_idx]
        X_test, y_test = windows[test_idx], labels[test_idx]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
    )

    # --- Model ---
    _, window_size, n_channels = windows.shape
    model = SensorTransformer(
        n_channels=n_channels,
        window_size=window_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=n_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Training loop ---
    best_test_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if save_path:
                torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
                f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}"
            )

    # --- Final eval ---
    _, final_test_acc = evaluate(model, test_loader, criterion, device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": final_test_acc,
        "best_test_acc": best_test_acc,
        "num_params": num_params,
    }


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.data.load_data import load_data
    from src.data.windowing import create_windows

    print("Loading data...")
    df = load_data()
    windows, labels, meta = create_windows(df)
    print(f"Windows: {windows.shape}, Labels: {labels.shape}")

    participants = np.unique(meta["participants"])
    print(f"Participants: {list(participants)}")

    # Leave-one-participant-out: train on P1-P3, test on P4
    train_parts = [p for p in participants if p != "P4"]
    test_parts = [p for p in participants if p == "P4"]

    print(f"\nTraining on: {train_parts}")
    print(f"Testing on:  {test_parts}")

    save_path = str(CHECKPOINT_DIR / "transformer_checkpoint.pt")

    results = run_transformer(
        windows=windows,
        labels=labels,
        train_participants=train_parts,
        epochs=20 if "--quick" in sys.argv else 50,
        save_path=save_path,
    )

    print(f"\n=== Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nCheckpoint saved to: {save_path}")
