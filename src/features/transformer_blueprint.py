"""Blueprint: PyTorch Transformer for raw sensor time-series classification (future work)."""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class SensorTransformer(nn.Module):
    """Encoder-only Transformer for multi-channel sensor classification."""

    def __init__(self, n_channels=9, window_size=250, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=128, num_classes=3, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(n_channels, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x) + self.pos_encoding
        x = self.encoder(x)
        return self.classifier(x.mean(dim=1))  # mean pool over time


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def run_transformer(windows, labels, train_participants=None, batch_size=32, epochs=50, lr=1e-3,
                    d_model=64, nhead=4, num_layers=3, seed=42, device_str="auto", save_path=None):
    """Train and evaluate Transformer on raw sensor windows."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)
    n_classes = len(np.unique(labels))

    if train_participants is not None:
        from src.data.windowing import create_windows
        from src.data.load_data import load_data
        df = load_data()
        w, l, meta = create_windows(df)
        mask = np.isin(meta["participants"], train_participants)
        X_train, y_train = w[mask], l[mask]
        X_test, y_test = w[~mask], l[~mask]
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(windows))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
        X_train, y_train = windows[train_idx], labels[train_idx]
        X_test, y_test = windows[test_idx], labels[test_idx]

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
                             batch_size=batch_size, shuffle=False)

    _, window_size, n_channels = windows.shape
    model = SensorTransformer(n_channels=n_channels, window_size=window_size, d_model=d_model,
                              nhead=nhead, num_layers=num_layers, num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            if save_path: torch.save(model.state_dict(), save_path)
        if epoch % 10 == 0 or epoch in (1, epochs):
            print(f"Epoch {epoch:3d}/{epochs}  Train Acc: {train_acc:.4f}  Test Acc: {test_acc:.4f}")

    _, final_test_acc = evaluate(model, test_loader, criterion, device)
    return {"train_acc": train_acc, "test_acc": final_test_acc, "best_test_acc": best_acc,
            "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}


if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.windowing import create_windows
    df = load_data()
    windows, labels, meta = create_windows(df)
    print(f"Windows: {windows.shape}")
    train_parts = [p for p in np.unique(meta["participants"]) if p != "P4"]
    print(f"Train: {train_parts}, Test: P4")
    results = run_transformer(windows, labels, train_participants=train_parts, epochs=20,
                              save_path=str(CHECKPOINT_DIR / "transformer_checkpoint.pt"))
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
