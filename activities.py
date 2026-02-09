# activities.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from temporalio import activity

# ======================
# Dataset paths
# ======================
X_PATH = "/home/exouser/02_triangular_mesh_autoencoder/Dataset/LDC/skelneton_lid_driven_cavity_X.npz"
Y_PATH = "/home/exouser/02_triangular_mesh_autoencoder/Dataset/LDC/skelneton_lid_driven_cavity_Y.npz"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# Dataset
# ======================
class LDCDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

        _, _, H, W = self.X.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, H),
            torch.linspace(0.0, 1.0, W),
            indexing="ij",
        )
        self.coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        self.H, self.W = H, W

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.coords, self.Y[idx]


def ldc_collate_fn(batch):
    Xs = [b[0] for b in batch]
    Ys = [b[2] for b in batch]
    coords = batch[0][1]
    return torch.stack(Xs), coords, torch.stack(Ys)

# ======================
# DeepONet model
# ======================
class BranchNet(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class TrunkNet(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, coords):
        return self.net(coords)


class DeepONet(nn.Module):
    def __init__(self, in_channels, latent_dim, out_channels, H, W):
        super().__init__()
        self.branch = BranchNet(in_channels, latent_dim)
        self.trunk = TrunkNet(latent_dim)
        self.final = nn.Linear(latent_dim, out_channels)
        self.H, self.W = H, W

    def forward(self, x, coords):
        B = x.size(0)
        b = self.branch(x)
        t = self.trunk(coords)
        fused = torch.einsum("bd,nd->bnd", b, t)
        out = self.final(fused)
        return out.permute(0, 2, 1).reshape(B, -1, self.H, self.W)

# ======================
# Temporal Activity
# ======================
@activity.defn
def train_deeponet(epochs: int = 50, batch_size: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load data ----
    X = np.load(X_PATH)["data"]
    Y = np.load(Y_PATH)["data"]

    dataset = LDCDataset(X, Y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ldc_collate_fn,
    )

    # ---- Model ----
    model = DeepONet(
        in_channels=X.shape[1],
        latent_dim=256,
        out_channels=Y.shape[1],
        H=dataset.H,
        W=dataset.W,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    coords_device = dataset.coords.to(device)

    # ======================
    # Training loop
    # ======================
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, _, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb, coords_device)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} | Avg Loss: {total_loss / len(loader):.6e}")

    # ======================
    # Inference (single sample)
    # ======================
    model.eval()
    x_sample, coords, y_true = dataset[0]

    with torch.no_grad():
        pred = model(
            x_sample.unsqueeze(0).to(device),
            coords_device
        )[0]

    # ======================
    # Visualization (NUMPY ONLY)
    # ======================
    y_true_np = y_true.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    u_true = y_true_np[0]
    v_true = y_true_np[1]
    p_true = y_true_np[2]

    u_pred = pred_np[0]
    v_pred = pred_np[1]
    p_pred = pred_np[2]

    coords_np = coords.detach().cpu().numpy()
    H, W = dataset.H, dataset.W

    xx = coords_np[:, 0].reshape(H, W)
    yy = coords_np[:, 1].reshape(H, W)

    diff_u = u_pred - u_true
    diff_v = v_pred - v_true
    diff_mag = np.sqrt(diff_u**2 + diff_v**2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].streamplot(xx, yy, u_true, v_true, density=2)
    axes[0, 0].set_title("True Velocity Streamlines")
    axes[0, 0].set_aspect("equal")

    im1 = axes[0, 1].imshow(
        p_true,
        cmap="jet",
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    )
    axes[0, 1].set_title("True Pressure")
    fig.colorbar(im1, ax=axes[0, 1])

    strm = axes[0, 2].streamplot(
        xx, yy,
        u_pred, v_pred,
        density=2,
        color=diff_mag,
        cmap="viridis",
    )
    axes[0, 2].set_title("Predicted Streamlines (Velocity Error)")
    axes[0, 2].set_aspect("equal")
    fig.colorbar(strm.lines, ax=axes[0, 2])

    axes[1, 0].streamplot(xx, yy, u_pred, v_pred, density=2, color="red")
    axes[1, 0].set_title("Predicted Velocity Streamlines")
    axes[1, 0].set_aspect("equal")

    im2 = axes[1, 1].imshow(
        p_pred,
        cmap="jet",
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    )
    axes[1, 1].set_title("Predicted Pressure")
    fig.colorbar(im2, ax=axes[1, 1])

    im3 = axes[1, 2].imshow(
        np.abs(p_pred - p_true),
        cmap="magma",
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    )
    axes[1, 2].set_title("Absolute Pressure Error")
    fig.colorbar(im3, ax=axes[1, 2])

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "ldc_deeponet_results.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path
