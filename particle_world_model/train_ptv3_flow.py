"""Train a PTv3 encoder-decoder to predict per-point 3D flow between timesteps.

Input:  point cloud of a single object at timestep t  (globally-normalized xyz)
Output: displacement vector Δxyz per point  (predicts position at t+1)

Each training sample is one (timestep, object) pair. All objects are treated
identically, so the dataset multiplies over objects with no per-object features.
Consumes ``tracked_particles.npz`` (produced by
``particle_world_model.run_flow``); writes the best checkpoint to
``ptv3_flow_best.pth``.

Run from the repo root::

    uv run python -m particle_world_model.train_ptv3_flow
    uv run python -m particle_world_model.train_ptv3_flow \\
        --epochs 200 --lr 5e-4
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from ptv3.ptv3 import PointTransformerV3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParticleFlowDataset(Dataset):
    """Consecutive-frame pairs from tracked_particles.npz, one object at a time.

    Each sample is (object_point_cloud_t, displacement_t→t+1) for a single object.
    Coordinates are globally normalized so the model sees absolute scene position.
    All objects are treated identically — no per-object features.
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        tracks = data["tracks_3d"].astype(np.float32)   # (T, N, 3)
        self.vis = data["pred_vis"]                       # (T, N) bool
        obj_ids = data["particle_obj_ids"]                # (N,)

        self.T = tracks.shape[0]

        # Normalize coordinates globally (preserves absolute position in scene)
        self.coord_mean = tracks.mean(axis=(0, 1), keepdims=True)   # (1,1,3)
        self.coord_std  = tracks.std(axis=(0, 1), keepdims=True) + 1e-6
        self.tracks_norm = (tracks - self.coord_mean) / self.coord_std  # (T, N, 3)

        # Group particle indices by object
        unique_ids = np.unique(obj_ids)
        self.obj_masks = [np.where(obj_ids == uid)[0] for uid in unique_ids]
        self.num_obj = len(unique_ids)

        self.in_channels = 3  # globally-normalized xyz only

    def __len__(self) -> int:
        return (self.T - 1) * self.num_obj

    def __getitem__(self, idx: int):
        t       = idx // self.num_obj
        obj_idx = idx  % self.num_obj
        mask    = self.obj_masks[obj_idx]

        coord        = torch.from_numpy(self.tracks_norm[t,     mask])   # (M, 3)
        displacement = torch.from_numpy(
            self.tracks_norm[t + 1, mask] - self.tracks_norm[t, mask]   # (M, 3)
        )
        vis = torch.from_numpy(self.vis[t][mask])                        # (M,)
        return coord, coord, displacement, vis


def collate_fn(batch):
    coords, feats, displacements, vis_masks = zip(*batch)
    offsets = []
    total = 0
    for c in coords:
        total += c.shape[0]
        offsets.append(total)
    return (
        torch.cat(coords,        dim=0),   # (B*N, 3)
        torch.cat(feats,         dim=0),   # (B*N, C)
        torch.cat(displacements, dim=0),   # (B*N, 3)
        torch.cat(vis_masks,     dim=0),   # (B*N,)
        torch.tensor(offsets, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PTv3FlowModel(nn.Module):
    """PTv3 encoder-decoder backbone with a linear displacement head."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Small architecture suited for ~140 points per scene
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 4, 2),
            enc_channels=(32, 64, 128, 256, 384),
            enc_num_head=(2, 4, 8, 16, 24),
            enc_patch_size=(48, 48, 48, 48, 48),
            dec_depths=(1, 1, 1, 1),
            dec_channels=(32, 32, 64, 128),
            dec_num_head=(2, 2, 4, 8),
            dec_patch_size=(48, 48, 48, 48),
            drop_path=0.1,
            enc_mode=False,   # enable decoder for per-point output
        )
        # decoder stage 0 output has 32 channels
        self.head = nn.Linear(32, 3)

    def forward(self, data_dict: dict) -> torch.Tensor:
        point = self.backbone(data_dict)   # per-point features at original resolution
        return self.head(point.feat)       # (N_total, 3)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_data_dict(coord, feat, offset, grid_size: float, device):
    return {
        "coord":     coord.to(device),
        "feat":      feat.to(device),
        "offset":    offset.to(device),
        "grid_size": grid_size,
    }


def run_epoch(model, loader, optimizer, device, grid_size, train: bool):
    model.train(train)
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for coord, feat, displacement, vis, offset in loader:
            displacement = displacement.to(device)
            vis          = vis.to(device)
            data_dict    = make_data_dict(coord, feat, offset, grid_size, device)

            pred = model(data_dict)           # (B*N, 3)
            # Only supervise on visible points
            mask = vis.unsqueeze(-1).expand_as(pred)
            loss = nn.functional.mse_loss(pred[mask], displacement.to(device)[mask])

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="tracked_particles.npz")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int, default=4)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out",     default="ptv3_flow_best.pth")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = ParticleFlowDataset(args.data)
    print(f"Dataset: {len(dataset)} samples  "
          f"({dataset.T - 1} timestep pairs × {dataset.num_obj} objects)  |  "
          f"in_channels={dataset.in_channels}")

    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              collate_fn=collate_fn)

    # grid_size: coords are normalized → range ≈ [-2, 2]; 0.05 gives depth≈6
    grid_size = 0.05

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PTv3FlowModel(in_channels=dataset.in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    
    val_loss   = run_epoch(model, val_loader,   optimizer, device, grid_size, train=False)
    print(f"Validation loss after initialization: {val_loss:.5f}")
    

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, grid_size, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, grid_size, train=False)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "state_dict":  model.state_dict(),
                "val_loss":    val_loss,
                "in_channels": dataset.in_channels,
                "coord_mean":  dataset.coord_mean.tolist(),
                "coord_std":   dataset.coord_std.tolist(),
                "grid_size":   grid_size,
            }, args.out)

        if epoch % 1 == 0:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"best={best_val_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\nDone. Best val MSE (normalized): {best_val_loss:.5f}")
    print(f"Checkpoint saved to: {args.out}")


if __name__ == "__main__":
    main()
