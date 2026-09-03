"""Roll out a trained PTv3 flow model on the Airport environment's initial scene.

Loads a checkpoint from ``train_ptv3_flow``, extracts particles from the Airport
env's first frame, and autoregressively advances each object's particle cloud
for N steps using the model's predicted per-point displacements (no physics).

Produces:
  rollout_ptv3.mp4  — the predicted trajectories overlaid on the initial frame

With --human:
  Also animates the rollout live in the PyBullet GUI with trajectory trails.

Run from the repo root (PTv3 needs the ``predicators2`` env for spconv /
flash-attn)::

    conda run -n predicators2 python -m particle_world_model.rollout_ptv3_flow
    conda run -n predicators2 python -m particle_world_model.rollout_ptv3_flow \\
        --human --steps 60
    conda run -n predicators2 python -m particle_world_model.rollout_ptv3_flow \\
        --weights other.pth
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn

from ptv3.ptv3 import PointTransformerV3
from particle_world_model import particles, tracking
from predicators import utils
from predicators.envs import gymnasium_wrapper as robodisco


# ── Model (must match train_ptv3_flow.py exactly) ─────────────────────────────

class PTv3FlowModel(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
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
            enc_mode=False,
        )
        self.head = nn.Linear(32, 3)

    def forward(self, data_dict: dict) -> torch.Tensor:
        point = self.backbone(data_dict)
        return self.head(point.feat)


# ── Rollout ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_model(model, pts_per_obj, coord_mean, coord_std,
                  num_steps, grid_size, device):
    """Roll out the model from initial world-coord particle positions.

    pts_per_obj : list of (M_i, 3) float32 numpy arrays (one per object)
    coord_mean  : (3,) float32 numpy array
    coord_std   : (3,) float32 numpy array

    Returns list of (num_steps+1, M_i, 3) world-coord arrays, one per object.
    The first frame [0] is the actual initial position from the environment.
    """
    model.eval()
    cm = torch.from_numpy(coord_mean).float().to(device)  # (3,)
    cs = torch.from_numpy(coord_std).float().to(device)   # (3,)

    # Normalize to model's coordinate space
    current = [
        (torch.from_numpy(pts).float().to(device) - cm) / cs
        for pts in pts_per_obj
    ]

    # Trajectories in world coords; start with the actual initial positions
    trajs = [[pts.copy()] for pts in pts_per_obj]

    for _ in range(num_steps):
        coords_cat = torch.cat(current, dim=0)   # (sum M_i, 3)
        offsets, total = [], 0
        for c in current:
            total += c.shape[0]
            offsets.append(total)
        offset_t = torch.tensor(offsets, dtype=torch.long, device=device)

        data_dict = {
            "coord":     coords_cat,
            "feat":      coords_cat,
            "offset":    offset_t,
            "grid_size": grid_size,
        }
        displacements = model(data_dict)   # (sum M_i, 3)

        new_current, start = [], 0
        for i, end in enumerate(offsets):
            new_c = current[i] + displacements[start:end]
            new_current.append(new_c.detach())
            world = (new_c * cs + cm).cpu().numpy()
            trajs[i].append(world)
            start = end
        current = new_current

    return [np.stack(traj, axis=0) for traj in trajs]




# ── PyBullet GUI display ───────────────────────────────────────────────────────

def display_rollout_pybullet(trajs, colors_per_obj, physics_client,
                              frame_delay=0.08):
    import pybullet as p
    T = trajs[0].shape[0]
    pos_ids, trail_ids = [], []

    for step in range(T):
        for did in pos_ids:
            p.removeUserDebugItem(did, physicsClientId=physics_client)
        pos_ids.clear()

        if step > 0:
            for traj, col in zip(trajs, colors_per_obj):
                for n in range(traj.shape[1]):
                    did = p.addUserDebugLine(
                        traj[step - 1, n].tolist(),
                        traj[step,     n].tolist(),
                        col, lineWidth=1.5,
                        physicsClientId=physics_client)
                    trail_ids.append(did)

        all_pts  = np.concatenate([traj[step] for traj in trajs], axis=0)
        all_cols = np.concatenate(
            [np.tile(col, (traj.shape[1], 1))
             for traj, col in zip(trajs, colors_per_obj)], axis=0)
        did = p.addUserDebugPoints(
            all_pts.tolist(), all_cols.tolist(),
            pointSize=10.0, physicsClientId=physics_client)
        pos_ids.append(did)

        time.sleep(frame_delay)

    for did in pos_ids:
        p.removeUserDebugItem(did, physicsClientId=physics_client)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="ptv3_flow_best.pth")
    parser.add_argument("--steps",   type=int, default=60,
                        help="Number of rollout steps")
    parser.add_argument("--human",   action="store_true",
                        help="Open PyBullet GUI and animate rollout live")
    parser.add_argument("--out",     default="rollout_ptv3.mp4",
                        help="Output path for the overlay animation")
    parser.add_argument("--obj-ids", type=int, nargs="+", default=[9, 10],
                        help="PyBullet body ids of the objects to roll out")
    args = parser.parse_args()

    # ── Initialize Airport environment ───────────────────────────────────────
    print("Initializing Airport environment …")
    utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})
    robodisco.register_all_environments()
    render_mode = "human" if args.human else "rgb_array"
    env = robodisco.make("robodisco/Airport-v0", render_mode=render_mode)
    pybullet_env   = env.unwrapped._env
    physics_client = pybullet_env._physics_client_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"Loading weights from {args.weights} …")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", 3)
    coord_mean  = np.array(ckpt["coord_mean"], dtype=np.float32).reshape(3)
    coord_std   = np.array(ckpt["coord_std"],  dtype=np.float32).reshape(3)
    grid_size   = float(ckpt["grid_size"])
    print(f"  epoch={ckpt.get('epoch')}  val_loss={ckpt.get('val_loss', float('nan')):.5f}  "
          f"in_channels={in_channels}  grid_size={grid_size}")

    model = PTv3FlowModel(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ── Get initial particles ─────────────────────────────────────────────────
    print("Getting initial scene particles …")
    env.reset()
    env.step(np.zeros(env.action_space.shape))

    rgb_arr, depth_arr, seg_arr, vm, pm, w, h = particles.capture_frame(
        pybullet_env)
    particles_by_id = particles.get_particles_from_rgbd_and_matrices(
        rgb_arr, depth_arr, seg_arr, vm, pm)

    pts_list, col_list = [], []
    for obj in pybullet_env._objects:
        if obj.id in particles_by_id and obj.id in args.obj_ids:
            pts, cols = particles_by_id[obj.id]
            pts, cols = particles.downsample_particles(pts, cols)
            pts_list.append(pts)
            col_list.append(cols)
            print(f"  obj id={obj.id}: {len(pts)} particles")

    if not pts_list:
        print(f"No particles found for objects {args.obj_ids}; exiting.")
        env.close()
        return

    # ── Rollout ───────────────────────────────────────────────────────────────
    print(f"Rolling out {len(pts_list)} object(s) for {args.steps} steps …")
    trajs = rollout_model(
        model, pts_list, coord_mean, coord_std,
        num_steps=args.steps, grid_size=grid_size, device=device)
    # trajs: list of (steps+1, M_i, 3) world-coord arrays

    for i, traj in enumerate(trajs):
        disp = np.linalg.norm(traj[-1] - traj[0], axis=-1).mean()
        print(f"  obj {i}: shape={traj.shape}  "
              f"mean displacement over rollout={disp:.4f} m")

    # ── Visualize: 2D overlay on initial PyBullet scene ──────────────────────
    obj_colors = [
        [0.0, 0.5, 1.0],   # blue
        [1.0, 0.4, 0.0],   # orange
        [0.0, 0.75, 0.0],  # green
        [0.8, 0.0, 0.8],   # purple
    ][:len(trajs)]

    T = trajs[0].shape[0]
    tracks_3d = np.concatenate(trajs, axis=1)           # (T, N_total, 3)
    N_total = tracks_3d.shape[1]
    visibility = np.ones((T, N_total), dtype=np.float32)
    particle_colors = np.concatenate([
        np.tile(obj_colors[i], (trajs[i].shape[1], 1))
        for i in range(len(trajs))
    ], axis=0).astype(np.float32)                       # (N_total, 3)

    print(f"Saving 2D overlay animation to {args.out} …")
    tracking.save_tracked_particles_video(
        rgb_frames=[rgb_arr] * T,
        tracks_3d=tracks_3d,
        visibility=visibility,
        cam_params=[(vm, pm, w, h)] * T,
        output_path=args.out,
        colors=particle_colors,
        fps=12,
    )

    # ── Visualize: PyBullet GUI ───────────────────────────────────────────────
    if args.human:
        import pybullet as p
        print("Animating rollout in PyBullet GUI …")
        env.reset()  # reset to show clean initial scene
        display_rollout_pybullet(trajs, obj_colors, physics_client, frame_delay=0.08)
        print("Rollout complete. Trajectory trails remain in the GUI. "
              "Close the window to exit.")
        while p.isConnected(physicsClientId=physics_client):
            time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    main()
