# particle_world_model

RGB-D → per-object 3D particle extraction and CoTracker-v3 video point tracking,
built on top of the PyBullet envs in `predicators/`.

This code previously lived in `predicators.utils` and as methods on
`predicators.envs.pybullet_env.PyBulletEnv`. It was pulled out here so the
pipeline can be developed independently of the core `predicators` package. This
directory is the main place ongoing particle-world-model work happens.

## Layout

| File | Contents |
|------|----------|
| `geometry.py`  | `project_3d_to_2d`, `backproject_2d_to_3d` — camera projection math (PyBullet column-major matrix convention). The two are exact inverses at integer pixel locations. |
| `particles.py` | `get_particles_from_rgbd_and_matrices`, `downsample_particles` (pure functions on images/arrays); `capture_frame`, `extract_particles`, `hide_all_bodies`, `ParticleDebugDraw` (helpers that take a live `PyBulletEnv`). |
| `tracking.py`  | `track_particles_in_video` (CoTracker v3, offline mode), `save_tracked_particles_video` (trajectory overlay video). |
| `run.py`       | Live demo: steps the Airport env with zero actions and redraws each object's downsampled point cloud as PyBullet debug points every step. |
| `run_flow.py`  | Full 5-phase tracking pipeline (see below) → `tracked_particles.npz`. |
| `train_ptv3_flow.py`   | Trains a PTv3 flow model on `tracked_particles.npz` → `ptv3_flow_best.pth` (see [Learned flow world-model](#learned-flow-world-model)). |
| `rollout_ptv3_flow.py` | Rolls a trained flow model forward from the Airport env's initial scene → `rollout_ptv3.mp4`. |

## Two stages

1. **Tracking** (`run_flow.py`): observe the sim, track each object's particles
   through a short rollout, and dump the trajectories to `tracked_particles.npz`.
   This is ground-truth motion, not a model.
2. **Learned flow world-model** (`train_ptv3_flow.py` / `rollout_ptv3_flow.py`):
   fit a Point Transformer V3 to that data so it can predict per-point motion for
   an arbitrary object point cloud, then autoregressively roll a fresh scene
   forward with no physics.

Both stages run in the same `uv`-managed environment (see [Installation](../README.md#installation)).

## Pipeline (`run_flow.py`)

1. **Collect** RGB frames + depth maps over `num_steps` (60) simulation steps.
2. **Project** the initial frame-0 3D particles → 2D query pixels for CoTracker.
3. **Track** those pixels across all frames with CoTracker v3 (offline).
4. **Backproject** tracked 2D pixels + per-frame depth → consistent 3D positions.
5. **Replay** the tracked particles in the live PyBullet GUI and draw trajectory
   trails.

Outputs, written to the current working directory:

- `tracked_particles.npz` — `tracks_3d` `(T, N, 3)` float32, `particle_obj_ids`
  `(N,)` int32, `pred_vis` `(T, N)` float32.
- `tracked_particles.mp4` — RGB frames with accumulating 2D trajectory trails.

## Learned flow world-model

`tracked_particles.npz` is the training set for a small Point Transformer V3
(PTv3) that learns point-cloud dynamics.

**Model.** PTv3 encoder-decoder backbone (arch is small — ~140 points per scene:
`enc_channels=(32,64,128,256,384)`, `dec_channels=(32,32,64,128)`,
`enc_mode=False` so the decoder produces per-point features) with a
`Linear(32, 3)` head. Defined identically in `train_ptv3_flow.py` and
`rollout_ptv3_flow.py` — **keep the two `PTv3FlowModel` classes in sync**.

**Task.** One sample = one `(timestep, object)` pair: given an object's point
cloud at frame `t`, predict the per-point displacement Δxyz to frame `t+1`. All
objects share the model; there are no per-object features. Coordinates are
globally normalized (mean/std over the whole `npz`), so the model sees absolute
scene position. Loss is MSE on visible points only (`pred_vis > 0.5`).

### Training (`train_ptv3_flow.py`)

```bash
uv run python -m particle_world_model.train_ptv3_flow
uv run python -m particle_world_model.train_ptv3_flow \
    --epochs 200 --lr 5e-4
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--data`   | `tracked_particles.npz` | training set |
| `--epochs` | `100`  | epochs (cosine LR schedule to `1e-5`) |
| `--lr`     | `1e-3` | AdamW learning rate (`weight_decay=1e-4`) |
| `--batch`  | `4`    | scenes per batch (points concatenated with an `offset` tensor) |
| `--seed`   | `42`   | seeds torch/numpy and the 80/20 train/val split |
| `--out`    | `ptv3_flow_best.pth` | checkpoint written on every val-loss improvement |

The checkpoint stores `state_dict`, `in_channels`, the coordinate
`coord_mean` / `coord_std`, and `grid_size` (0.05) — everything the rollout
needs to reconstruct the model and un-normalize its predictions.

### Rollout (`rollout_ptv3_flow.py`)

```bash
uv run python -m particle_world_model.rollout_ptv3_flow
uv run python -m particle_world_model.rollout_ptv3_flow \
    --human --steps 60 --obj-ids 9 10
```

Boots `robodisco/Airport-v0`, extracts particles from frame 0, filters to
`--obj-ids` (default `[9, 10]` — the conveyor items), downsamples, then feeds
them through the model `--steps` times, each step adding the predicted
displacement and re-feeding. Pure model rollout — the simulator is not stepped.

| Flag | Default | Meaning |
|------|---------|---------|
| `--weights` | `ptv3_flow_best.pth` | checkpoint from training |
| `--steps`   | `60`  | autoregressive rollout steps |
| `--obj-ids` | `9 10` | PyBullet body ids to roll out |
| `--human`   | off   | also animate the rollout live in the PyBullet GUI |
| `--out`     | `rollout_ptv3.mp4` | predicted trajectories overlaid on the initial frame |

Outputs `rollout_ptv3.mp4`; with `--human` also draws growing trajectory trails
in the GUI and holds the window open until closed.

## Running

Run from the repo root (module form, so `predicators` and `particle_world_model`
resolve):

```bash
uv run python -m particle_world_model.run                # live particle visualization
uv run python -m particle_world_model.run_flow           # pipeline, offscreen (rgb_array)
uv run python -m particle_world_model.run_flow --human   # pipeline, PyBullet GUI
```

## Viewing the PyBullet domains

The particle pipeline runs on top of the `robodisco/Airport-v0` env. Both the
Airport and Donut domains have a `__main__` GUI entry point (same pattern as the
other concrete pybullet envs) — run from the repo root:

```bash
uv run python predicators/envs/pybullet_airport.py   # conveyor belt, items, button/pusher
uv run python predicators/envs/pybullet_donut.py     # donuts + target area
```

Each opens a PyBullet GUI window, loads train task 0, then holds the robot arm
still in a loop so the scene stays put and you can interact with it. `Ctrl-C` in
the terminal to quit.

Mouse controls in the GUI:

| Action | Control |
|--------|---------|
| Move a dynamic object (item / donut) | `Ctrl` + left-click-drag on the body |
| Rotate camera | `Ctrl` + left-drag on empty space |
| Pan camera | `Ctrl` + middle-drag (or `Ctrl` + right-drag) |
| Zoom | scroll wheel |
| Toggle side panels / debug overlays | `g` |

There is also a scripted Airport demo that drives the pusher against the button
instead of idling:

```bash
uv run python scripts/show_airport_interaction.py
```

## Dependencies

For the **tracking** stage, beyond the base `predicators` install:

- `open3d` — voxel downsampling in `downsample_particles`.
- `cotracker` (`git+https://github.com/facebookresearch/co-tracker.git`) — the
  tracker; the `cotracker3_offline` checkpoint is fetched via `torch.hub` on
  first run.

Both are listed (along with everything below) in the top-level `pyproject.toml`.

The **learned flow world-model** (`train_ptv3_flow.py` / `rollout_ptv3_flow.py`)
also needs the `ptv3/` package (vendored at the repo root), which pulls in
`spconv` and `flash-attn`. These used to require a separate `predicators2` conda
env because they're pinned to a specific torch/CUDA build; `uv` resolves and
installs the whole stack — torch, `spconv-cu126`, `flash-attn` (as a prebuilt
wheel, see `[tool.uv.sources]` in `pyproject.toml`), `torch-scatter`, `timm`,
`transforms3d` — into the single `.venv` from `uv sync`, so there's no second
environment to manage.

## Using it as a library

```python
from particle_world_model.particles import extract_particles, ParticleDebugDraw
from particle_world_model.tracking import track_particles_in_video

particles = extract_particles(pybullet_env)        # {Object: (points, colors)}

viz = ParticleDebugDraw(pybullet_env)
viz.display(particles, point_size=10.0)
viz.clear()

tracks_3d, visibility = track_particles_in_video(
    rgb_frames, depth_maps, cam_params, init_pts_3d)
```
