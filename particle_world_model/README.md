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
| `run_flow.py`  | Full 5-phase tracking pipeline (see below). |

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

## Running

Run from the repo root (module form, so `predicators` and `particle_world_model`
resolve):

```bash
python -m particle_world_model.run                # live particle visualization
python -m particle_world_model.run_flow           # pipeline, offscreen (rgb_array)
python -m particle_world_model.run_flow --human   # pipeline, PyBullet GUI
```

## Viewing the PyBullet domains

The particle pipeline runs on top of the `robodisco/Airport-v0` env. Both the
Airport and Donut domains have a `__main__` GUI entry point (same pattern as the
other concrete pybullet envs) — run from the repo root with the `predicators`
conda env active:

```bash
python predicators/envs/pybullet_airport.py   # conveyor belt, items, button/pusher
python predicators/envs/pybullet_donut.py     # donuts + target area
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
python scripts/show_airport_interaction.py
```

## Dependencies

Beyond the base `predicators` install:

- `open3d` — voxel downsampling in `downsample_particles`.
- `cotracker` (`git+https://github.com/facebookresearch/co-tracker.git`) — the
  tracker; the `cotracker3_offline` checkpoint is fetched via `torch.hub` on
  first run.

Both are listed in the top-level `setup.py`.

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
