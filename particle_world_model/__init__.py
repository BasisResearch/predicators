"""Particle world model.

RGB-D -> per-object 3D particle extraction and CoTracker-v3 video point
tracking.  This code previously lived in ``predicators.utils`` and as methods on
``predicators.envs.pybullet_env.PyBulletEnv``; it has been pulled out here so the
pipeline can be developed independently of the core ``predicators`` package.

Modules:
  * :mod:`particle_world_model.geometry`  -- camera projection / backprojection.
  * :mod:`particle_world_model.particles` -- RGB-D -> particles, downsampling,
    and PyBullet-env helpers (extract / draw / hide).
  * :mod:`particle_world_model.tracking`  -- CoTracker-v3 tracking and video
    rendering.

Demo entry points (run from the repo root, via ``uv run``)::

    uv run python -m particle_world_model.run            # live particle visualization
    uv run python -m particle_world_model.run_flow --human   # full tracking pipeline

Learned flow world-model::

    uv run python -m particle_world_model.train_ptv3_flow      # fit PTv3 on the .npz
    uv run python -m particle_world_model.rollout_ptv3_flow    # roll a trained model out
"""
