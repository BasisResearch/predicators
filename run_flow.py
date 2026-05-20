"""3D point tracking via CoTracker v3.

Workflow:
  1. Collect RGB frames + depth maps over N simulation steps.
  2. Project initial 3D particles → 2D query pixels for CoTracker.
  3. Run CoTracker v3 (offline) to track those pixels across all frames.
  4. Backproject tracked 2D pixels + depth → consistent 3D positions.
  5. Animate tracked particles in the live PyBullet GUI, then draw trajectories.
"""

import time
import numpy as np
import pybullet as p
import argparse

from predicators import utils
from predicators.envs import gymnasium_wrapper as robodisco


# ── Camera capture helper ─────────────────────────────────────────────────────

def capture_frame(pybullet_env):
    """getCameraImage → (rgb uint8 H×W×3, depth float32 H×W, seg, vm, pm, W, H)."""
    vm, pm, w, h = pybullet_env._get_camera_matrices()
    _, _, rgb, depth, seg = p.getCameraImage(
        width=w, height=h,
        viewMatrix=vm, projectionMatrix=pm,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=pybullet_env._physics_client_id)
    rgb_arr   = np.array(rgb,   dtype=np.uint8 ).reshape((h, w, 4))[:, :, :3]
    depth_arr = np.array(depth, dtype=np.float32).reshape((h, w))
    seg_arr   = np.array(seg                   ).reshape((h, w))
    return rgb_arr, depth_arr, seg_arr, vm, pm, w, h


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # take arguments human
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true")
    args = parser.parse_args()
    
    utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})
    robodisco.register_all_environments()
    render_mode = "human" if args.human else "rgb_array"
    env = robodisco.make("robodisco/Airport-v0", render_mode=render_mode)
    pybullet_env   = env.unwrapped._env
    physics_client = pybullet_env._physics_client_id

    num_steps   = 60
    frame_delay = 0.08   # seconds between animation frames

    # ── Phase 1: Data collection ──────────────────────────────────────────
    print("Phase 1: Collecting frames …")
    env.reset()

    rgb_frames, depth_maps, cam_params = [], [], []
    init_pts_3d = init_colors = None

    for step in range(num_steps):
        env.step(np.zeros(env.action_space.shape))

        rgb_arr, depth_arr, seg_arr, vm, pm, w, h = capture_frame(pybullet_env)
        rgb_frames.append(rgb_arr)
        depth_maps.append(depth_arr)
        cam_params.append((vm, pm, w, h))

        if step == 0:
            particles_by_id = utils.get_particles_from_rgbd_and_matrices(
                rgb_arr, depth_arr, seg_arr, vm, pm)
            pts_list, col_list = [], []
            for obj in pybullet_env._objects:
                if obj.id in particles_by_id:
                    pts, cols = particles_by_id[obj.id]
                    pts, cols = utils.downsample_particles(pts, cols)
                    pts_list.append(pts)
                    col_list.append(cols)
            if not pts_list:
                print("No particles found at frame 0; exiting.")
                env.close()
                return
            init_pts_3d = np.concatenate(pts_list, axis=0)   # (N, 3)
            init_colors = np.concatenate(col_list, axis=0)   # (N, 3) in [0,1]

    N = len(init_pts_3d)
    T = num_steps
    print(f"  → {T} frames, {N} initial particles")

    # ── Phases 2 + 3: Track particles via CoTracker v3, backproject to 3D ──
    print("Phase 2+3: Running CoTracker v3 and backprojecting to 3D …")
    print("  (downloads model checkpoint on first run)")
    try:
        tracks_3d, pred_vis = utils.track_particles_in_video(
            rgb_frames, depth_maps, cam_params, init_pts_3d)
    except Exception as exc:
        print(f"Tracking failed: {exc}")
        print("Install CoTracker: pip install git+https://github.com/facebookresearch/co-tracker.git")
        env.close()
        return
    # tracks_3d : (T, N, 3); pred_vis : (T, N)

    print(f"  → tracks_3d shape: {tracks_3d.shape}")

    # ── Phase 4: Save video with 2D trajectory overlay ────────────────────
    print("Phase 4: Rendering trajectory video …")
    video_path = "tracked_particles.mp4"
    fps = max(1, int(1 / frame_delay))
    utils.save_tracked_particles_video(
        rgb_frames, tracks_3d, pred_vis, cam_params, video_path,
        colors=init_colors, fps=fps)
    print(f"  → saved {video_path}  ({T} frames @ {fps} fps)")

    # ── Phase 5: Animated display in PyBullet GUI ─────────────────────────
    print("Phase 5: Replaying with tracked particles …")

    traj_debug_ids: list = []   # accumulated — never cleared until the end
    pos_debug_ids:  list = []   # current-frame dots — cleared each step

    env.reset()
    for step in range(T):
        env.step(np.zeros(env.action_space.shape))
        if not p.isConnected(physicsClientId=physics_client):
            break

        for did in pos_debug_ids:
            p.removeUserDebugItem(did, physicsClientId=physics_client)
        pos_debug_ids.clear()

        if step > 0:
            for n in range(N):
                if pred_vis[step - 1, n] > 0.5 and pred_vis[step, n] > 0.5:
                    did = p.addUserDebugLine(
                        tracks_3d[step - 1, n].tolist(),
                        tracks_3d[step,     n].tolist(),
                        init_colors[n].tolist(), lineWidth=1.5,
                        physicsClientId=physics_client)
                    traj_debug_ids.append(did)

        vis_mask = pred_vis[step] > 0.5
        vis_pts  = tracks_3d[step][vis_mask]
        vis_cols = init_colors[vis_mask]
        if len(vis_pts):
            did = p.addUserDebugPoints(
                vis_pts.tolist(), vis_cols.tolist(),
                pointSize=10.0, physicsClientId=physics_client)
            pos_debug_ids.append(did)

        time.sleep(frame_delay)

    # Remove position dots; leave full trajectory trails in the GUI.
    for did in pos_debug_ids:
        p.removeUserDebugItem(did, physicsClientId=physics_client)

    if p.isConnected(physicsClientId=physics_client):
        print("Trajectories displayed. Rotate camera as needed. Close window to exit.")
        while p.isConnected(physicsClientId=physics_client):
            time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    main()
