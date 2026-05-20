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
import torch
import pybullet as p

from predicators import utils
from predicators.envs import gymnasium_wrapper as robodisco


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _vp_matrix(view_matrix, proj_matrix):
    vm = np.array(view_matrix).reshape(4, 4).T
    pm = np.array(proj_matrix).reshape(4, 4).T
    return pm @ vm


def project_3d_to_2d(points_3d, view_matrix, proj_matrix, width, height):
    """World-coordinate points (N,3) → pixel coordinates (N,2)."""
    vp = _vp_matrix(view_matrix, proj_matrix)
    ones = np.ones((len(points_3d), 1))
    pts_h = np.concatenate([points_3d, ones], axis=1)  # (N, 4)
    clip = pts_h @ vp.T                                  # (N, 4)
    ndc_x = clip[:, 0] / clip[:, 3]
    ndc_y = clip[:, 1] / clip[:, 3]
    px = (ndc_x + 1.0) * width / 2.0
    # NDC y=+1 → top row (v=0); NDC y=-1 → bottom row (v=height)
    py = height - (ndc_y + 1.0) * height / 2.0
    return np.stack([px, py], axis=1)


def backproject_2d_to_3d(pixels_xy, depth_map, view_matrix, proj_matrix,
                          width, height):
    """Tracked pixel coordinates (N,2) + depth buffer → world points (N,3).

    Uses the same NDC convention as utils.get_particles_from_rgbd_and_matrices.
    """
    inv_vp = np.linalg.inv(_vp_matrix(view_matrix, proj_matrix))
    u = np.clip(np.round(pixels_xy[:, 0]).astype(int), 0, width - 1)
    v = np.clip(np.round(pixels_xy[:, 1]).astype(int), 0, height - 1)
    depth = depth_map[v, u]
    ndc_x = 2.0 * u / width - 1.0
    ndc_y = 2.0 * (height - v) / height - 1.0
    ndc_z = 2.0 * depth - 1.0
    pts = np.stack([ndc_x, ndc_y, ndc_z, np.ones_like(ndc_x)], axis=1)
    world = pts @ inv_vp.T
    world /= world[:, 3:]
    return world[:, :3]


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

def run_flow():
    utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})
    robodisco.register_all_environments()
    env = robodisco.make("robodisco/Airport-v0", render_mode="human")
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

    # ── Phase 2: CoTracker v3 (offline) ──────────────────────────────────
    print("Phase 2: Loading CoTracker v3 … (downloads checkpoint on first run)")
    try:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline",
            trust_repo=True, verbose=False)
    except Exception as exc:
        print(f"Failed to load CoTracker: {exc}")
        print("Install: pip install git+https://github.com/facebookresearch/co-tracker.git")
        env.close()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cotracker = cotracker.to(device).eval()

    # Build video tensor (1, T, 3, H, W)
    video = (torch.tensor(np.stack(rgb_frames, axis=0))   # (T, H, W, 3)
             .permute(0, 3, 1, 2)[None].float().to(device))

    # Build query tensor (1, N, 3) — each row is (frame_t, x_pixel, y_pixel)
    vm0, pm0, w0, h0 = cam_params[0]
    query_px = project_3d_to_2d(init_pts_3d, vm0, pm0, w0, h0)   # (N, 2)
    queries  = torch.zeros(1, N, 3, dtype=torch.float32, device=device)
    queries[0, :, 0] = 0.0
    queries[0, :, 1] = torch.tensor(query_px[:, 0], dtype=torch.float32)
    queries[0, :, 2] = torch.tensor(query_px[:, 1], dtype=torch.float32)

    print(f"  Running CoTracker on {T} frames with {N} query points …")
    with torch.no_grad():
        pred_tracks, pred_vis = cotracker(video, queries=queries)
    # pred_tracks : (1, T, N, 2)
    # pred_vis    : (1, T, N) or (1, T, N, 1)

    pred_tracks = pred_tracks[0].cpu().numpy()   # (T, N, 2)
    pred_vis    = pred_vis[0].cpu().numpy()      # (T, N[, 1])
    if pred_vis.ndim == 3:
        pred_vis = pred_vis[..., 0]              # (T, N)
    print(f"  → tracks shape: {pred_tracks.shape}")

    # ── Phase 3: Backproject 2D tracks → 3D ──────────────────────────────
    print("Phase 3: Backprojecting 2D tracks to 3D …")
    tracks_3d = np.stack([
        backproject_2d_to_3d(pred_tracks[t], depth_maps[t],
                              *cam_params[t])
        for t in range(T)
    ], axis=0)   # (T, N, 3)

    # ── Phase 4: Animated display in PyBullet GUI ─────────────────────────
    print("Phase 4: Replaying with tracked particles …")

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

    # ── Phase 5: Save video with 2D trajectory overlay ────────────────────
    # getCameraImage does not include PyBullet debug items, so we draw
    # trajectories directly onto the stored RGB frames using PIL.
    # pred_tracks is already in pixel space, so no reprojection is needed.
    print("Phase 5: Rendering trajectory video …")
    import imageio
    from PIL import Image, ImageDraw

    dot_radius = 4
    video_frames_out: list = []

    for t in range(T):
        img  = Image.fromarray(rgb_frames[t])
        draw = ImageDraw.Draw(img)

        for n in range(N):
            r, g, b = (init_colors[n] * 255).astype(np.uint8)
            color = (int(r), int(g), int(b))

            # Accumulated trail: draw all segments from frame 0 up to t.
            for s in range(t):
                if pred_vis[s, n] > 0.5 and pred_vis[s + 1, n] > 0.5:
                    x0, y0 = pred_tracks[s,     n]
                    x1, y1 = pred_tracks[s + 1, n]
                    draw.line([(x0, y0), (x1, y1)], fill=color, width=2)

            # Current-frame position dot.
            if pred_vis[t, n] > 0.5:
                cx, cy = pred_tracks[t, n]
                draw.ellipse(
                    [(cx - dot_radius, cy - dot_radius),
                     (cx + dot_radius, cy + dot_radius)],
                    fill=color)

        video_frames_out.append(np.array(img))

    video_path = "tracked_particles.mp4"
    fps = max(1, int(1 / frame_delay))
    imageio.mimsave(video_path, video_frames_out, fps=fps)
    print(f"  → saved {video_path}  ({T} frames @ {fps} fps)")

    if p.isConnected(physicsClientId=physics_client):
        print("Trajectories displayed. Rotate camera as needed. Close window to exit.")
        while p.isConnected(physicsClientId=physics_client):
            time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    run_flow()
