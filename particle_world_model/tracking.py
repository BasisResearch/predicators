"""CoTracker-v3 video point tracking and trajectory-video rendering."""

import colorsys
from typing import List, Optional, Tuple

import imageio
import numpy as np
import PIL.Image
import torch
from numpy.typing import NDArray
from PIL import ImageDraw

from particle_world_model.geometry import backproject_2d_to_3d, project_3d_to_2d


def track_particles_in_video(
    rgb_frames: List[NDArray],
    depth_maps: List[NDArray],
    cam_params: List[Tuple],
    init_pts_3d: NDArray,
) -> Tuple[NDArray, NDArray]:
    """Track 3D particles across a video using CoTracker v3 (offline mode).

    Given a sequence of RGB frames, depth maps, and camera parameters, plus an
    initial set of 3D particle positions at frame 0, returns the consistent 3D
    positions and visibility of each particle at every frame.

    Args:
        rgb_frames:   T-length list of (H, W, 3) uint8 arrays.
        depth_maps:   T-length list of (H, W) float32 depth-buffer arrays.
        cam_params:   T-length list of (view_matrix, proj_matrix, width, height)
                      tuples, one per frame.
        init_pts_3d:  (N, 3) world-coordinate positions of the N particles to
                      track, measured at frame 0.

    Returns:
        tracks_3d:  (T, N, 3) float array of world-coordinate positions.
        visibility: (T, N) float array of per-particle visibility scores in
                    [0, 1]; values > 0.5 indicate a reliably tracked point.
    """
    T = len(rgb_frames)
    N = len(init_pts_3d)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cotracker = torch.hub.load("facebookresearch/co-tracker",
                               "cotracker3_offline",
                               trust_repo=True,
                               verbose=False)
    cotracker = cotracker.to(device).eval()

    # Build video tensor (1, T, 3, H, W).
    video = (torch.tensor(np.stack(rgb_frames, axis=0)).permute(
        0, 3, 1, 2)[None].float().to(device))

    # Build query tensor (1, N, 3): each row is (frame_t, x_pixel, y_pixel).
    vm0, pm0, w0, h0 = cam_params[0]
    query_px = project_3d_to_2d(init_pts_3d, vm0, pm0, w0, h0)  # (N, 2)
    queries = torch.zeros(1, N, 3, dtype=torch.float32, device=device)
    queries[0, :, 0] = 0.0  # all queries start at frame 0
    queries[0, :, 1] = torch.tensor(query_px[:, 0], dtype=torch.float32)
    queries[0, :, 2] = torch.tensor(query_px[:, 1], dtype=torch.float32)

    with torch.no_grad():
        pred_tracks, pred_vis = cotracker(video, queries=queries)
    # pred_tracks: (1, T, N, 2); pred_vis: (1, T, N) or (1, T, N, 1)

    pred_tracks_np = pred_tracks[0].cpu().numpy()  # (T, N, 2)
    pred_vis_np = pred_vis[0].cpu().numpy()  # (T, N[, 1])
    if pred_vis_np.ndim == 3:
        pred_vis_np = pred_vis_np[..., 0]  # (T, N)

    # Backproject each frame's 2D tracks to 3D using the per-frame depth map.
    tracks_3d = np.stack([
        backproject_2d_to_3d(pred_tracks_np[t], depth_maps[t], *cam_params[t])
        for t in range(T)
    ],
                         axis=0)  # (T, N, 3)

    return tracks_3d, pred_vis_np


def save_tracked_particles_video(
    rgb_frames: List[NDArray],
    tracks_3d: NDArray,
    visibility: NDArray,
    cam_params: List[Tuple],
    output_path: str,
    colors: Optional[NDArray] = None,
    fps: int = 12,
    dot_radius: int = 4,
    line_width: int = 2,
) -> None:
    """Save a video with particle trajectories overlaid on the RGB frames.

    Trajectories are drawn as accumulating 2D trails: at frame t the trail
    covers all segments from frame 0 up to t, so the paths grow visibly as the
    video plays.  A filled dot marks each particle's current position.

    Args:
        rgb_frames:  T-length list of (H, W, 3) uint8 arrays -- the raw frames.
        tracks_3d:   (T, N, 3) array returned by track_particles_in_video.
        visibility:  (T, N) array returned by track_particles_in_video.
                     Particles with score <= 0.5 are skipped.
        cam_params:  T-length list of (view_matrix, proj_matrix, width, height)
                     tuples used to project 3D tracks back to pixel space.
        output_path: Destination file path (e.g. "tracked_particles.mp4").
        colors:      (N, 3) float array of per-particle RGB colors in [0, 1].
                     If None, colors are sampled uniformly from the HSV wheel.
        fps:         Output video frame rate.
        dot_radius:  Radius in pixels of the current-position dot.
        line_width:  Width in pixels of the trajectory trail lines.
    """
    T, N = tracks_3d.shape[:2]

    if colors is None:
        # Evenly spaced hues so each particle gets a distinct color.
        colors = np.array([
            colorsys.hsv_to_rgb(i / max(N, 1), 0.9, 0.9) for i in range(N)
        ])

    # Project 3D tracks to 2D pixel coordinates for each frame.
    tracks_2d = np.stack(
        [project_3d_to_2d(tracks_3d[t], *cam_params[t]) for t in range(T)],
        axis=0)  # (T, N, 2)

    out_frames = []
    for t in range(T):
        img = PIL.Image.fromarray(rgb_frames[t])
        draw = ImageDraw.Draw(img)

        for n in range(N):
            r, g, b = (colors[n] * 255).astype(np.uint8)
            color = (int(r), int(g), int(b))

            # Accumulated trail up to the current frame.
            for s in range(t):
                if visibility[s, n] > 0.5 and visibility[s + 1, n] > 0.5:
                    x0, y0 = tracks_2d[s, n]
                    x1, y1 = tracks_2d[s + 1, n]
                    draw.line([(x0, y0), (x1, y1)],
                              fill=color,
                              width=line_width)

            # Current-frame position dot.
            if visibility[t, n] > 0.5:
                cx, cy = tracks_2d[t, n]
                draw.ellipse([(cx - dot_radius, cy - dot_radius),
                              (cx + dot_radius, cy + dot_radius)],
                             fill=color)

        out_frames.append(np.array(img))

    imageio.mimsave(output_path, out_frames, fps=fps)
