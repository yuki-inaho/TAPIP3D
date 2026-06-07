"""CPU-only unit tests for the pure helpers used by the inference pipeline."""
import numpy as np
import torch

from utils.inference_utils import resize_depth_bilinear, get_grid_queries


def test_resize_depth_bilinear_shape():
    depth = np.ones((10, 20), dtype=np.float32)
    # cv2 uses (width, height) ordering -> output array is (height, width)
    out = resize_depth_bilinear(depth, (40, 30))
    assert out.shape == (30, 40)
    assert np.all(out > 0)


def test_resize_depth_bilinear_all_invalid_stays_zero():
    depth = np.zeros((8, 8), dtype=np.float32)
    out = resize_depth_bilinear(depth, (16, 16))
    assert out.shape == (16, 16)
    assert np.all(out == 0.0)


def test_resize_depth_bilinear_preserves_value():
    depth = np.full((6, 6), 2.5, dtype=np.float32)
    out = resize_depth_bilinear(depth, (12, 12))
    # valid everywhere -> mean should stay close to the original value
    assert np.allclose(out.mean(), 2.5, atol=1e-3)


def test_get_grid_queries_identity_extrinsics():
    # Use a 64px image: get_points_on_a_grid leaves a margin of W/64, so smaller
    # images can round the last grid point onto the border (out of bounds).
    T, H, W = 3, 64, 64
    depths = torch.ones(T, H, W, dtype=torch.float32)
    fx = fy = 40.0
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    intrinsics = K[None].repeat(T, 1, 1)
    extrinsics = torch.eye(4, dtype=torch.float32)[None].repeat(T, 1, 1)

    q = get_grid_queries(6, depths, intrinsics, extrinsics)

    # queries: (N, 4) -> [t, x, y, z]
    assert q.ndim == 2 and q.shape[1] == 4
    assert q.shape[0] > 0
    # all queries anchored at the first frame
    assert torch.all(q[:, 0] == 0)
    # depth == 1 everywhere with identity extrinsics -> camera-frame z == 1
    assert torch.allclose(q[:, 3], torch.ones_like(q[:, 3]), atol=1e-4)
