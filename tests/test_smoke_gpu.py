"""GPU smoke test: run a tiny end-to-end inference on synthetic input.

Skipped automatically when there is no CUDA device or no downloaded checkpoint.
Doubles as a minimal VRAM probe (prints peak allocation).
"""
import os

import pytest
import torch

CKPT = "checkpoints/tapip3d_final.pth"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(CKPT),
    reason="requires a CUDA GPU and the downloaded checkpoint",
)


def test_inference_smoke():
    from utils.inference_utils import load_model, inference, get_grid_queries

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = load_model(CKPT)
    model.to("cuda")

    H, W = 384, 512
    model.set_image_size((H, W))

    T = 6
    g = torch.Generator().manual_seed(0)
    video = torch.rand(T, 3, H, W, generator=g).cuda()
    depths = (torch.rand(T, H, W, generator=g) + 0.5).cuda()

    fx = fy = 300.0
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    intrinsics = K[None].repeat(T, 1, 1).cuda()
    extrinsics = torch.eye(4, dtype=torch.float32)[None].repeat(T, 1, 1).cuda()

    query = get_grid_queries(8, depths, intrinsics, extrinsics)

    # fp16 autocast: works on Pascal+ and keeps the smoke test light on VRAM.
    with torch.autocast("cuda", dtype=torch.float16):
        coords, visibs = inference(
            model=model,
            video=video,
            depths=depths,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            query_point=query,
            num_iters=2,
            grid_size=0,
        )

    assert coords.shape[0] == T
    assert visibs.shape[0] == T
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[smoke] peak VRAM: {peak:.2f} GB, coords shape {tuple(coords.shape)}")
