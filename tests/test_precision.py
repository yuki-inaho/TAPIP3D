"""Tests for the GPU-aware autocast selection added for Pascal/older GPUs."""
import contextlib

import pytest
import torch

from inference import resolve_autocast


def test_cpu_returns_nullcontext():
    ctx = resolve_autocast("auto", "cpu")
    assert isinstance(ctx, contextlib.nullcontext)


def test_unknown_precision_raises():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to reach the precision branch")
    with pytest.raises(ValueError):
        resolve_autocast("float8", "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_auto_never_selects_unsupported_bf16():
    # On a GPU without bf16 support (e.g. GTX 1070) auto must not blow up and
    # must produce a usable autocast context.
    ctx = resolve_autocast("auto", "cuda")
    with ctx:
        x = torch.randn(4, 4, device="cuda")
        y = x @ x
    assert y.shape == (4, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp32_disables_autocast():
    ctx = resolve_autocast("fp32", "cuda")
    with ctx:
        assert not torch.is_autocast_enabled()
