"""Tests for 2DTS training control logic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_2dts_rasterizer_backward_accepts_expanded_gradients():
    from msbench.vendor._cmod import d2ts_rasterization as d2ts

    if d2ts._C is None:
        pytest.skip("2DTS CUDA extension is not built")

    eye = torch.eye(4, device="cuda")
    settings = d2ts.TriangleRasterizationSettings(
        image_width=32, image_height=32, tanfovx=1.0, tanfovy=1.0,
        viewmatrix=eye, projmatrix=eye, campos=torch.zeros(3, device="cuda"),
        sh_degree=0, gamma=1.0, background_depth=10.0,
        background=torch.zeros(3, device="cuda"), back_culling=False,
        rich_info=True, sort_level=0, debug=False,
    )
    vertices = torch.tensor(
        [[[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]]],
        device="cuda", requires_grad=True,
    )
    features = torch.ones(1, 3, device="cuda", requires_grad=True)
    opacity = torch.full((1, 1), 0.9, device="cuda", requires_grad=True)
    image, *_ = d2ts.TriangleRasterizer(settings)(
        vertices, torch.zeros_like(vertices), opacity, feature=features,
    )

    assert image.max() > 0
    # sum() supplies an expanded, zero-stride gradient to the CUDA wrapper.
    image.sum().backward()
    for tensor in (vertices, features, opacity):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_2dts_model_enforce_max_primitives_prunes_optimizer_state():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")

    from msbench.vendor.d2ts.diff_recon.models.VanillaTS_model import VanillaTSModel

    model = VanillaTSModel.__new__(VanillaTSModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(model_update=SimpleNamespace(max_primitives=2))
    model.use_vertex_color = False
    model._vertex = torch.nn.Parameter(torch.arange(36, dtype=torch.float32).view(4, 3, 3))
    model._opacity = torch.nn.Parameter(torch.zeros(4, 1))
    model._f_dc = torch.nn.Parameter(torch.zeros(4, 1, 3))
    model._f_rest = torch.nn.Parameter(torch.zeros(4, 1, 3))
    model.gradient_accum = torch.zeros(4)
    model.gradient_denom = torch.zeros(4)
    model.max_radii2D = torch.zeros(4)
    model.contrib_sum = torch.zeros(4)
    model.contrib_max = torch.tensor([0.0, 10.0, 1.0, 5.0])
    model.contrib_denom = torch.zeros(4)
    model.opacity_floor = 0.0
    model.optimizer = torch.optim.Adam(
        [
            {"params": [model._vertex], "lr": 0.0, "name": "vertex"},
            {"params": [model._opacity], "lr": 0.0, "name": "opacity"},
            {"params": [model._f_dc], "lr": 0.0, "name": "f_dc"},
            {"params": [model._f_rest], "lr": 0.0, "name": "f_rest"},
        ],
        lr=0.0,
    )

    removed = model.enforce_max_primitives()

    assert removed == 2
    assert model.get_vertex.shape[0] == 2
    assert torch.equal(model.get_vertex[:, 0, 0], torch.tensor([9.0, 27.0]))
    assert model.contrib_max.tolist() == [10.0, 5.0]
