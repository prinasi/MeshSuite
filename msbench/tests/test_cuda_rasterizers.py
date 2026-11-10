"""Small real CUDA forward/backward checks for independently installed methods."""

import importlib
import math

import pytest
import torch

from msbench.methods import CUDA_MODULES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _operator(method):
    # A method environment may intentionally install only its own operators.
    name = CUDA_MODULES[method]
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        pytest.skip(f"Operator is not installed: {name}")
    if spec is None:
        pytest.skip(f"Operator is not installed: {name}")
    # If a binary exists but cannot load (e.g. ABI mismatch), fail the test.
    importlib.import_module(name)
    package = name.rsplit(".", 1)[0]
    return importlib.import_module(package)


def _vertices():
    return torch.tensor([
        [-0.6, -0.5, 2.0], [0.6, -0.5, 2.0],
        [-0.6, 0.5, 2.0], [0.6, 0.5, 2.0],
    ], device="cuda", requires_grad=True)


def _faces():
    return torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32, device="cuda")


def _matrices():
    from msbench.vendor.triangle_splatting.utils.graphics_utils import getProjectionMatrix

    view = torch.eye(4, device="cuda")
    projection = getProjectionMatrix(0.01, 10.0, math.pi / 2, math.pi / 2)
    return view, projection.T.contiguous().cuda()


def _settings(module):
    view, projection = _matrices()
    return module.TriangleRasterizationSettings(
        image_height=32, image_width=32, tanfovx=1.0, tanfovy=1.0,
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=view, projmatrix=projection, sh_degree=0,
        campos=torch.zeros(3, device="cuda"), prefiltered=False, debug=False,
    )


def _assert_render_and_gradients(image, parameters):
    assert image.is_cuda and image.numel() == 32 * 32 * 3
    assert torch.isfinite(image).all()
    assert image.abs().max() > 0, "No triangle contributed to the rendered image"
    image.square().mean().backward()
    torch.cuda.synchronize()
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0


def test_triangle_splatting_cuda_forward_backward():
    module = _operator("triangle-splatting")
    vertices = _vertices()
    triangles = vertices[_faces().long()]
    colors = torch.full((2, 3), 0.7, device="cuda", requires_grad=True)
    opacity = torch.full((2, 1), 0.8, device="cuda", requires_grad=True)
    points_per_triangle = torch.full((2,), 3, dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    output = module.TriangleRasterizer(_settings(module))(
        triangles_points=triangles.flatten(), sigma=torch.ones(2, device="cuda"),
        num_points_per_triangle=points_per_triangle, cumsum_of_points_per_triangle=starts,
        number_of_points=2, opacities=opacity,
        means2D=torch.zeros(2, 3, device="cuda", requires_grad=True),
        scaling=torch.zeros(2, device="cuda"), density_factor=torch.zeros(2, device="cuda"),
        colors_precomp=colors,
    )
    _assert_render_and_gradients(output[0], [vertices, colors, opacity])


def test_mesh_splatting_cuda_forward_backward():
    module = _operator("mesh-splatting")
    vertices = _vertices()
    colors = torch.full((4, 3), 0.7, device="cuda", requires_grad=True)
    weights = torch.full((4,), 0.8, device="cuda", requires_grad=True)
    output = module.TriangleRasterizer(_settings(module))(
        vertices=vertices, triangles_indices=_faces(), vertex_weights=weights,
        sigma=1.0, scaling=torch.zeros(2, device="cuda"), colors_precomp=colors,
    )
    _assert_render_and_gradients(output[0], [vertices, colors, weights])


def test_2dts_cuda_forward_backward():
    module = _operator("2dts")
    vertices = _vertices()
    colors = torch.full((2, 3), 0.7, device="cuda", requires_grad=True)
    opacity = torch.full((2, 1), 0.8, device="cuda", requires_grad=True)
    view, projection = _matrices()
    settings = module.TriangleRasterizationSettings(
        image_width=32, image_height=32, tanfovx=1.0, tanfovy=1.0,
        viewmatrix=view, projmatrix=projection, campos=torch.zeros(3, device="cuda"),
        sh_degree=0, gamma=1.0, background_depth=10.0,
        background=torch.zeros(3, device="cuda"), back_culling=False,
        rich_info=True, sort_level=0, debug=False,
    )
    output = module.TriangleRasterizer(settings)(
        vertex=vertices[_faces().long()].contiguous(),
        grad_holder=torch.zeros(2, device="cuda", requires_grad=True),
        opacity=opacity, feature=colors,
    )
    _assert_render_and_gradients(output[0], [vertices, colors, opacity])


def test_diffsoup_cuda_forward_backward():
    _operator("diffsoup")
    import msbench.vendor.diffsoup as ds

    positions = torch.tensor([[
        [-0.6, -0.5, 0.5, 1.0], [0.6, -0.5, 0.5, 1.0],
        [-0.6, 0.5, 0.5, 1.0], [0.6, 0.5, 0.5, 1.0],
    ]], device="cuda", requires_grad=True)
    faces = _faces()
    features = torch.full((2, ds.feats_at_level(0), 3), 0.7,
                          device="cuda", requires_grad=True)
    alpha = torch.full((2, ds.feats_at_level(0), 1), 0.8,
                       device="cuda", requires_grad=True)
    raster = ds.rasterize_multires_triangle_alpha(
        (32, 32), positions, faces, level=0, alpha_src=alpha, stochastic=False,
    )
    colors = ds.multires_triangle_color(raster, level=0, feat=features)
    image = ds.edge_grad(colors, raster, positions, faces)
    auxiliary_loss = ds.opacity_aux_loss(
        colors, torch.zeros_like(colors), raster, positions, faces, 0, alpha,
    )
    assert image.is_cuda and torch.isfinite(image).all() and image.abs().max() > 0
    (image.square().mean() + auxiliary_loss).backward()
    torch.cuda.synchronize()
    for parameter in (positions, features, alpha):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0


def test_simple_knn_cuda_matches_three_neighbor_reference():
    module = _operator("simple-knn")
    points = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.],
                           [0., 0., 3.], [3., 2., 1.]], device="cuda")
    distances = torch.cdist(points, points).square()
    distances.fill_diagonal_(float("inf"))
    expected = distances.topk(3, largest=False).values.mean(dim=1)
    actual = module.distCUDA2(points)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
