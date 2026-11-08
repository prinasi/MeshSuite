"""Compare native ExMesh tensors/gradients to an explicitly supplied reference repo."""

from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference.resolve()))

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    reference = module("exmesh_reference_renderer", args.reference / "renderer/__init__.py")
    reference_losses = module("exmesh_reference_losses", args.reference / "utils/loss_utils.py")
    from msbench.vendor.exmesh import renderer, losses
    from msbench.vendor.exmesh.runtime import native_camera
    from msbench.core.cameras import CameraBatch

    torch.manual_seed(7)
    vertices = torch.tensor(
        [[-0.6, -0.4, 0], [0.6, -0.4, 0], [0.6, 0.4, 0], [-0.6, 0.4, 0]], device="cuda"
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda")
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], device="cuda")
    texture = torch.rand(3, 32, 32, device="cuda")
    view = torch.eye(4, device="cuda")[None]
    theta = 0.23
    view[0, :3, :3] = torch.tensor(
        [
            [torch.cos(torch.tensor(theta)), 0, torch.sin(torch.tensor(theta))],
            [0, 1, 0],
            [-torch.sin(torch.tensor(theta)), 0, torch.cos(torch.tensor(theta))],
        ],
        device="cuda",
    )
    view[0, :3, 3] = torch.tensor([0.12, -0.08, 2.0], device="cuda")
    k = torch.tensor([[[50.0, 0, 32], [0, 48.0, 24], [0, 0, 1]]], device="cuda")
    camera = native_camera(CameraBatch(view, torch.linalg.inv(view), k, 64, 48))

    def run(implementation, loss_module):
        v = vertices.clone().requires_grad_()
        t = texture.clone().requires_grad_()
        model = SimpleNamespace(
            get_vertices=v, get_faces=faces, get_uvs=uv, get_uv_indices=faces, get_texture=t
        )
        out = implementation.render(camera, model, None, torch.zeros(3, device="cuda"))
        target = torch.ones_like(out["render"]) * 0.3
        rgb = 0.8 * loss_module.l1_loss(out["render"], target) + 0.2 * (
            1 - loss_module.ssim(out["render"], target)
        )
        mask = (out["rend_alpha"].detach() > 0.5).float()
        sil = loss_module.silhouette_bce_loss(out["rend_alpha"], mask) * 0.005
        smooth = loss_module.laplacian_smooth_loss(v, faces)
        depth_target = torch.linspace(0.8, 2.4, 48 * 64, device="cuda").reshape(48, 64)
        depth, _ = loss_module.pearson_correlation_loss(
            out["rend_depth"][0], depth_target, mask[0].bool()
        )
        reference_vertices = vertices + torch.tensor([0.01, -0.02, 0.03], device="cuda")
        deviation = loss_module.double_vertex_deviation_loss(v, reference_vertices, faces)
        total = rgb + sil + smooth * 0.01 + depth * 0.1 + deviation
        total.backward()
        result = {
            **{k: v.detach() for k, v in out.items()},
            "loss": total.detach(),
            "image_loss": rgb.detach(),
            "silhouette_loss": sil.detach(),
            "depth_loss": depth.detach(),
            "smooth_loss": smooth.detach(),
            "deviation_loss": deviation.detach(),
            "vertices_grad": v.grad.clone(),
            "texture_grad": t.grad.clone(),
        }
        optimizer = torch.optim.Adam(
            [{"params": [v], "lr": 0.0005}, {"params": [t], "lr": 0.0025}], eps=1e-15
        )
        optimizer.step()
        result.update(vertices_after_step=v.detach().clone(), texture_after_step=t.detach().clone())
        return result

    a = run(reference, reference_losses)
    b = run(renderer, losses)
    delta = {k: float((a[k] - b[k]).abs().max()) for k in a}
    # Repeated reference run measures GPU reduction nondeterminism separately.
    repeat = run(reference, reference_losses)
    noise = {k: float((a[k] - repeat[k]).abs().max()) for k in a}
    passed = all(torch.allclose(a[k], b[k], atol=1e-6, rtol=1e-5) for k in a)
    report = dict(
        passed=passed,
        max_abs_error=delta,
        reference_repeat_error=noise,
        tolerances=dict(atol=1e-6, rtol=1e-5),
        reference=str(args.reference.resolve()),
        scope="fixed UV, nonidentity camera, forward/all five losses/gradients/one Adam step",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
