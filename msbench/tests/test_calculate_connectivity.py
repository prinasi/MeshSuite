from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.calculate_connectivity import MeshData, calculate_connectivity_metrics


def _mesh(vertices: list[list[float]], faces: list[list[int]]) -> MeshData:
    return MeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        source_type="test",
        source_path=__file__,
        source_format="memory",
    )


def test_square_patch_has_boundary_but_manifold_vertices():
    mesh = _mesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
    )

    result = calculate_connectivity_metrics(mesh)

    assert result["counts"]["edges"] == 5
    assert result["counts"]["boundary_edges"] == 4
    assert result["counts"]["non_manifold_edges"] == 0
    assert result["counts"]["non_manifold_vertices"] == 0
    assert result["components"]["count"] == 1
    assert result["metrics"]["largest_component_ratio"] == 1.0
    assert result["quality_flags"]["edge_manifold"] is True
    assert result["quality_flags"]["closed_edge_manifold"] is False


def test_three_faces_on_one_edge_are_non_manifold():
    mesh = _mesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        faces=[[0, 1, 2], [1, 0, 3], [0, 1, 4]],
    )

    result = calculate_connectivity_metrics(mesh)

    assert result["counts"]["non_manifold_edges"] == 1
    assert result["counts"]["non_manifold_vertices"] == 2
    assert result["edges"]["max_face_incidence"] == 3
    assert result["quality_flags"]["edge_manifold"] is False
    assert result["quality_flags"]["vertex_manifold"] is False


def test_largest_component_ratio_reports_fragmentation():
    mesh = _mesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        faces=[[0, 1, 2], [3, 4, 5]],
    )

    result = calculate_connectivity_metrics(mesh)

    assert result["components"]["count"] == 2
    assert result["components"]["single_face_component_count"] == 2
    assert result["metrics"]["largest_component_ratio"] == 0.5


def _grid(n: int, offset: float = 0.0) -> tuple[list[list[float]], list[list[int]]]:
    vertices = [[offset + x, float(y), 0.0] for y in range(n + 1) for x in range(n + 1)]
    faces = []
    for y in range(n):
        for x in range(n):
            a, b = y * (n + 1) + x, y * (n + 1) + x + 1
            c, d = a + n + 1, b + n + 1
            faces += [[a, b, d], [a, d, c]]
    return vertices, faces


def test_components_follow_shared_edges_on_larger_meshes():
    # Two separate 6x6 grids (72 faces each); small meshes can hide face/edge index mix-ups.
    vertices, faces = _grid(6)
    other_vertices, other_faces = _grid(6, offset=10.0)
    shift = len(vertices)
    mesh = _mesh(vertices + other_vertices, faces + [[i + shift for i in face] for face in other_faces])

    result = calculate_connectivity_metrics(mesh)

    assert result["components"]["count"] == 2
    assert result["components"]["single_face_component_count"] == 0
    assert result["metrics"]["largest_component_ratio"] == 0.5


def test_single_connectivity_reads_exmesh_exported_mesh(tmp_path):
    import trimesh

    # A closed tetrahedron: no boundary edges, one component.
    out = tmp_path / "outputs/exmesh/dtu-fg/scan24"
    (out / "mesh").mkdir(parents=True)
    trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        process=False,
    ).export(out / "mesh/mesh.ply")
    config = tmp_path / "configs/exmesh/dtu/scan24.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "dataset: {name: dtu, scene: scan24}\n"
        "adapter: {type: exmesh, checkpoint: %s/ckpt/final.pt}\n"
        "mesh: {output: %s/mesh/mesh.ply}\n"
        "output: {dir: %s}\n" % (out, out, out)
    )

    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["bash", str(root / "single_connectivity.sh"), "exmesh", "dtu/scan24",
         "--config_root", str(tmp_path / "configs"), "--python", sys.executable,
         "--connectivity_script", str(root / "tools/calculate_connectivity.py")],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    result = json.loads((out / "connectivity_metrics.json").read_text())
    assert result["counts"]["boundary_edges"] == 0
    assert result["components"]["count"] == 1
