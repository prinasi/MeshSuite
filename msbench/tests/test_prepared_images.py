"""Prepared COLMAP images must be shared by training and evaluation."""

from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from msbench.core.config import apply_overrides, finalize_config, resolve_dataset_config
from msbench.core.datasets import ColmapDataset
from msbench.core.image_sources import resolve_colmap_image_source


@pytest.fixture
def scene(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 11 7 8 9 5.5 3.5\n")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.png\n\n")
    (tmp_path / "images").mkdir()
    Image.new("RGB", (11, 7), (255, 0, 0)).save(tmp_path / "images" / "frame.png")
    (tmp_path / "images_4").mkdir()
    Image.new("RGB", (3, 2), (0, 255, 0)).save(tmp_path / "images_4" / "frame.png")
    return tmp_path


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_config_records_effective_prepared_source(factor):
    cfg = resolve_dataset_config({"type": "colmap", "root": "unmounted", "resolution": factor})
    assert cfg["image_dir"] == f"images_{factor}"
    assert cfg["resolution"] == 1
    assert resolve_dataset_config(cfg) == cfg


def test_resolution_override_is_applied_before_source_selection():
    original = finalize_config({"dataset": {"type": "colmap", "image_dir": "images", "resolution": 4}})
    overridden = apply_overrides(original, ["dataset.resolution=2"])
    assert resolve_dataset_config(overridden["dataset"])["image_dir"] == "images_2"
    assert original["dataset"]["resolution"] == 4


@pytest.mark.parametrize("dtype", ["dtu", "synthetic"])
def test_other_dataset_protocols_keep_resolution(dtype):
    cfg = finalize_config({"dataset": {"type": dtype, "resolution": 2}})
    assert cfg["dataset"]["resolution"] == 2


def test_explicit_custom_folder_and_width_controls_are_preserved():
    assert resolve_colmap_image_source("custom", 2) == ("custom", 2)
    assert resolve_colmap_image_source("images", -1) == ("images", -1)
    assert resolve_colmap_image_source("images", 1600) == ("images", 1600)


def test_eval_uses_stored_pixels_dimensions_and_calibration(scene):
    with patch.object(Image.Image, "resize", side_effect=AssertionError("unexpected resize")):
        sample = ColmapDataset(scene, split="all", resolution=4).sample(0)
    assert sample.image_path == scene / "images_4" / "frame.png"
    assert (sample.camera.width, sample.camera.height) == (3, 2)
    assert torch.equal(sample.image[0, 0], torch.tensor([0.0, 1.0, 0.0]))
    expected = torch.tensor([[8 * 3 / 11, 0, 1.5], [0, 9 * 2 / 7, 1], [0, 0, 1]])
    assert torch.allclose(sample.camera.Ks[0], expected)


def test_missing_prepared_directory_fails_without_original_fallback(scene):
    with pytest.raises(FileNotFoundError, match="Generate images_2"):
        ColmapDataset(scene, split="all", resolution=2)


def test_incomplete_prepared_directory_fails_without_original_fallback(scene):
    (scene / "images_4" / "frame.png").unlink()
    with pytest.raises(FileNotFoundError, match="Prepared COLMAP image missing"):
        ColmapDataset(scene, split="all", resolution=4)


def test_2dts_native_config_and_pixels_use_prepared_source(scene, tmp_path):
    from msbench.trainers.d2ts_native import _build_d2ts_native_config
    from msbench.vendor.d2ts.diff_recon.datasets.Colmap_dataset import ColmapDataset as NativeDataset
    from msbench.vendor.d2ts.diff_recon.datasets.colmap_loader import readColmapCameras
    from msbench.vendor.d2ts.diff_recon.utils.file_handler import LocalHandler

    native = _build_d2ts_native_config(
        cfg={"dataset": {"type": "colmap", "resolution": 4},
             "d2ts": {"native_config": "configs/2dts/native/mipnerf360.yaml"}},
        dataset_root=str(scene), output_dir=tmp_path / "out", max_steps=100,
    )
    assert native.dataset.image_dir == "images_4"
    assert native.dataset.train_target_res == native.dataset.test_target_res == 1
    cameras = readColmapCameras(str(scene / "sparse/0/images.txt"), str(scene / "sparse/0/cameras.txt"), "images_4")
    dataset = NativeDataset(LocalHandler(str(scene)), cameras, target_res=1, use_alpha=False)
    with patch.object(Image.Image, "resize", side_effect=AssertionError("unexpected resize")):
        image = dataset._get_image(cameras[0].image_path)
    assert image.shape == (3, 2, 3)
    assert np.array_equal(image[:, 0, 0], [0, 1, 0])


@pytest.mark.parametrize("method", ["triangle-splatting", "mesh-splatting"])
def test_native_and_generic_method_paths_agree(scene, tmp_path, method):
    from msbench.cli.train import _mesh_splatting_native_argv, _triangle_splatting_native_argv
    from msbench.trainers.mesh_splatting_method import MeshSplattingTrainingMethod
    from msbench.trainers.triangle_splatting_method import TriangleSplattingTrainingMethod

    cls, argv_fn = (
        (TriangleSplattingTrainingMethod, _triangle_splatting_native_argv)
        if method == "triangle-splatting" else (MeshSplattingTrainingMethod, _mesh_splatting_native_argv)
    )
    args = cls(scene, tmp_path / "out", resolution=4)._build_dataset_args()
    assert (args.images, args.resolution) == ("images_4", 1)
    argv = argv_fn(trainer_cfg={"resolution": 4}, dataset_root=str(scene),
                   output_dir=tmp_path / "out", max_steps=100, quiet=True)
    assert argv[argv.index("-i") + 1] == "images_4"
    assert argv[argv.index("-r") + 1] == "1"


def test_diffsoup_uses_prepared_pixels_and_rejects_missing_folder(scene):
    from msbench.vendor.diffsoup.examples.utils import load_mipnerf360_scene

    with patch("torchvision.transforms.functional.resize", side_effect=AssertionError("unexpected resize")):
        data = load_mipnerf360_scene(str(scene), split="test", downscale=4)
    assert (data["W"], data["H"]) == (3, 2)
    assert torch.equal(data["frames"][0]["image"][0, 0], torch.tensor([0.0, 1.0, 0.0]))
    with pytest.raises(FileNotFoundError, match="Generate images_2"):
        load_mipnerf360_scene(str(scene), split="test", downscale=2)
