"""Image caches reproduce historical native 2DTS inputs, not just sizes."""

import json
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from msbench.core.builder import build_dataset
from msbench.core.config import Config, resolve_dataset_config
from msbench.core.image_sources import LEGACY_PREPROCESSING, legacy_image_cache_dir, validate_legacy_image_cache
from msbench.core.preprocessing import prepare_legacy_image_cache
from msbench.trainers.d2ts_native import _build_d2ts_native_config
from msbench.vendor.d2ts.diff_recon.datasets.Colmap_dataset import ColmapDataset as NativeDataset, ColmapDatasetFactory
from msbench.vendor.d2ts.diff_recon.datasets.colmap_loader import readColmapCameras
from msbench.vendor.d2ts.diff_recon.utils.file_handler import LocalHandler
from msbench.vendor.d2ts.diff_recon.utils.logger import stdout_logger


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.setenv("MSBENCH_IMAGE_CACHE_DIR", str(tmp_path / "cache"))
    root = tmp_path / "scene"
    sparse = root / "sparse/0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 19 13 12 13 9.5 6.5\n")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 nested/frame.JPG\n\n")
    (root / "images/nested").mkdir(parents=True)
    rng = np.random.default_rng(53)
    Image.fromarray(rng.integers(0, 256, (13, 19, 3), dtype=np.uint8)).save(root / "images/nested/frame.JPG")
    return root


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_lossless_cache_is_bit_exact_to_native_dynamic_preprocessing(scene, factor):
    cams = readColmapCameras(str(scene / "sparse/0/images.txt"), str(scene / "sparse/0/cameras.txt"), "images")
    original = NativeDataset(LocalHandler(str(scene)), cams, target_res=factor, use_alpha=False)
    expected = original._get_image(cams[0].image_path)
    manifest = prepare_legacy_image_cache(scene, factor, workers=2)
    cache = legacy_image_cache_dir(scene, factor)
    actual = NativeDataset(LocalHandler(str(scene)), cams, target_res=1, use_alpha=False)
    with patch.object(Image.Image, "resize", side_effect=AssertionError("runtime resize")):
        result = actual._get_image(str(cache / "nested/frame.JPG.png"))
    assert np.array_equal(result, expected)
    assert manifest["images"][0]["size"] == [19 // factor, 13 // factor]
    assert len(manifest["images"][0]["source_sha256"]) == 64
    assert prepare_legacy_image_cache(scene, factor) == manifest


def test_native_and_eval_use_same_cache_without_resize(scene, tmp_path):
    prepare_legacy_image_cache(scene, 4)
    ds_cfg = {"type": "colmap", "root": str(scene), "resolution": 4,
              "image_preprocessing": LEGACY_PREPROCESSING, "split": "all"}
    resolved = resolve_dataset_config(ds_cfg)
    assert resolve_dataset_config(resolved) == resolved
    cache = legacy_image_cache_dir(scene, 4)
    assert resolved["image_dir"] == str(cache)
    assert resolved["resolution"] == 1
    native = _build_d2ts_native_config(
        cfg={"dataset": ds_cfg, "d2ts": {"native_config": "configs/2dts/native/mipnerf360.yaml"}},
        dataset_root=str(scene), output_dir=tmp_path / "out", max_steps=100,
    )
    assert native.dataset.image_dir == str(cache)
    assert native.dataset.train_target_res == native.dataset.test_target_res == 1
    factory = ColmapDatasetFactory.__new__(ColmapDatasetFactory)
    factory._config = native.dataset
    factory._file_handler = LocalHandler(str(scene))
    factory._logger = stdout_logger
    _, infos = factory.getCameraInfos()
    native_ds = NativeDataset(factory._file_handler, infos, target_res=1, use_alpha=False)
    with patch.object(Image.Image, "resize", side_effect=AssertionError("runtime resize")):
        sample = build_dataset(ds_cfg).sample(0)
        native_sample = native_ds[0]
    assert torch.equal(sample.image.permute(2, 0, 1), native_sample.gt_image)
    assert sample.image_path == cache / "nested/frame.JPG.png"
    assert (sample.camera.width, sample.camera.height) == (4, 3)
    expected = torch.tensor([[12 * 4 / 19, 0, 2], [0, 13 * 3 / 13, 1.5], [0, 0, 1]])
    assert torch.allclose(sample.camera.Ks[0], expected)


def test_missing_or_changed_cache_is_rejected(scene):
    cfg = {"type": "colmap", "root": str(scene), "resolution": 4,
           "image_preprocessing": LEGACY_PREPROCESSING, "split": "all"}
    with pytest.raises(FileNotFoundError, match="prepare_2dts_images"):
        build_dataset(cfg)
    prepare_legacy_image_cache(scene, 4)
    with (scene / "images/nested/frame.JPG").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="Source image changed"):
        build_dataset(cfg)


def test_incomplete_cache_is_rejected(scene):
    prepare_legacy_image_cache(scene, 4)
    cache = legacy_image_cache_dir(scene, 4)
    (cache / "nested/frame.JPG.png").unlink()
    with pytest.raises(FileNotFoundError, match="Cached preprocessing image missing"):
        validate_legacy_image_cache(scene, cache)


def test_manifest_protocol_is_validated(scene):
    prepare_legacy_image_cache(scene, 4)
    cache = legacy_image_cache_dir(scene, 4)
    path = cache / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["protocol"] = "other"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol/source mismatch"):
        validate_legacy_image_cache(scene, cache)


def test_only_2dts_selects_legacy_cache_in_existing_configs():
    cfg = Config.fromfile("configs/2dts/mipnerf360/bicycle.yaml")
    assert cfg.dataset.image_preprocessing == LEGACY_PREPROCESSING
    for method in ("triangle-splatting", "mesh-splatting", "diffsoup"):
        cfg = Config.fromfile(f"configs/{method}/mipnerf360/bicycle.yaml")
        assert cfg.dataset.get("image_preprocessing", "stored") == "stored"
        assert resolve_dataset_config(cfg.dataset)["image_dir"] == "images_4"


def test_unity_defaults_and_image_matching_use_same_protocol(scene):
    from msbench.core.image_sources import unity_reference_image_path
    from tools.evaluate_deployment_images import _image_key

    prepare_legacy_image_cache(scene, 4)
    cache = unity_reference_image_path(scene, "2dts", "images_4")
    assert cache == legacy_image_cache_dir(scene, 4)
    assert _image_key(cache / "nested/frame.JPG.png") == _image_key(scene / "images/nested/frame.JPG")
    assert _image_key(cache / "nested/frame.JPG.png") == "frame"
