Source: https://github.com/t2-splat/T2Splat, commit b1b44d1 ("docs: format T2-Splat name with superscript").

Vendored: train.py (Stage I geometry), train_face_texture.py (Stage II residual
per-face Mip-textures), create_ply.py, arguments/, scene/, triangle_renderer/ and
the utils/ modules they import. The CUDA rasterizer
(submodules/diff-triangle-mesh-rasterization) is unmodified under
msbench/vendor/_csrc/t2splat_rasterization. Upstream's effrdel changes
(rdel.run face_selector/return_debug, backward compatible) are merged into
submodules/effrdel. Not vendored: render.py/metrics.py (MeshSuite renders and
evaluates), the TNT/DTU evaluators, prior-generation tools, GLB/video scripts and
tests.

MeshSuite modifications (training math unchanged):
- Imports are package-absolute (msbench.vendor.t2splat.*, _cmod.t2splat_rasterization,
  _cmod.simple_knn).
- DTU scenes with cameras.npz use the loader shared by every MeshSuite method
  (scene/dataset_readers.readDTUSceneInfo); ModelParams gains dtu_eval_mode and
  camera_utils ignores the RGBA alpha in DTU "full" mode, as for MeshSplatting.
- train.py: the __main__ block is exposed as build_parser()/run_training(argv);
  the VGG LPIPS net for in-training test reports is built lazily; PyTorch's
  CUDA cache is released around RDT so effrdel's Taichi BVH can allocate.
- train_face_texture.py: LPIPS is built lazily; alpha ground truth is composited
  with the render background as in Stage I (a no-op for RGB scenes).
See LICENSE.md (Apache-2.0 for the T²-Splat modifications) and LICENSE_GS.md
(Inria Gaussian Splatting license for the original code).
