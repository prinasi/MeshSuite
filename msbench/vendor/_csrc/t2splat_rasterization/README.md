# T²-Splat differentiable triangle-mesh rasterizer

Vendored from [t2-splat/T2Splat](https://github.com/t2-splat/T2Splat) at commit
`b1b44d1` (`submodules/diff-triangle-mesh-rasterization`), unmodified.

It is the MeshSplatting rasterizer extended with per-face residual textures and
their Mip pyramid (`face_textures`, `mip_textures`, `mip_offsets`). The original
code is © 2023 Inria (GRAPHDECO) under `LICENSE_GS.md` (non-commercial research
and evaluation use); the MeshSplatting modifications are © University of Liege,
KAUST and University of Oxford under `LICENSE.md`; the T²-Splat modifications are
by the T²-Splat authors (Apache-2.0 repository). Several upstream files dropped
the original copyright headers; the licenses above still apply to them.

MeshSuite builds it as `msbench.vendor._cmod.t2splat_rasterization._C`
(`MSBENCH_CUDA_BACKENDS=t2splat`). GLM headers are shared with the MeshSplatting
rasterizer (`../mesh_splatting_rasterization/third_party/glm`).
