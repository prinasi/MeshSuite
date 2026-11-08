Source: ../ExMesh, commit 0bf121895ebd74642c1bab824a992c48f95296e8.

Core topology, UV baking, training losses and renderer are retained. Imports are package-relative; unused renderer depth-normal helper removed. The renderer can reuse a CUDA context without changing its math. Broken Gaussian-only capture/restore fields have been replaced by actual mesh state; NumPy color export explicitly detaches tensors and PLY export disables autograd. See per-file notices and LICENSE.md. MeshSuite's full iteration/RNG/sampler checkpoint handling lives in its training method.
