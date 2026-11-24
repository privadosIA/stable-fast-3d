# Design

## Context

What exists: the Python container (`docker/model3d/server.py`, `fetch.py`, `triposr-local.patch` and `sf3d-local.patch`) and its measured behaviour (see the README of mediaOrquestator's `docs/MODEL_GENERATION.md`): one model in 1 to 2 s with the engine loaded (baked texture 512, 1024 and 2048 pixels (the mesh does not change: 13,000 to 26,000 triangles on five objects)), 8 GB of VRAM. What exists here: the library with `EngineInfo`, the service with the full contract and `NotPortedEngine`, and 11 tests.

## Decisions

**1. The contract is frozen at the container's.** The service speaks exactly the calls, bodies, status codes and field names of `server.py`; `openapi.yaml` and a test that compares it with the route table keep it honest. The orchestrator does not change how it asks; it only learns where the service is.

**2. The port version decides the cache.** The orchestrator keys a model by its `engine_version`; this port says `sf3d-net@<PortVersion>`, so a model of the container and a model of the port are never taken for each other, and `PortVersion` changes whenever a model would.

**3. Parity first, speed later.** Each stage is ported in fp32 on the CPU first and checked against fixtures (`max_abs <= 1e-4`, `cos_sim >= 0.9999`); the CUDA path comes with an opt-in test, as in `lion-clap`. The mesh is checked by triangles count, bounding box and a Chamfer distance (marching cubes is exact on the same density; it is the density that carries the tolerance).

**4. Weights are the container's.** The files and their names are those of `models/model3d`; the C# modules replicate the keys of the checkpoint so the strict loader needs no mapping. The checkpoint is already safetensors; the tetrahedra grid is the one extra file and goes in the weights folder.

**5. Python only in `tools/`.** The golden run needs the reference; nothing else does. Fixtures are in the repository (small) or regenerated (big), checked against a sha256 table; the weights never are.

**6. The texture ops are the hard part; they are ported to C# (spike 7.1, measured).** `uv_unwrapper` and `texture_baker` are about 2,400 lines of C++/CUDA (plus 670 of Python for the unwrapper) with no managed equivalent. Two ways were open: reimplement them in C#, or build their C++ as a native library and call it through P/Invoke. **Decision: C#.** The native route is ruled out by the rules of the project, not by speed: it needs a C++ toolchain and OpenMP in the build and a binary per platform, while the build is C# and shell only; the sources also use `torch/extension.h` (so they would need libtorch's headers or their glue rewritten) and x86 SSE intrinsics. What the spike measured on the five reference objects (CPU, fp32, 13,000 to 30,000 triangles; reference timings are the extension's own with OpenMP on 20 threads):

| piece | result in C# | time (C# vs reference) |
|---|---|---|
| `texture_baker.rasterize` and `interpolate` (`Texture/UvBvh.cs`, `TextureBaker.cs`) | **bit for bit**: the sha256 of the rasterized map, the interpolated positions and the interpolated normals equal the reference's at 512, 1024 and 2048, on all five objects (tests `TextureBakerTests`) | 512: 17-69 ms vs 11-22; 1024: 28-79 vs 33-53; 2048: 60-120 vs 121-166 ms |
| `torch.pca_lowrank` with `manual_seed(0)` (`UvUnwrapper.AlignMeshWithMainAxis`), the one step that depends on a random generator | **bit for bit** (positions and normals after the alignment, `max_abs` 0): libtorch's generator, QR and SVD give the same numbers through TorchSharp | negligible (3 x 2 matrices) |
| the rest of `unwrap.py` | vectorised tensor operations, to be ported with TorchSharp in 7.2 and checked step by step against `tools/dump_unwrap.py` (rebuilt from the golden run: every step, and the final UVs, equal the recorded ones) | reference unwrapper: 45-225 ms per object |
| `assign_faces_uv_to_atlas_index` (C++: per cube face a BVH, 2D triangle-triangle overlap tests, the occluded triangle moves to the next atlas slot) | not ported yet (7.2): about 1,100 lines of scalar float32 geometry with no torch in them (4 x 4 determinants against 1e-7, polygon clipping, area threshold 1e-10). Its result is a plain integer array, checked exactly against `atlas_index` of the fixtures | part of the unwrapper's time above |

Two quirks of upstream that the port keeps because the pixel that decides them is on an edge shared by two triangles, where the traversal order of the BVH chooses the triangle: the triangle centroid uses `0.3333` and not 1/3, and the SAH of the baker runs, on x86, the SSE variant whose right-hand areas take the box of the neighbouring bin (the node bounds are then recomputed by the plain loop). With them the map is identical; the rest of the pipeline's texture work (task 7.2) is the albedo and normal bake and the dilation. A whole bake is a few hundred milliseconds on the CPU against 1.5 s for the model on the card.

**7. The picture code is Pillow's, step by step.** Parity at the first stage means the same pixels, not similar ones: the resize of the picture to 512 (bicubic through premultiplied alpha) and the cut-out's resizes (Lanczos) are ported from Pillow's fixed-point resampler and checked against Pillow's own output (`tools/dump_pil.py`) and against the five pictures. Decoding is SkiaSharp (MIT; libpng, libjpeg-turbo and libwebp, as Pillow uses), the cut-out is `u2net.onnx` on ONNX Runtime (`Microsoft.ML.OnnxRuntime`, MIT), as the sibling `TripoSR.Net` does; a jpeg is turned as its EXIF says in the cut-out path only, as `rembg.remove` does (a picture with its own alpha is used as it is).

## Risks / Trade-offs

- **Parity of a transformer this size in fp32** can drift by layers; the stages are checked one by one so a drift is found where it starts.
- **Memory and time on the card.** The reference needs 8 GB of VRAM; the port must not need more, and its first load time is measured, not assumed.
- **Licence.** The code and the weights of Stable Fast 3D are under the Stability AI Community License; a port derived from the code is a Derivative Work. Settle it (see `NOTICE`) before any publication.

## Migration Plan

Nothing changes in mediaOrquestator until the port passes its end-to-end check; then its own change adds the service next to the container, and the person chooses (the container stays as the optional path). Rollback is not using the service.

## Open Questions

- One service per engine (this repository's port 8791) while the container has both engines behind one port: the orchestrator's `ServiceSpec` per engine is a change of mediaOrquestator.
