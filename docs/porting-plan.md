# Porting plan

The model, stage by stage, with what each stage is in the reference and where it goes here. Every stage ends in a parity test against fixtures of the golden run (see `openspec/config.yaml` for the thresholds). The tasks, in order, are in `openspec/changes/port-stable-fast-3d/tasks.md`.

| stage | in the reference (`sf3d/`) | here (planned) |
|---|---|---|
| Picture | RGBA 512x512 (the object cut out by `rembg`), composited over the background colour with its mask | `Picture/`: decode, cut-out with ONNX Runtime (`u2net.onnx`), the same resize and composite |
| Camera | `LinearCameraEmbedder` (c2w and normalised intrinsics, 25 -> 768) | `Camera/` |
| Image tokens | `DINOV2SingleImageTokenizer`: DINOv2-large (1024 channels) with per-layer modulation from the camera embedding | `Vision/Dinov2.cs` and `Vision/Modulation.cs` |
| Triplane tokens | `TriplaneLearnablePositionalEmbedding`: 3 planes of 96x96, 1024 channels | `Triplane/Tokenizer.cs` |
| Backbone | `TwoStreamInterleaveTransformer`: 4 blocks of 3 basic blocks, 16 heads of 64, 1792 latents, image stream and triplane stream interleaved | `Transformer/TwoStream.cs` |
| Scene codes | `PixelShuffleUpsampleNetwork`: 1024 -> 40 channels, x4 | `Triplane/Upsample.cs` |
| Heads | `MaterialMLP`: density (trunc_exp), features, perturbed normal, vertex offset | `Field/MaterialMlp.cs` |
| Material | `ClipBasedHeadEstimator` (OpenCLIP ViT-B/32: roughness, metallic) and `MultiHeadEstimator` (global: 24 spherical-Gaussian amplitudes) | `Estimators/` (CLIP vision tower) |
| Surface | marching **tetrahedra** on the `160_tets.npz` grid of the repository, with the vertex offsets | `Mesh/MarchingTetrahedra.cs` (and the tetrahedra file in the weights folder) |
| UV and texture | `uv_unwrapper` (C++: BVH, intersections, unwrap) and `texture_baker` (C++/CUDA/Metal rasteriser) bake albedo, roughness, metallic, normals | `Texture/`: the two ops reimplemented in C# (CPU first) or the C++ built as a native library: to be decided in the spike |
| Remesh | optional (`none` is what the container uses); `pynanoinstantmeshes` for triangle or quad | not ported (the container uses `none`) |
| Output | `trimesh` PBR `.glb` (base colour texture, metallic-roughness texture, normal map) | `Mesh/GlbWriter.cs` (with textures) |

## Checking it

1. **Golden run** (`tools/`, needs Docker and the weights, offline): the Python reference runs inside `mediaorquestator/model3d:1` over a handful of pictures (the five objects of the tests of mediaOrquestator: skate, chest, lamp, trophy, cat) and dumps, stage by stage, the tensors that the port has to match: the cut-out picture, the image tokens, the triplane tokens, the backbone output, the scene codes, the field on a fixed set of points, the density grid and the mesh. It also converts the weights to safetensors once, with the names of the checkpoint.
2. **Parity tests** in `tests/`: each stage loads its fixtures through `FixtureDirectory` (checked against `tests/fixtures/MANIFEST.md`), runs the C# stage and compares (fp32, max_abs <= 1e-4, cos_sim >= 0.9999).
3. **End to end**: the same five pictures through the service; the triangles, the bounding box and a Chamfer distance against the reference meshes.

## What stays Python

Only `tools/dump_reference.py`, inside the reference container, to make the fixtures. Nothing the product, its tests or its build need.
