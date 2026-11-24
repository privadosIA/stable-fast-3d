# Porting plan

The model, stage by stage, with what each stage is in the reference and where it goes here. Every stage ends in a parity test against fixtures of the golden run (see `openspec/config.yaml` for the thresholds). The tasks, in order, are in `openspec/changes/port-stable-fast-3d/tasks.md`.

| stage | in the reference (`sf3d/`) | here |
|---|---|---|
| Picture | RGBA 512x512 (the object cut out by `rembg`), composited over the background colour with its mask | `Picture/`: decode, cut-out with ONNX Runtime (`u2net.onnx`), the same resize and composite |
| Camera | `LinearCameraEmbedder` (c2w and normalised intrinsics, 25 -> 768) | `Camera/` |
| Image tokens | `DINOV2SingleImageTokenizer`: DINOv2-large (1024 channels) with per-layer modulation from the camera embedding | `Vision/Dinov2.cs` and `Transformer/Modulation.cs` |
| Triplane tokens | `TriplaneLearnablePositionalEmbedding`: 3 planes of 96x96, 1024 channels | `Triplane/TriplaneTokenizer.cs` |
| Backbone | `TwoStreamInterleaveTransformer`: 4 blocks of 3 basic blocks, 16 heads of 64, 1792 latents, image stream and triplane stream interleaved | `Transformer/Backbone.cs` |
| Scene codes | `PixelShuffleUpsampleNetwork`: 1024 -> 40 channels, x4 | `Triplane/PixelShuffleUpsampleNetwork.cs` |
| Heads | `MaterialMLP`: density (trunc_exp), features, perturbed normal, vertex offset | `Field/MaterialMlp.cs` |
| Material | `ClipBasedHeadEstimator` (OpenCLIP ViT-B/32: roughness, metallic) and `MultiHeadEstimator` (global: 24 spherical-Gaussian amplitudes) | `Vision/ClipVision.cs` and `Vision/ClipHeadEstimator.cs` (the CLIP vision tower and its Beta heads; the global estimator is not ported: the container runs with `estimate_illumination` off) |
| Surface | marching **tetrahedra** on the `160_tets.npz` grid of the repository, with the vertex offsets | `Mesh/MarchingTetrahedra.cs` and `Mesh/NpzReader.cs` (the tetrahedra file is `sf3d/160_tets.npz` in the weights folder) |
| UV and texture | `uv_unwrapper` (C++: BVH, intersections, unwrap) and `texture_baker` (C++/CUDA/Metal rasteriser) bake albedo, roughness, metallic, normals | `Texture/`: the two ops reimplemented in C# (the spike measured it: the baker and the PCA bit for bit, the atlas op a literal port); only the albedo and the tangent-space normal are baked, roughness and metallic are two numbers of the picture |
| Remesh | optional (`none` is what the container uses); `pynanoinstantmeshes` for triangle or quad | not ported (the container uses `none`) |
| Output | `trimesh` PBR `.glb` (base colour texture, normal map, roughness and metallic factors) | `Mesh/ModelAssembly.cs` and `Mesh/GlbWriter.cs` (base colour texture, normal map and the roughness and metallic as material factors) |

## Checking it

1. **Golden run** (`tools/`, needs Docker and the weights, offline): the Python reference runs inside `mediaorquestator/model3d:1` over a handful of pictures (five of the demo pictures of the upstream repository: chair1, character1, axe, tree and stylized-rocks) and dumps, stage by stage, the tensors that the port has to match: the cut-out picture, the image tokens, the triplane tokens, the backbone output, the scene codes, the field on a fixed set of points, the density grid and the mesh. It also converts the weights to safetensors once, with the names of the checkpoint.
2. **Parity tests** in `tests/`: each stage loads its fixtures through `FixtureDirectory` (checked against `tests/fixtures/MANIFEST.md`), runs the C# stage and compares (fp32, max_abs <= 1e-4, cos_sim >= 0.9999).
3. **End to end**: the same five pictures through the service; the triangles, the bounding box and a Chamfer distance against the reference meshes.

## What stays Python

Only `tools/dump_reference.py`, inside the reference container, to make the fixtures. Nothing the product, its tests or its build need.
