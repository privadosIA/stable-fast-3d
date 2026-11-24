# StableFast3D.Net

> Part of [mediaOrquestator](https://orquestator.xuxo.net/), where it makes the 3D models of the *models3d* category. **Status: the model is ported and the service makes models** (a picture in, a `.glb` out, identical in triangles to the Python container's on the reference pictures); the end-to-end measurements and the integration with the orchestrator are still open. The plan is in [`openspec/changes/port-stable-fast-3d`](openspec/changes/port-stable-fast-3d) and in [`docs/porting-plan.md`](docs/porting-plan.md).

C# port (.NET 8 + [TorchSharp](https://github.com/dotnet/TorchSharp)) of [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d), inference only: one picture in, one binary glTF (`.glb`) out. It replaces, as the product's engine, the Python container `docker/model3d` of mediaOrquestator (which stays as an optional fallback), the same way HeartMuLa, DiffRhythm2 and LAION-CLAP were migrated.

## What is in it

| part | where | what |
|---|---|---|
| Library | `src/StableFast3D.Net/` | `EngineInfo` today (id, version, files of the weights, details); the model as it is ported (see the table below) |
| HTTP service | `src/StableFast3D.Service/` | `GET /health`, `GET /engines`, `POST /generate`, `GET /jobs/{id}`, `GET /jobs/{id}/model`, `DELETE /jobs/{id}`, Swagger UI at `/docs`; **the same interface as `docker/model3d/server.py`**, so mediaOrquestator uses it with no change in how it asks. Default port 8791 |
| Tests | `tests/StableFast3D.Net.Tests/` | xUnit; the service is tested in process (11 tests today) |
| Tools | `tools/` | the golden run (Python only inside the reference container, to make the fixtures once) |

## What it ports

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

`detail` is the size in pixels of the baked texture (the mesh is the same at the three): baked texture 512, 1024 and 2048 pixels (the mesh does not change: 13,000 to 26,000 triangles on five objects).

## Build and run

```bash
dotnet build StableFast3D.Net.sln -p:TorchBackend=cpu            # cpu | cuda-linux | cuda-windows
dotnet test  StableFast3D.Net.sln -p:TorchBackend=cpu
scripts/run-service.sh --model ~/Proyectos/mediaOrquestator/models/model3d --port 8791
```

The weights are the ones of the container (`models/model3d` of mediaOrquestator: `sf3d/config.yaml, sf3d/model.safetensors, sf3d/dinov2-large-config.json, sf3d/160_tets.npz, rembg/u2net.onnx`), never inside the repository. The shared model-agnostic code is in the sibling repository [`musicgen-core`](../musicgen-core) (`MusicGenCoreDir`).

## How it fits mediaOrquestator

The orchestrator makes a model with one HTTP call per picture; what it needs from a service is the contract in [`docs/http-service.md`](docs/http-service.md) (`engine_version` of a model is part of what decides it: this port says `sf3d-net@<version of the port>`). The server starts and stops the service like the other .NET ones (published to `bin/services/`), or the container, as the person chooses.

**Licence to settle before publishing:** the code and the weights of Stable Fast 3D are under the Stability AI Community License, and a port derived from that code is a Derivative Work (see `NOTICE`).
