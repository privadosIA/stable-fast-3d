# StableFast3D.Net

**Powered by Stability AI.** A C# port of Stable Fast 3D; see the [License](#license).

> Part of [mediaOrquestator](https://orquestator.xuxo.net/), where it makes the 3D models of the *models3d* category. **Status: the model is ported and the service makes models** (a picture in, a `.glb` out, identical in triangles to the Python container's on the reference pictures); the end-to-end measurements and the integration with the orchestrator are still open. The plan is in [`openspec/changes/port-stable-fast-3d`](openspec/changes/port-stable-fast-3d) and in [`docs/porting-plan.md`](docs/porting-plan.md).

C# port (.NET 8 + [TorchSharp](https://github.com/dotnet/TorchSharp)) of [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d), inference only: one picture in, one binary glTF (`.glb`) out. It replaces, as the product's engine, the Python container `docker/model3d` of mediaOrquestator (which stays as an optional fallback), the same way HeartMuLa, DiffRhythm2 and LAION-CLAP were migrated.

## What is in it

| part | where | what |
|---|---|---|
| Library | `src/StableFast3D.Net/` | the whole model: picture, cut-out, network, surface, UV unwrapping, texture bake and `.glb` (see below); `ModelGenerator` is the entry point |
| HTTP service | `src/StableFast3D.Service/` | `GET /health`, `GET /engines`, `POST /generate`, `GET /jobs/{id}`, `GET /jobs/{id}/model`, `DELETE /jobs/{id}`, Swagger UI at `/docs`; **the same interface as the Python container** (`docker/model3d/server.py` of mediaOrquestator), see [`docs/http-service.md`](docs/http-service.md) |
| Tests | `tests/StableFast3D.Net.Tests/` | xUnit, 187 tests: parity with the golden run stage by stage, the service in process and the real engine; those that need the weights (`SF3D_CKPT`), the fixtures or a card are skipped, saying why, when they are not there |
| Tools | `tools/` | the golden run (Python only inside the reference container, to make the fixtures once), the manifest of the fixtures and the check of the `.glb` with the orchestrator's own validator |

## What it ports

Every stage is checked against fixtures of the golden run (Stable Fast 3D at commit `ff21fc4`, fp32 on the CPU).

| stage | here | parity with the reference |
|---|---|---|
| Picture and cut-out | `Picture/`: decode (png, jpg, webp), `u2net.onnx` on ONNX Runtime, Pillow's resampling step by step | byte for byte |
| Camera | `Camera/`: the default camera and `LinearCameraEmbedder` | bit for bit (constants), 1e-4 (embedding) |
| Image tokens | `Vision/Dinov2.cs`: DINOv2-large with the per-layer modulation | tokens within 8e-5 |
| Triplane, backbone, scene codes | `Triplane/`, `Transformer/`: learned planes, two-stream interleave transformer, pixel-shuffle upsampling | planes and scene codes bit for bit, backbone within 1.5e-5 |
| Heads and material | `Field/` (decoder, triplane query), `Vision/Clip*.cs` (ViT-B/32 and its Beta heads for roughness and metallic) | query bit for bit, decoder within 1.5e-5, roughness and metallic within 4e-7 |
| Surface | `Mesh/`: marching tetrahedra on `sf3d/160_tets.npz` (read from its `.npz` with no numpy) | the reference mesh, index by index |
| UV and texture | `Texture/`: the unwrapper (a literal port of its C++, the rest with TorchSharp), the bake, the filling of borders, the dither | textures byte for byte at 512; see *Limits* |
| Output | `Mesh/GlbWriter.cs`: the container's file structure, two JPEG textures and the roughness and metallic factors | indices and UV identical; read by `blender.ReadModel` of mediaOrquestator |

Not ported: training, the global illumination estimator (off in the container), remeshing (the container uses `none`), the Metal baker.

`detail` is the size of the baked textures: `low` 512, `normal` 1024, `high` 2048 pixels (the mesh does not change: 6,000 to 30,000 triangles on the five reference pictures).

## Measurements

Five reference pictures, the container's `.glb` as the reference. Seconds are from the picture to the `.glb`, with the weights already loaded.

| | picture to model, `low` / `normal` / `high` | memory |
|---|---|---|
| RTX 5060 Ti (`cuda-linux` build) | 1.4-1.8 s / 1.6-1.8 s / 2.5-3.1 s (the Python container: 1.5 to 2 s) | **7.5 GiB of video memory** at most (what `nvidia-smi` shows for the process: the weights 3.7 GiB, the CUDA context and the allocator's cache; the container needs 6.4 to 7.3 GiB) and 4 GiB of RAM |
| CPU, 20 threads | 13-17 s at `low`, about 20 s at `high` | 8 GiB of RAM |

The first model after the service starts also loads the weights: 4 s on the card, 3 s on the CPU (with the files in the system cache). What the port makes, against the container's model of each picture: the **same number of triangles** (0.00 %), the bounding box and the Chamfer distance within 1e-5 of the size on the card and 1e-6 on the CPU (the limit is 1e-3), and the same roughness and metallic (to 6 decimals). The same picture and detail always give the same model, byte for byte.

## Limits

- **The unwrapping is not bit for bit on every picture.** Every stage is identical when it is given the reference's input, but PyTorch 2.10's CPU `atan2`, `sin` and `cos` (Sleef, 1 ulp) do not round as TorchSharp's libtorch does, and that 1e-7 is enough for the overlap tests, which compare against thresholds of that size, to place a handful of triangles in another slot of the atlas (none on three pictures, 0.3 % and 0.6 % on the other two). The textures are byte for byte the reference's when the layout is.
- **The dither of the textures is seeded.** The container does not seed it, so its textures differ in their last bit from run to run; the port always uses the same seed.
- **Video memory is tight on an 8 GB card** (7.5 GiB at most). TorchSharp has no `empty_cache`, so the allocator keeps what it took; the model frees what it can as it goes (`Frugal`).
- **`resolution` is the surface's** (160 points a side) and not the container's 128, 192 or 256, which are TripoSR's grids.

## Build and run

```bash
dotnet build StableFast3D.Net.sln -p:TorchBackend=cpu            # cpu | cuda-linux | cuda-windows
dotnet test  StableFast3D.Net.sln -p:TorchBackend=cpu
scripts/run-service.sh --model ~/Proyectos/mediaOrquestator/models/model3d --port 8791
```

The weights are the ones of the container (`models/model3d` of mediaOrquestator: `sf3d/config.yaml, sf3d/model.safetensors, sf3d/dinov2-large-config.json, sf3d/160_tets.npz, rembg/u2net.onnx`), never inside the repository. The shared model-agnostic code is in the sibling repository [`musicgen-core`](../musicgen-core) (`MusicGenCoreDir`).

## How it fits mediaOrquestator

The orchestrator makes a model with one HTTP call per picture; what it needs from a service is the contract in [`docs/http-service.md`](docs/http-service.md) (`engine_version` of a model is part of what decides it: this port says `sf3d-net@<version of the port>`). The server starts and stops the service like the other .NET ones (published to `bin/services/`), or the container, as the person chooses.

## Related repositories

Depends on `musicgen-core` (shared code) and follows the original Stable Fast 3D. Used by `veritmo`, which makes the 3D models of the *models3d* category with it.

| Repository | What it is |
|---|---|
| [`veritmo`](https://github.com/privadosIA/veritmo) | The server: the HTTP API and its contract, the pipelines (video, images, music), the run history and the music services it starts. |
| [`dashboards-labs`](https://github.com/privadosIA/dashboards-labs) | The dashboards (Angular) that talk to the server through its OpenAPI contract. |
| [`heartmula`](https://github.com/privadosIA/heartmula) | C# (.NET 8 + TorchSharp) port of [heartlib](https://github.com/HeartMuLa/heartlib): the HeartMuLa song generator, run by the server as a service. |
| [`DiffRhythm2`](https://github.com/privadosIA/DiffRhythm2) | C# port of [DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2): a diffusion model that writes a whole song. |
| [`lion-clap`](https://github.com/privadosIA/lion-clap) | C# port of [LAION-CLAP](https://github.com/LAION-AI/CLAP): audio and text embeddings, used to recognise songs. |
| [`musicgen-core`](https://github.com/privadosIA/musicgen-core) | The TorchSharp building blocks shared by the five ports. |
| [`triposr`](https://github.com/privadosIA/triposr) | C# port of [TripoSR](https://github.com/VAST-AI-Research/TripoSR): one picture in, one 3D model (`.glb`) out; makes the models of the *models3d* category (MIT original). |
| [`stable-fast-3d`](https://github.com/privadosIA/stable-fast-3d) (this repository) | C# port of [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d): one picture in, one textured 3D model (`.glb`) out. **Under the Stability AI Community License, not Apache-2.0** (free for research and non-commercial use, and for commercial use under US$1M of yearly revenue). |
| [`forms-lib`](https://github.com/privadosIA/forms-lib) | `@core/dynamic-forms`, the Angular library that draws forms and tables from JSON. |

The links are to the public copies of these repositories. The third-party tools, models and libraries they build on, with their licenses, are listed in [THIRD_PARTY.md](https://github.com/privadosIA/veritmo/blob/main/THIRD_PARTY.md).


## License

Apache 2.0 for what is original here (see [LICENSE](LICENSE)), as the other ports of this family. **This is a port of work by other people** and a Derivative Work of Stable Fast 3D, which is under the [Stability AI Community License](STABILITY-AI-COMMUNITY-LICENSE.md): that agreement applies to the port and to the weights as well (free for research and non-commercial use, and for commercial use by people and organisations under US$1M of yearly revenue who register with Stability AI; above that, a licence from Stability AI). See [NOTICE](NOTICE) for the details, what was changed and the third-party packages and, for the whole set, the mediaOrquestator `dotnet/THIRD_PARTY.md`.

This licence covers the code of this repository. The model weights (`stabilityai/stable-fast-3d`) are not in it: they are gated on Hugging Face and are under the Stability AI Community License.

## Maintenance

Maintained by one person in their free time: issues and pull requests are welcome, with no promised response time. See [CONTRIBUTING.md](CONTRIBUTING.md) and, for vulnerabilities, [SECURITY.md](SECURITY.md).
