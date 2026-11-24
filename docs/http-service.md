# The HTTP service

`stablefast3d-service` (project `src/StableFast3D.Service`) makes a 3D model from one picture with Stable Fast 3D. It is the same interface as the Python service `docker/model3d/server.py` of mediaOrquestator, so the orchestrator uses it with no change in how it asks, and any other program can use it with `curl`. Open `http://HOST:PORT/docs` for the interactive page (Swagger UI, bundled, works offline); the machine-readable description is `http://HOST:PORT/openapi.yaml`.

It speaks plain HTTP and is meant for localhost or a trusted network. Put a reverse proxy in front of it if it must be reached from elsewhere, and set a token.

> **The engine** loads the weights when the first model is asked for (about 10 seconds on the CPU) and keeps them; `/health` and `/engines` answer at once and never load them. A model takes about 13 to 17 seconds on the CPU (the textures at `low`; more at `high`) and about two seconds on a card.

## Start it

```bash
scripts/run-service.sh --model ~/Proyectos/mediaOrquestator/models/model3d --port 8791
```

| variable | default | meaning |
|---|---|---|
| `SF3D_CKPT` | `/ckpt` | where the weights are: the layout of the container's `/ckpt` (`sf3d/config.yaml, sf3d/model.safetensors, sf3d/dinov2-large-config.json, sf3d/160_tets.npz, rembg/u2net.onnx`) |
| `SF3D_TOKEN` | empty | bearer token; empty leaves the API open |
| `SF3D_PORT` | `8791` | port |
| `SF3D_HOST` | `0.0.0.0` | the address to listen on: `0.0.0.0` (all interfaces, as the Python container does) or `127.0.0.1` / `localhost` for this machine only. With no token, listen on loopback |
| `SF3D_VERSION` | `1` | the version string `/health` reports (the version of a *model* is `sf3d-net@<EngineInfo.PortVersion>`) |
| `SF3D_DEVICE` | `cpu` | `cpu`, `cuda` or `cuda:N` (a CUDA device needs a CUDA build) |

## The calls

| call | answer |
|---|---|
| `GET /health` | `{ready, version, busy, queued, gpu, missing, engines}`: `ready` is false while a file of the weights is missing (named in `missing`) or the device does not work |
| `GET /engines` | `{engines: {sf3d: {built, missing, ready, version}}}` |
| `POST /generate?engine=sf3d&detail=low\|normal\|high` | the picture (png, jpg or webp, up to 20 MB) as the body -> `202 {id}`; `400` for another engine, a detail that is none or a body that is not a picture; `413` over 20 MB |
| `GET /jobs/{id}` | `{id, state (queued, running, done, failed), progress, error, seconds, engine, engine_version, detail, detail_used, resolution, triangles}` |
| `GET /jobs/{id}/model` | the `.glb` (`model/gltf-binary`); `409` until the job is done |
| `DELETE /jobs/{id}` | forgets the job and its model |

`detail` is the size in pixels of the baked textures: `low` 512, `normal` 1024, `high` 2048 (the mesh is the same at the three, 6,000 to 30,000 triangles for the reference pictures), so `detail_used` is always the detail asked for and `resolution` is the resolution of the surface, 160 points a side, whatever the detail. (The Python container reports 128, 192 or 256 in `resolution` for this engine, the grids of TripoSR; nothing reads that field.) A model over 500,000 triangles, or none at all, fails with a sentence in `error`: the mesh does not depend on the detail, so there is nothing to make again. Every model leaves upright (Y up) with its origin at the middle of its base, its colours in two JPEG textures (base colour and normal map) and the roughness and metallic of the picture as material factors. The textures' dither is seeded: the same picture and detail give the same model, byte for byte.

## How mediaOrquestator uses it

The server runs it as a service of `bin/services/` (published with `dotnet publish`, like HeartMuLa and DiffRhythm2) or, with the Python container, the other way; both answer this contract. The `engine_version` of a model is part of what the orchestrator caches by: it never reuses a model made by another version of the port.
