# Tasks

## 1. Reference and weights

- [x] 1.1 `tools/dump_reference.py` and `tools/golden_all.sh`: run the Python reference inside the image built with `SF3D=1` over five pictures and dump the camera embedding, the DINOv2 tokens, the triplane tokens, the backbone output, the scene codes, the heads on a fixed set of points, the tetrahedra mesh, the UVs, the baked textures and the final `.glb`; write `tests/fixtures/MANIFEST.md`.
- [x] 1.2 Read `sf3d/model.safetensors` (the checkpoint already holds the DINOv2 and the CLIP towers; strict loader test both ways) and the DINOv2 config; add the tetrahedra file `160_tets.npz` of the upstream repository to the weights folder (as `sf3d/160_tets.npz`) and to `EngineInfo.RequiredFiles`; the CLIP text tower and the global estimator are the only keys allowed to be left over.

## 2. The picture and the camera

- [x] 2.1 Decoder, cut-out with ONNX Runtime and `u2net.onnx`, resize to 512 and the composite over the background colour with the mask; the default camera and the intrinsics; `LinearCameraEmbedder`; parity.

## 3. Image tokens

- [x] 3.1 DINOv2-large (patch embedding, 24 blocks, layer scale, interpolated position encoding) with the per-layer modulation from the camera embedding; parity of the tokens.

## 4. Triplane, backbone and scene codes

- [x] 4.1 Learnable triplane embedding (3 planes of 96x96 x 1024); `TwoStreamInterleaveTransformer` (4 blocks of 3 basic blocks, 1792 latents); `PixelShuffleUpsampleNetwork` (x4, 40 channels); parity of the scene codes.

## 5. Heads and estimators

- [x] 5.1 `MaterialMLP` (density, features, perturbed normal, vertex offset) and the triplane query; the CLIP ViT-B/32 vision tower and its heads (roughness, metallic: two values per picture, the mode of a Beta distribution); parity on the fixed points. The global estimator (24 spherical-Gaussian amplitudes) is not ported.

## 6. The mesh

- [x] 6.1 Marching tetrahedra on the `160_tets.npz` grid with the vertex offsets; the triangles count and the bounding box against the reference; the finish of the container (the origin at the middle of the base, as the container only grounds the model: the turn to Y up is part of `run_image`; at most 500,000 triangles, or the job fails with a sentence).

## 7. UV and texture

- [x] 7.1 Spike (measured on the five objects, written in `design.md`): reimplement `uv_unwrapper` and `texture_baker` in C# or build their C++ as a native library; decide and record it.
- [ ] 7.2 UV unwrapping and the baking of the albedo and the tangent-space normals at 512, 1024 and 2048 (roughness and metallic are not baked: they are the two values of the CLIP heads); parity of the two textures on the fixtures (mean absolute error per channel) and visual comparison.
- [ ] 7.3 `GlbWriter` with a base colour texture, a normal map and the roughness and metallic factors, readable by `blender.ParseModel` of mediaOrquestator.

## 8. The service engine

- [ ] 8.1 `StableFast3DEngine : IModelEngine` replaces `NotPortedEngine` in `EngineFactory`; the model loads on the first job; `PortVersion` is set; `detail` is the bake size and the triangles do not change with it.
- [ ] 8.2 End to end: the five pictures through the service; triangles, bounding box, textures and times against the Python one; the memory on the 5060 Ti (8 GB of VRAM in the container) written in the README.
- [ ] 8.3 `bin/services/stable-fast-3d` publish script and the `ServiceSpec` in mediaOrquestator (task of that repository).
