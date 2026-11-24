# tools/

The golden run: the only place Python exists in this repository, and only inside the reference container (`mediaorquestator/model3d:sf3d`, the image built with `SF3D=1`, offline, with the weights of `~/Proyectos/mediaOrquestator/models/model3d` mounted read-only).

| file | what |
|---|---|
| `dump_reference.py` | runs the real reference pipeline (`SF3D.run_image`, as the container's service does) over five pictures and records, with hooks, the tensors the port has to match (`.safetensors`), the mesh, the UVs, the textures and the `.glb` of every bake size |
| `dump_pil.py` | Pillow's own output for the exact calls the reference makes (bicubic RGBA resize through premultiplied alpha, Lanczos on RGB and one channel, the composite of rembg, the EXIF turn of a jpeg) on tiny synthetic pictures: `reference/pil/`, committed, to check the port's resampling pixel by pixel |
| `dump_unwrap.py` | the steps of Stable Fast 3D's UV unwrapper on the five meshes (PCA alignment, cube-face assignment, atlas indices, placed UVs), rebuilt from the golden run and checked against its recorded UVs: `<picture>/unwrap.safetensors` (tiny tensors, committed) and `<picture>/big/unwrap.safetensors` |
| `bench_baker.py` | times upstream's `texture_baker` at 512, 1024 and 2048 and records the sha256 of what it makes (`reference/baker_bench.json`), so the port is checked bit for bit even where the tensors are too big to keep |
| `verify_glb_with_orchestrator.sh` | reads `.glb` files with mediaOrquestator's own validator (`blender.ReadModel`) through go's `-overlay`, without writing anything in that repository (the test is `orchestrator/zz_sf3d_parse_test.go.txt`) |
| `golden_all.sh` | runs the container twice (fp32 on the CPU: the numerical truth; and on the card: times, VRAM, the container's `.glb`), copies the pictures and `160_tets.npz` out of the image and writes the manifest |
| `write_manifest.sh` / `verify_manifest.sh` | write and check `tests/fixtures/MANIFEST.md` (the sha256 of every fixture) |

```bash
tools/golden_all.sh                              # about 10 minutes; needs docker and the image
SF3D_GPU='"device=1"' tools/golden_all.sh        # the card for the CUDA run (1 is the RTX 5060 Ti here)
SF3D_GOLDEN_DEVICES=cpu tools/golden_all.sh      # only the CPU run
tools/verify_manifest.sh                         # the fixtures against the table
```

What is in a fixture directory (`tests/fixtures/reference/<picture>/`):

- `small.safetensors` (committed): the camera, the picture the model sees, CLIP features and the roughness / metallic, the mesh, the UVs, the baked 8-bit textures at 512, the heads over 4096 fixed points, the global estimator.
- `big/stages.safetensors` (regenerated, 400 MB or so each): DINOv2 layers 0, 11 and 23, image tokens, triplane tokens, backbone output, direct and scene codes, the decoder over the tetrahedra grid, the baking buffers and the float textures.
- `big/glb/{512,1024,2048}.glb` (regenerated): the model as the container writes it.
- `stats.json` (committed): triangles, bounds, seconds and VRAM per bake size, and the sha256 of the picture and of every `.glb`.

`tests/fixtures/pictures/` holds the upstream demo pictures the run used (extracted from the image, not committed: their licence is that of the upstream repository) and `tests/fixtures/160_tets.npz` the tetrahedra grid of the upstream repository.

Nothing here is needed to build, test or run the product once the fixtures exist.
