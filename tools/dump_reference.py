#!/usr/bin/env python3
"""The golden run of the port: runs Stable Fast 3D (the reference, commit ff21fc4) over some pictures and writes, stage by stage, what the C#
port has to match. It runs INSIDE the reference container (see golden_all.sh); nothing else in this repository needs Python.

It runs the real pipeline (`SF3D.run_image`, the same call the container's service makes) and records what goes through it with hooks, so
the fixtures cannot drift from what the reference does:

  camera_embeds        LinearCameraEmbedder output                          (B, 1, 768)
  rgb_cond / mask_cond the picture the model sees, before and after the image processor
  dino_layer_{0,11,23} DINOv2 encoder layers (where a drift would start)
  image_tokens         DINOv2 tokenizer output                              (B, 1, 1024, 1297)   (1 + 36x36 tokens)
  triplane_tokens      learnable triplane embedding                         (B, 1024, 3*96*96)
  backbone_out         TwoStreamInterleaveTransformer output
  direct_codes         detokenized triplane                                 (B, 3, 1024, 96, 96)
  scene_codes          PixelShuffleUpsampleNetwork output                   (B, 3, 40, 384, 384)
  clip_features, image_estimator_*   CLIP ViT-B/32 features, the Beta parameters and the decoder_roughness / decoder_metallic values
  grid_*               the decoder over the tetrahedra grid (density, vertex offset)
  mesh_*               the mesh of marching tetrahedra (before the UV unwrapping)
  uv_*                 after the unwrapping: positions, triangles, UVs, normals, tangents (the vertices are duplicated at the seams)
  bake_*               the rasteriser output, the mask and the interpolated position / normal / tangent
  tex_*                the textures before they are made 8-bit (albedo and tangent-space normal)
  final_*              what trimesh holds at the end: vertices, faces, UVs, the two 8-bit textures, the roughness and metallic factors
  heads_*              the decoder over a fixed set of points (cube + surface), and the global estimator over the direct codes

Everything is routed by size: what is over 1 MiB goes to <object>/big/ (regenerated, never committed), the rest to <object>/small.safetensors.
The .glb of every bake size (512, 1024, 2048) is written as the container writes it (grounded, then trimesh's export).

  python dump_reference.py --ckpt /ckpt --out /out --device cpu
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time

# The service never reaches the network: the same two switches as the container's server.py.
os.environ.setdefault("SF3D_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PICTURES = ["chair1", "character1", "axe", "tree", "stylized-rocks"]
BAKES = [512, 1024, 2048]
BIG = 1 << 20
EXAMPLES = os.environ.get("SF3D_DIR", "/opt/stable-fast-3d") + "/demo_files/examples"


class Recorder:
    """Collects tensors by name; `on` switches it, so the extra calls the script makes itself are not recorded as the pipeline's."""

    def __init__(self):
        self.on = False
        self.tensors = {}

    def put(self, name, value):
        import torch

        if not self.on or not isinstance(value, torch.Tensor):
            return
        t = value.detach().to("cpu")
        if t.dtype == torch.bool:
            t = t.to(torch.uint8)
        self.tensors[name] = t.contiguous().clone()

    def take(self):
        out, self.tensors = self.tensors, {}
        return out


def sha256_of(data):
    return hashlib.sha256(data).hexdigest()


def write_tensors(tensors, small_path, big_path):
    """Writes the tensors in two files by size. Returns the names that went to each."""
    from safetensors.torch import save_file

    small = {k: v for k, v in tensors.items() if v.numel() * v.element_size() <= BIG}
    big = {k: v for k, v in tensors.items() if k not in small}
    if small:
        save_file(small, small_path)
    if big:
        os.makedirs(os.path.dirname(big_path), exist_ok=True)
        save_file(big, big_path)
    return sorted(small), sorted(big)


def install_hooks(model, rec, state):
    import torch
    import sf3d.system as system
    from sf3d.models.mesh import Mesh
    from sf3d.models.utils import ImageProcessor
    from texture_baker import TextureBaker

    def out_hook(name):
        def hook(module, args, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            rec.put(name, out)

        return hook

    orig_prepare = model.prepare_image

    def prepare_image(image):
        mask, rgb = orig_prepare(image)
        rec.put("mask_cond_pre", mask)
        rec.put("rgb_cond_pre", rgb)
        return mask, rgb

    model.prepare_image = prepare_image

    # ImageProcessor is called with the rgb first and the mask second (system.SF3D.generate_mesh)
    orig_proc = ImageProcessor.__call__
    proc_calls = {"n": 0}

    def processed(self, image, size):
        r = orig_proc(self, image, size)
        rec.put("rgb_cond" if proc_calls["n"] % 2 == 0 else "mask_cond", r)
        proc_calls["n"] += 1
        return r

    ImageProcessor.__call__ = processed

    model.camera_embedder.register_forward_hook(out_hook("camera_embeds"))
    model.image_tokenizer.register_forward_hook(out_hook("image_tokens"))
    model.tokenizer.register_forward_hook(out_hook("triplane_tokens"))
    model.backbone.register_forward_hook(out_hook("backbone_out"))
    model.post_processor.register_forward_hook(out_hook("scene_codes"))
    layers = model.image_tokenizer.model.encoder.layer
    for i in (0, 11, len(layers) - 1):
        layers[i].register_forward_hook(out_hook(f"dino_layer_{i}"))
    model.image_estimator.model.visual.register_forward_hook(out_hook("clip_features"))

    def estimator_hook(module, args, out):
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                rec.put(f"image_estimator_{k}", v)
            elif hasattr(v, "concentration1"):  # the Beta distributions
                rec.put(f"image_estimator_{k}_alpha", v.concentration1)
                rec.put(f"image_estimator_{k}_beta", v.concentration0)

    model.image_estimator.register_forward_hook(estimator_hook)

    def detokenize_wrapper(orig):
        def wrapped(*a, **k):
            r = orig(*a, **k)
            rec.put("direct_codes", r)
            return r

        return wrapped

    model.tokenizer.detokenize = detokenize_wrapper(model.tokenizer.detokenize)

    # The decoder is called twice by the pipeline: over the tetrahedra grid, then over the baked points.
    def decoder_hook(module, args, out):
        if state.get("pause"):
            return
        n = state["decoder_calls"]
        state["decoder_calls"] += 1
        prefix = "grid" if n == 0 else "bake_decoded"
        for k, v in out.items():
            rec.put(f"{prefix}_{k}", v)

    model.decoder.register_forward_hook(decoder_hook)

    orig_t2m = model.triplane_to_meshes

    def triplane_to_meshes(triplanes):
        meshes = orig_t2m(triplanes)
        rec.put("mesh_v_pos", meshes[0].v_pos)
        rec.put("mesh_t_pos_idx", meshes[0].t_pos_idx)
        return meshes

    model.triplane_to_meshes = triplane_to_meshes

    orig_unwrap = Mesh.unwrap_uv

    def unwrap_uv(self, *a, **k):
        r = orig_unwrap(self, *a, **k)
        rec.put("uv_v_pos", self.v_pos)
        rec.put("uv_t_pos_idx", self.t_pos_idx)
        rec.put("uv_v_tex", self.v_tex)
        rec.put("uv_v_nrm", self.v_nrm)
        rec.put("uv_v_tng", self.v_tng)
        return r

    Mesh.unwrap_uv = unwrap_uv

    for method, names in (("rasterize", ["bake_rast"]), ("get_mask", ["bake_mask"]), ("interpolate", ["bake_pos", "bake_nrm", "bake_tng"])):
        orig = getattr(TextureBaker, method)
        calls = {"n": 0}

        def make(orig, names, calls):
            def wrapped(self, *a, **k):
                r = orig(self, *a, **k)
                rec.put(names[min(calls["n"], len(names) - 1)], r)
                calls["n"] += 1
                if calls["n"] >= len(names) and len(names) > 1:
                    calls["n"] = 0
                return r

            return wrapped

        setattr(TextureBaker, method, make(orig, names, calls))

    # the textures before they are made 8-bit: the first call is the albedo, the second the normal
    orig_u8 = system.float32_to_uint8_np
    u8_calls = {"n": 0}

    def float32_to_uint8_np(x, *a, **k):
        import torch as _t

        rec.put("tex_albedo" if u8_calls["n"] % 2 == 0 else "tex_normal", _t.from_numpy(x))
        u8_calls["n"] += 1
        return orig_u8(x, *a, **k)

    system.float32_to_uint8_np = float32_to_uint8_np
    return u8_calls


def grounded(mesh):
    """The same as the container's: the origin at the middle of the base."""
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    return mesh


def final_tensors(rec, mesh):
    import numpy as np
    import torch

    mat = mesh.visual.material
    rec.put("final_vertices", torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)))
    rec.put("final_faces", torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)))
    rec.put("final_uv", torch.from_numpy(np.asarray(mesh.visual.uv, dtype=np.float32)))
    rec.put("final_albedo_u8", torch.from_numpy(np.array(mat.baseColorTexture, dtype=np.uint8)))
    if mat.normalTexture is not None:
        rec.put("final_normal_u8", torch.from_numpy(np.array(mat.normalTexture, dtype=np.uint8)))
    rec.put("final_roughness", torch.tensor([float(mat.roughnessFactor)]))
    rec.put("final_metallic", torch.tensor([float(mat.metallicFactor)]))


def heads_on_fixed_points(model, rec, state, scene_codes, direct_codes, mesh_v_pos):
    """The decoder (all its heads) over 2048 points of the cube and 2048 on the surface, and the global estimator, which the container does not
    use (estimate_illumination is off) but whose weights the checkpoint holds."""
    import torch

    g = torch.Generator().manual_seed(1234)
    r = model.cfg.radius
    cube = (torch.rand(2048, 3, generator=g) * 2 - 1) * r
    idx = torch.randint(0, mesh_v_pos.shape[0], (2048,), generator=g)
    surface = mesh_v_pos.cpu()[idx] + (torch.rand(2048, 3, generator=g) - 0.5) * 0.01
    points = torch.cat([cube, surface], 0).to(scene_codes.device)
    state["pause"] = True  # these calls are the script's, not the pipeline's
    try:
        with torch.no_grad():
            values = model.query_triplane(points, scene_codes[0])[0]
            decoded = model.decoder(values)
            sg = model.global_estimator(direct_codes)
    finally:
        state["pause"] = False
    rec.put("heads_points", points)
    rec.put("heads_query", values)
    for k, v in decoded.items():
        rec.put(f"heads_{k}", v)
    for k, v in sg.items():
        rec.put(f"heads_global_{k}", v)


def camera_constants(model, rec):
    """The default camera and the intrinsics, as run_image builds them."""
    from sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w

    c2w = default_cond_c2w(model.cfg.default_distance)
    intrinsic, normed = create_intrinsic_from_fov_deg(model.cfg.default_fovy_deg, model.cfg.cond_image_size, model.cfg.cond_image_size)
    rec.put("c2w_cond", c2w)
    rec.put("intrinsic_cond", intrinsic)
    rec.put("intrinsic_normed_cond", normed)


def cutout_fixture(out_dir, ckpt):
    """One picture with no transparency through the cut-out the container uses (rembg + u2net), to pin the port's decoder + u2net stage."""
    import numpy as np
    import rembg
    from PIL import Image

    os.environ.setdefault("U2NET_HOME", os.path.join(ckpt, "rembg"))
    src = Image.open(os.path.join(EXAMPLES, "stylized-rocks.png")).convert("RGBA")
    flat = Image.new("RGB", src.size, (255, 255, 255))
    flat.paste(src, mask=src.getchannel("A"))
    buf = io.BytesIO()
    flat.save(buf, format="PNG")
    session = rembg.new_session("u2net")
    t = time.time()
    rgba = rembg.remove(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), session=session)
    seconds = time.time() - t
    os.makedirs(os.path.join(out_dir, "cutout"), exist_ok=True)
    with open(os.path.join(out_dir, "cutout", "input.png"), "wb") as f:
        f.write(buf.getvalue())
    import torch

    from safetensors.torch import save_file

    arr = np.asarray(rgba, dtype=np.uint8)
    os.makedirs(os.path.join(out_dir, "cutout", "big"), exist_ok=True)
    save_file({"rgba": torch.from_numpy(arr.copy())}, os.path.join(out_dir, "cutout", "big", "rgba.safetensors"))
    with open(os.path.join(out_dir, "cutout", "stats.json"), "w") as f:
        json.dump({"size": list(rgba.size), "mode": rgba.mode, "seconds": round(seconds, 3), "rembg_model": "u2net"}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--pictures", default=",".join(PICTURES))
    ap.add_argument("--bakes", default=",".join(str(b) for b in BAKES))
    ap.add_argument("--no-cutout", action="store_true")
    args = ap.parse_args()

    sf3d_dir = os.environ.get("SF3D_DIR", "/opt/stable-fast-3d")
    sys.path.insert(0, sf3d_dir)
    os.environ.setdefault("SF3D_DINO_CONFIG", os.path.join(args.ckpt, "sf3d", "dinov2-large-config.json"))
    if args.device == "cpu":
        os.environ["SF3D_USE_CPU"] = "1"

    import numpy as np
    import torch
    from PIL import Image

    torch.backends.cuda.matmul.allow_tf32 = False  # the fixtures are fp32, not tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    torch.manual_seed(0)

    from sf3d.system import SF3D as Model

    t0 = time.time()
    model = Model.from_pretrained(os.path.join(args.ckpt, "sf3d"), config_name="config.yaml", weight_name="model.safetensors")
    model.to(args.device)
    model.eval()
    load_seconds = time.time() - t0

    rec = Recorder()
    state = {"decoder_calls": 0}
    install_hooks(model, rec, state)

    os.makedirs(args.out, exist_ok=True)
    reference = {
        "sf3d_commit": os.environ.get("SF3D_COMMIT", "unknown"),
        "torch": torch.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "load_seconds": round(load_seconds, 2),
        "tf32": False,
        "pictures": {},
    }
    bakes = [int(b) for b in args.bakes.split(",")]

    for name in args.pictures.split(","):
        path = os.path.join(EXAMPLES, name + ".png")
        raw = open(path, "rb").read()
        rgba = Image.open(io.BytesIO(raw))
        rgba.load()
        obj_dir = os.path.join(args.out, name)
        os.makedirs(os.path.join(obj_dir, "big", "glb"), exist_ok=True)
        stats = {"picture": name + ".png", "picture_sha256": sha256_of(raw), "picture_size": list(rgba.size), "bakes": {}}

        for bake in bakes:
            full = bake == bakes[0]  # the stages do not depend on the bake size: record them once
            rec.on = full
            state["decoder_calls"] = 0
            np.random.seed(0)  # the normal map is dithered with numpy's generator
            torch.manual_seed(0)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t = time.time()
            with torch.no_grad():
                mesh, global_dict = model.run_image(rgba, bake_resolution=bake, remesh="none")
            if args.device == "cuda":
                torch.cuda.synchronize()
            seconds = time.time() - t
            rec.on = full
            mesh = grounded(mesh)
            glb = mesh.export(file_type="glb")
            with open(os.path.join(obj_dir, "big", "glb", f"{bake}.glb"), "wb") as f:
                f.write(glb)
            lo, hi = mesh.bounds
            stats["bakes"][str(bake)] = {
                "seconds": round(seconds, 3),
                "triangles": int(len(mesh.faces)),
                "vertices": int(len(mesh.vertices)),
                "bounds_min": [float(x) for x in lo],
                "bounds_max": [float(x) for x in hi],
                "glb_bytes": len(glb),
                "glb_sha256": sha256_of(glb),
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if args.device == "cuda" else None,
            }
            if full:
                final_tensors(rec, mesh)
                scene = rec.tensors["scene_codes"].to(args.device)
                direct = rec.tensors["direct_codes"].to(args.device)
                heads_on_fixed_points(model, rec, state, scene, direct, rec.tensors["mesh_v_pos"])
                camera_constants(model, rec)
                tensors = rec.take()
                small, big = write_tensors(tensors, os.path.join(obj_dir, "small.safetensors"), os.path.join(obj_dir, "big", "stages.safetensors"))
                stats["small_tensors"] = small
                stats["big_tensors"] = big
                stats["mesh_triangles_before_uv"] = int(tensors["mesh_t_pos_idx"].shape[0])
            rec.on = False
            print(f"{name} bake {bake}: {stats['bakes'][str(bake)]['triangles']} triangles in {seconds:.1f}s", flush=True)

        reference["pictures"][name] = stats["bakes"][str(bakes[0])]["triangles"]
        with open(os.path.join(obj_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)

    if not args.no_cutout:
        cutout_fixture(args.out, args.ckpt)

    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(reference, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
