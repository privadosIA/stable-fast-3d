#!/usr/bin/env python3
"""Baseline for the texture baker of the port (spike 7.1): times Stable Fast 3D's own `texture_baker` extension on the UV mesh of each reference picture at the three
bake sizes, and records a sha256 of what it makes, so the port can be checked bit for bit even at sizes whose tensors are too big to keep. Runs inside the
reference container (CPU, the extension's OpenMP path; see golden_all.sh).

  python bench_baker.py --fixtures /fixtures/reference --out /out/baker_bench.json
"""

import argparse
import hashlib
import json
import os
import time

import torch
from safetensors import safe_open
from texture_baker import TextureBaker

PICTURES = ["chair1", "character1", "axe", "tree", "stylized-rocks"]
SIZES = [512, 1024, 2048]


def tensor(fixtures, picture, key):
    for f in (f"{fixtures}/{picture}/small.safetensors", f"{fixtures}/{picture}/big/stages.safetensors"):
        with safe_open(f, "pt") as g:
            if key in g.keys():
                return g.get_tensor(key)
    raise KeyError(key)


def sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    baker = TextureBaker()
    result = {"threads": torch.get_num_threads(), "torch": torch.__version__, "pictures": {}}
    for name in PICTURES:
        uv = tensor(args.fixtures, name, "uv_v_tex")
        faces = tensor(args.fixtures, name, "uv_t_pos_idx")
        pos = tensor(args.fixtures, name, "uv_v_pos")
        nrm = tensor(args.fixtures, name, "uv_v_nrm")
        entry = {"triangles": int(faces.shape[0]), "sizes": {}}
        for size in SIZES:
            baker.rasterize(uv, faces, size)   # warm up
            times, times_i = [], []
            for _ in range(args.repeats):
                t = time.perf_counter()
                rast = baker.rasterize(uv, faces, size)
                times.append(time.perf_counter() - t)
                t = time.perf_counter()
                p = baker.interpolate(pos, rast, faces)
                times_i.append(time.perf_counter() - t)
            mask = baker.get_mask(rast)
            entry["sizes"][str(size)] = {
                "rasterize_ms_median": round(sorted(times)[len(times) // 2] * 1000, 2),
                "interpolate_ms_median": round(sorted(times_i)[len(times_i) // 2] * 1000, 2),
                "mask_pixels": int(mask.sum()),
                "rast_sha256": sha(rast),
                "rast_triangle_channel_sha256": sha(rast[..., 3]),
                "pos_sha256": sha(p),
                "nrm_sha256": sha(baker.interpolate(nrm, rast, faces)),
            }
        result["pictures"][name] = entry
        print(name, entry["triangles"], {k: v["rasterize_ms_median"] for k, v in entry["sizes"].items()}, flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
