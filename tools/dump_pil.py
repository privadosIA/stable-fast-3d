#!/usr/bin/env python3
"""Small differential fixtures for the picture code of the port: Pillow's own output for the exact calls the reference makes, on tiny synthetic
pictures (so they are committed), to check the port's resampling pixel by pixel. Runs inside the reference container (see golden_all.sh).

  resize_rgba_*   Image.resize((w, h)) of an RGBA picture (bicubic, through premultiplied RGBa: what SF3D.prepare_image does)
  resize_rgb_*    Image.resize(size, LANCZOS) of an RGB picture (rembg's normalize: the picture to 320x320)
  resize_l_*      the same on one channel (rembg: the mask back to the picture's size)
  composite       Image.composite(rgb, transparent RGBA, mask) (rembg.naive_cutout)
  exif            what the reference does with the EXIF orientation of a JPEG (size after rembg's fix_image_orientation)
"""

import argparse
import io
import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import save_file


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(7)
    cases = {}

    def rgba(w, h):
        a = rng.randint(0, 256, (h, w, 4)).astype(np.uint8)
        alpha = a[:, :, 3]
        alpha[: h // 5] = 0        # transparent rows (premultiplying zeroes the colour)
        alpha[h // 5 : 2 * h // 5] = 255
        alpha[2 * h // 5 : 3 * h // 5] = rng.randint(1, 254, alpha[2 * h // 5 : 3 * h // 5].shape)  # partial alpha (the unpremultiply rounds)
        return a

    for name, (w, h), size in [("down", (200, 150), (100, 100)), ("up", (60, 40), (100, 100)), ("wide", (97, 61), (50, 70))]:
        a = rgba(w, h)
        cases[f"resize_rgba_{name}_in"] = t(a)
        cases[f"resize_rgba_{name}_out"] = t(np.asarray(Image.fromarray(a, "RGBA").resize(size)))

    for name, (w, h), size in [("down", (200, 150), (64, 64)), ("up", (40, 30), (90, 70))]:
        a = rng.randint(0, 256, (h, w, 3)).astype(np.uint8)
        cases[f"resize_rgb_{name}_in"] = t(a)
        cases[f"resize_rgb_{name}_out"] = t(np.asarray(Image.fromarray(a, "RGB").resize(size, Image.Resampling.LANCZOS)))
        m = rng.randint(0, 256, (h, w)).astype(np.uint8)
        cases[f"resize_l_{name}_in"] = t(m)
        cases[f"resize_l_{name}_out"] = t(np.asarray(Image.fromarray(m, "L").resize(size, Image.Resampling.LANCZOS)))

    a = rng.randint(0, 256, (40, 50, 3)).astype(np.uint8)
    m = rng.randint(0, 256, (40, 50)).astype(np.uint8)
    m[:5] = 0
    m[5:10] = 255
    cases["composite_rgb"] = t(a)
    cases["composite_mask"] = t(m)
    cases["composite_out"] = t(np.asarray(Image.composite(Image.fromarray(a, "RGB"), Image.new("RGBA", (50, 40), 0), Image.fromarray(m, "L"))))

    # EXIF orientation: a 30x20 jpeg that says "turn me" (orientation 6: 90 degrees), as the server's cutout would see it
    img = Image.fromarray(rng.randint(0, 256, (20, 30, 3)).astype(np.uint8), "RGB")
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    opened = Image.open(io.BytesIO(buf.getvalue()))
    opened.load()
    converted = opened.convert("RGB")
    turned = ImageOps.exif_transpose(converted)
    cases["exif_jpeg"] = t(np.frombuffer(buf.getvalue(), dtype=np.uint8))
    cases["exif_turned"] = t(np.asarray(turned))   # RGB, 30 rows by 20 columns
    info = {"jpeg_size": [30, 20], "orientation": 6, "after_convert_and_exif_transpose": list(turned.size)}

    save_file(cases, os.path.join(args.out, "cases.safetensors"))
    with open(os.path.join(args.out, "exif.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
