#!/usr/bin/env bash
# The golden run: regenerates every fixture under tests/fixtures/ with the Python reference, inside the reference container (offline, the weights of
# mediaOrquestator mounted read-only). Python lives only there; nothing else in this repository needs it.
#
#   tools/golden_all.sh                  # CPU fp32 run (the numerical truth) + one CUDA run (times, VRAM, the .glb the container makes)
#   SF3D_GOLDEN_DEVICES=cpu tools/golden_all.sh
#
# Variables:  SF3D_CKPT   the weights folder      (default ~/Proyectos/mediaOrquestator/models/model3d)
#             SF3D_IMAGE  the reference image      (default mediaorquestator/model3d:sf3d: the image built with SF3D=1)
#             SF3D_GPU    the docker --gpus value  (default all; the card of the product is the 5060 Ti: SF3D_GPU='"device=1"')
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
ckpt="${SF3D_CKPT:-$HOME/Proyectos/mediaOrquestator/models/model3d}"
image="${SF3D_IMAGE:-mediaorquestator/model3d:sf3d}"
gpu="${SF3D_GPU:-all}"
devices="${SF3D_GOLDEN_DEVICES:-cpu cuda}"
fixtures="$root/tests/fixtures"

for f in sf3d/config.yaml sf3d/model.safetensors sf3d/dinov2-large-config.json rembg/u2net.onnx; do
  [ -f "$ckpt/$f" ] || { echo "missing $ckpt/$f" >&2; exit 1; }
done
docker image inspect "$image" >/dev/null 2>&1 || { echo "the image $image is not built (docker/model3d with --build-arg SF3D=1, see mediaOrquestator)" >&2; exit 1; }

run() { # run <device> <out dir> <extra args...>
  local device="$1" out="$2"; shift 2
  local gpus=()
  [ "$device" = cuda ] && gpus=(--gpus "$gpu")
  docker run --rm "${gpus[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp -e U2NET_HOME=/ckpt/rembg \
    --entrypoint python -v "$ckpt:/ckpt:ro" -v "$root/tools:/tools:ro" -v "$out:/out" "$image" \
    -I /tools/dump_reference.py --ckpt /ckpt --out /out --device "$device" "$@"
}

for device in $devices; do
  case "$device" in
    cpu)  out="$fixtures/reference" ;;
    cuda) out="$fixtures/reference-cuda" ;;
    *) echo "unknown device $device" >&2; exit 1 ;;
  esac
  rm -rf "$out"   # only the run being redone
  mkdir -p "$out"
  echo "== $device -> $out"
  if [ "$device" = cuda ]; then run "$device" "$out" --no-cutout; else run "$device" "$out"; fi
  if [ "$device" = cpu ]; then   # Pillow's own output on tiny pictures, to check the port's resampling pixel by pixel
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint python -v "$root/tools:/tools:ro" -v "$out:/out" "$image" \
      -I /tools/dump_pil.py --out /out/pil
  fi
done

# the pictures the fixtures were made from, out of the image (they are the upstream demo pictures: not committed, see tests/fixtures/.gitignore)
mkdir -p "$fixtures/pictures"
cid="$(docker create "$image")"
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
for p in chair1 character1 axe tree stylized-rocks; do
  docker cp "$cid:/opt/stable-fast-3d/demo_files/examples/$p.png" "$fixtures/pictures/$p.png"
done
# the tetrahedra grid of the upstream repository (a file of the weights folder of the port, task 1.2)
docker cp "$cid:/opt/stable-fast-3d/load/tets/160_tets.npz" "$fixtures/160_tets.npz"

"$root/tools/write_manifest.sh"
