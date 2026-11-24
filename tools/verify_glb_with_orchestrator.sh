#!/usr/bin/env bash
# Reads the .glb files of a folder with mediaOrquestator's own validator (blender.ReadModel / ParseModel), through go's -overlay, so nothing is written to that repository:
# the temporary test tools/orchestrator/zz_sf3d_parse_test.go.txt is injected for the run. Needs go and the sibling repository (MEDIAORQUESTATOR_DIR, default ~/Proyectos/mediaOrquestator).
#
#   SF3D_GLB_DIR=/tmp/glb dotnet test ... --filter GlbTests     # writes the files (see GlbTests)
#   tools/verify_glb_with_orchestrator.sh /tmp/glb
set -euo pipefail
dir="${1:?usage: verify_glb_with_orchestrator.sh <folder with .glb files>}"
dir="$(cd "$dir" && pwd)"
root="$(cd "$(dirname "$0")/.." && pwd)"
orch="${MEDIAORQUESTATOR_DIR:-$HOME/Proyectos/mediaOrquestator}"
[ -d "$orch/internal/video/blender" ] || { echo "no mediaOrquestator at $orch (set MEDIAORQUESTATOR_DIR)" >&2; exit 1; }
command -v go >/dev/null || { echo "go is not installed" >&2; exit 1; }

overlay="$(mktemp)"
trap 'rm -f "$overlay"' EXIT
printf '{"Replace": {"%s/internal/video/blender/zz_sf3d_parse_test.go": "%s/tools/orchestrator/zz_sf3d_parse_test.go.txt"}}' "$orch" "$root" > "$overlay"
cd "$orch"
SF3D_GLB_DIR="$dir" go test -overlay "$overlay" -run TestSF3DGlbReadByParseModel -v ./internal/video/blender
