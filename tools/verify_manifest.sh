#!/usr/bin/env bash
# Checks tests/fixtures against the sha256 table of MANIFEST.md. A committed file that is missing or different fails; a regenerated one
# (under big/, the pictures) that is missing is reported and skipped, one that is there and different fails.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root/tests/fixtures"
[ -f MANIFEST.md ] || { echo "no MANIFEST.md: run tools/golden_all.sh" >&2; exit 1; }
bad=0; ok=0; skipped=0
while IFS='|' read -r _ path sum kind _; do
  path="$(echo "$path" | tr -d ' `')"; sum="$(echo "$sum" | tr -d ' `')"; kind="$(echo "$kind" | tr -d ' ')"
  [[ "$sum" =~ ^[0-9a-f]{64}$ ]] || continue
  if [ ! -f "$path" ]; then
    if [ "$kind" = regenerated ]; then skipped=$((skipped + 1)); continue; fi
    echo "MISSING   $path"; bad=$((bad + 1)); continue
  fi
  if [ "$(sha256sum "$path" | cut -d' ' -f1)" = "$sum" ]; then ok=$((ok + 1)); else echo "DIFFERENT $path"; bad=$((bad + 1)); fi
done < MANIFEST.md
echo "$ok ok, $skipped regenerated files not present, $bad wrong"
[ "$bad" -eq 0 ]
