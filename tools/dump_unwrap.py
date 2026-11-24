#!/usr/bin/env python3
"""The steps of Stable Fast 3D's UV unwrapper (`uv_unwrapper.Unwrapper.forward`) on the mesh of each reference picture, for the port to match stage by stage
(spike 7.1 and task 7.2). The input is the marching-tetrahedra mesh recorded by the golden run (`mesh_v_pos`, `mesh_t_pos_idx`); the call is rebuilt step by step
and its result is checked against the UVs the golden run recorded (`uv_v_tex`), so a mistake here cannot pass silently. Runs on the CPU inside the reference
container (see golden_all.sh).

  vertex_normals   the per-vertex normals of the mesh before the unwrapping (`Mesh.v_nrm`)
  aligned_pos / aligned_nrm   after `_align_mesh_with_main_axis` (PCA with torch.manual_seed(0)), and pca_v the two principal axes it found
  face_uv, face_index         `_box_assign_vertex_to_cube_face`; face_uv_rotated after `_rotate_uv_slices_consistent_space`
  atlas_index                 the C++ op `assign_faces_uv_to_atlas_index` (0..11: the cube face, plus 6 when a triangle is occluded)
  placed_uv                   `_distribute_individual_uvs_in_atlas`; unique_uv / vtex_idx `_get_unique_face_uv`
"""

import argparse
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PICTURES = ["chair1", "character1", "axe", "tree", "stylized-rocks"]
BIG = 64 << 10   # what is over 64 KiB is regenerated (big/), not committed


def read(fixtures, picture, key):
    for f in (f"{fixtures}/{picture}/small.safetensors", f"{fixtures}/{picture}/big/stages.safetensors"):
        with safe_open(f, "pt") as g:
            if key in g.keys():
                return g.get_tensor(key)
    raise KeyError(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, os.environ.get("SF3D_DIR", "/opt/stable-fast-3d"))
    from sf3d.models.mesh import Mesh
    from uv_unwrapper import Unwrapper

    padding = 0.02   # Mesh.unwrap_uv's island_padding
    for name in PICTURES:
        v_pos = read(args.fixtures, name, "mesh_v_pos")
        t_idx = read(args.fixtures, name, "mesh_t_pos_idx")
        mesh = Mesh(v_pos.clone(), t_idx.clone())
        nrm = mesh.v_nrm
        u = Unwrapper()
        out = {"vertex_normals": nrm.clone()}

        pos, nr = u._align_mesh_with_main_axis(v_pos, nrm)
        torch.manual_seed(0)
        _, _, pca_v = torch.pca_lowrank(v_pos, q=2)
        out.update(aligned_pos=pos.clone(), aligned_nrm=nr.clone(), pca_v=pca_v.clone())

        bbox = torch.stack([pos.min(dim=0).values, pos.max(dim=0).values], dim=0)
        face_uv, face_index = u._box_assign_vertex_to_cube_face(pos, nr, t_idx, bbox)
        out.update(face_uv=face_uv.clone(), face_index=face_index.clone())
        face_uv = u._rotate_uv_slices_consistent_space(pos, nr, t_idx, face_uv, face_index)
        out["face_uv_rotated"] = face_uv.clone()
        assigned = u._assign_faces_uv_to_atlas_index(pos, t_idx, face_uv, face_index)
        out["atlas_index"] = assigned.clone()
        ox, oy, dx, dy = u._find_slice_offset_and_scale(assigned)
        placed = u._distribute_individual_uvs_in_atlas(face_uv, assigned, ox, oy, dx, dy, padding)
        out["placed_uv"] = placed.clone()
        unique_uv, vtex_idx = u._get_unique_face_uv(placed)
        out.update(unique_uv=unique_uv.clone(), vtex_idx=vtex_idx.clone())

        # the same call as the golden run made, and what it recorded
        uv, indices = Unwrapper()(v_pos, nrm, t_idx, padding)
        uv_flat = uv[indices].reshape((-1, 2))
        recorded = read(args.fixtures, name, "uv_v_tex")
        same = torch.equal(uv_flat, recorded)
        print(f"{name}: rebuilt steps equal the unwrapper's own result: {torch.equal(unique_uv, uv) and torch.equal(vtex_idx, indices)}; "
              f"equal to the golden run's uv_v_tex: {same}; atlas indices used: {sorted(set(assigned.tolist()))}", flush=True)
        if not same:
            raise SystemExit(f"{name}: the rebuilt unwrapping is not the one the golden run recorded")

        small = {k: v.contiguous() for k, v in out.items() if v.numel() * v.element_size() <= BIG}
        big = {k: v.contiguous() for k, v in out.items() if k not in small}
        os.makedirs(f"{args.out}/{name}/big", exist_ok=True)
        if small:
            save_file(small, f"{args.out}/{name}/unwrap.safetensors")
        if big:
            save_file(big, f"{args.out}/{name}/big/unwrap.safetensors")


if __name__ == "__main__":
    main()
