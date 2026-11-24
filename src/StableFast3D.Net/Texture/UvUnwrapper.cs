using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Texture;

/// <summary>
/// <c>uv_unwrapper.Unwrapper</c> (spike 7.1: the first step, the alignment of the mesh with its principal axes, which is the one that depends on a random generator).
/// The rest of the unwrapper is task 7.2.
/// </summary>
public static class UvUnwrapper
{
    /// <summary>
    /// <c>torch.pca_lowrank(A, q)</c> of a (m, 3) cloud: Halko's randomized subspace iteration (a random (3, q) start from <paramref name="generator"/>, QR, two rounds of
    /// <c>A^T</c> and <c>A</c> products with a QR after each, then an SVD of the small matrix) on the centred cloud. Returns V, (3, q): the principal directions.
    /// </summary>
    public static Tensor PcaLowrank(Tensor a, Generator generator, int q = 2, int niter = 2)
    {
        using var _ = no_grad();
        var (m, n) = (a.shape[0], a.shape[1]);
        if (m < n) throw new NotSupportedException("a cloud with fewer points than coordinates is not unwrapped");
        var centred = a - a.mean([0], keepdim: true);
        var r = randn([n, q], dtype: a.dtype, device: a.device, generator: generator);
        var qm = linalg.qr(centred.matmul(r)).Q;
        for (var i = 0; i < niter; i++)
        {
            qm = linalg.qr(centred.t().matmul(qm)).Q;
            qm = linalg.qr(centred.matmul(qm)).Q;
        }
        var b = qm.t().matmul(centred);
        var (_, _, vh) = linalg.svd(b, fullMatrices: false);
        return vh.t();
    }

    /// <summary>
    /// <c>_align_mesh_with_main_axis</c>: rotates the mesh (positions and normals) so that its first principal axis lies along the canonical axis it is closest to, the second
    /// along the next one and the third (their cross product) along what is left. Upstream seeds the global generator with 0 first; this uses a generator of its own seeded
    /// with 0, which gives the same numbers (the same algorithm) without touching, or depending on, the state other threads share.
    /// </summary>
    public static (Tensor Positions, Tensor Normals) AlignMeshWithMainAxis(Tensor positions, Tensor normals)
    {
        using var _ = no_grad();
        using var generator = new Generator(0, positions.device);
        var v = PcaLowrank(positions, generator, q: 2);
        var mainAxis = F.normalize(v[TensorIndex.Colon, 0], p: 2, dim: -1, eps: 1e-6);
        var secondary = v[TensorIndex.Colon, 1];
        secondary = F.normalize(secondary - (secondary * mainAxis).sum(-1, keepdim: true) * mainAxis, p: 2, dim: -1, eps: 1e-6);
        var third = F.normalize(linalg.cross(mainAxis, secondary, dim: -1), p: 2, dim: -1, eps: 1e-6);

        int iMain = (int)mainAxis.abs().argmax().item<long>(), iSecondary = (int)secondary.abs().argmax().item<long>(), iThird = (int)third.abs().argmax().item<long>();
        var missing = new HashSet<int> { 0, 1, 2 };
        var current = 1;
        while (new HashSet<int> { iMain, iSecondary, iThird }.Count != 3)
        {
            missing = [0, 1, 2];
            missing.ExceptWith([iMain, iSecondary, iThird]);
            var free = missing.First();
            // the third axis contributes least to the shape: it takes the free one first, then the second
            if (current == 1) iThird = free;
            else if (current == 2) iSecondary = free;
            else throw new InvalidOperationException("Could not find 3 unique axis");
            current++;
        }
        var axes = new Tensor[3];
        axes[iMain] = mainAxis;
        axes[iSecondary] = secondary;
        axes[iThird] = third;
        var rot = stack(axes, dim: 1).t();
        return (einsum("ij,nj->ni", rot, positions), einsum("ij,nj->ni", rot, normals));
    }

    // The steps of uv_unwrapper/unwrap.py after the alignment. Each is the same tensor operations in the same order as upstream (the masked assignments become selections
    // and index copies), so the numbers are upstream's. Python scalars are doubles that PyTorch casts to float32 when they meet a float32 tensor: the same happens here.

    /// <summary>
    /// <c>_box_assign_vertex_to_cube_face</c>: each triangle is projected on the face of the bounding cube its (averaged) normal points at the most; returns the UVs of its corners,
    /// (Nf, 3, 2), and the face (0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z), (Nf).
    /// </summary>
    public static (Tensor FaceUv, Tensor FaceIndex) BoxAssignVertexToCubeFace(Tensor positions, Tensor normals, Tensor triangles, Tensor bbox)
    {
        using var _ = no_grad();
        var normalized = (positions - bbox[TensorIndex.Slice(0, 1)]) / (bbox[TensorIndex.Slice(1, 2)] - bbox[TensorIndex.Slice(0, 1)]);
        normalized = 2.0 * normalized - 1.0;

        var corner = new Tensor[3];
        var cornerNormal = new Tensor[3];
        for (var i = 0; i < 3; i++)
        {
            var idx = triangles[TensorIndex.Colon, i];
            corner[i] = normalized.index_select(0, idx);
            cornerNormal[i] = normals.index_select(0, idx);
        }
        var triStack = stack(corner, dim: 1);                                       // (Nf, 3, 3)
        var faceNormal = F.normalize(stack(cornerNormal, dim: 1).sum(1), dim: -1, eps: 1e-6);

        var axis = tensor(new float[] { 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1 }, [6, 3], dtype: faceNormal.dtype, device: faceNormal.device);
        var faceNormalAxis = (faceNormal.unsqueeze(1) * axis.unsqueeze(0)).sum(-1);
        var index = faceNormalAxis.argmax(-1);                                      // (Nf)

        var abs = triStack.abs();
        var (absX, absY, absZ) = (abs[TensorIndex.Ellipsis, 0], abs[TensorIndex.Ellipsis, 1], abs[TensorIndex.Ellipsis, 2]);
        var (x, y, z) = (triStack[TensorIndex.Ellipsis, TensorIndex.Slice(0, 1)], triStack[TensorIndex.Ellipsis, TensorIndex.Slice(1, 2)], triStack[TensorIndex.Ellipsis, TensorIndex.Slice(2, 3)]);

        var alongX = (index < 2).unsqueeze(-1);                                     // faces 0, 1
        var alongY = (index >= 2).logical_and(index < 4).unsqueeze(-1);             // faces 2, 3
        var maxAxis = where(alongX, absX, where(alongY, absY, absZ));               // (Nf, 3)
        var maskAlongX = alongX.unsqueeze(-1);
        var uc = where(maskAlongX, y, x);                                           // +-x use y; the rest use x
        var vc = where((index < 4).reshape(-1, 1, 1), -z, where((index == 4).reshape(-1, 1, 1), y, -y));

        // the largest extent of each corner over all faces
        var maxDimDiv = maxAxis.max(0, keepdim: true).values;
        uc = ((uc[TensorIndex.Ellipsis, 0] / maxDimDiv + 1.0) * 0.5).clip(0, 1);
        vc = ((vc[TensorIndex.Ellipsis, 0] / maxDimDiv + 1.0) * 0.5).clip(0, 1);
        return (stack([uc, vc], -1), index);
    }

    /// <summary><c>_calculate_tangents</c>: per vertex, the tangent of the cloud's own triangles for the UVs given per face corner.</summary>
    private static Tensor CalculateTangents(Tensor positions, Tensor normals, Tensor triangles, Tensor faceUv)
    {
        var tex = faceUv.unbind(1);
        var pos = new Tensor[3];
        var idx = new Tensor[3];
        for (var i = 0; i < 3; i++)
        {
            idx[i] = triangles[TensorIndex.Colon, i];
            pos[i] = positions.index_select(0, idx[i]);
        }
        var tangents = zeros_like(normals);
        var tansum = zeros_like(normals);
        var duv1 = tex[1] - tex[0];
        var duv2 = tex[2] - tex[0];
        var dpos1 = pos[1] - pos[0];
        var dpos2 = pos[2] - pos[0];
        var tngNom = dpos1 * duv2[TensorIndex.Ellipsis, TensorIndex.Slice(1, 2)] - dpos2 * duv1[TensorIndex.Ellipsis, TensorIndex.Slice(1, 2)];
        var denom = duv1[TensorIndex.Ellipsis, TensorIndex.Slice(0, 1)] * duv2[TensorIndex.Ellipsis, TensorIndex.Slice(1, 2)]
                    - duv1[TensorIndex.Ellipsis, TensorIndex.Slice(1, 2)] * duv2[TensorIndex.Ellipsis, TensorIndex.Slice(0, 1)];
        var tang = tngNom / denom.clamp_min(1e-6);
        for (var i = 0; i < 3; i++)
        {
            var index = idx[i].unsqueeze(1).repeat(1, 3);
            tangents.scatter_add_(0, index, tang);
            tansum.scatter_add_(0, index, ones_like(tang));
        }
        tangents = tangents / tansum;
        tangents = F.normalize(tangents, dim: 1);
        return F.normalize(tangents - (tangents * normals).sum(-1, keepdim: true) * normals);
    }

    /// <summary>
    /// <c>_rotate_uv_slices_consistent_space</c>: each cube face's UVs are turned by the angle between the tangents they give and the tangents expected for that face, and refitted to [0, 1].
    /// Upstream normalises the expected tangents with <c>F.normalize(x, -1)</c>, which is <c>p = -1</c> along dim 1 (not <c>dim = -1</c>); that is kept.
    /// </summary>
    public static Tensor RotateUvSlicesConsistentSpace(Tensor positions, Tensor normals, Tensor triangles, Tensor faceUv, Tensor faceIndex)
    {
        using var _ = no_grad();
        var uv = faceUv.clone();
        var tangents = CalculateTangents(positions, normals, triangles, uv);
        var posStack = stack([-positions[TensorIndex.Ellipsis, 1], positions[TensorIndex.Ellipsis, 0], zeros_like(positions[TensorIndex.Ellipsis, 0])], dim: -1);
        var expected = F.normalize(linalg.cross(normals, linalg.cross(posStack, normals, dim: -1), dim: -1), p: -1);
        var actualTangents = tangents.index_select(0, triangles.reshape(-1)).reshape(triangles.shape[0], 3, 3);
        var expectedTangents = expected.index_select(0, triangles.reshape(-1)).reshape(triangles.shape[0], 3, 3);

        var indexMod = faceIndex % 6;
        for (var i = 0; i < 6; i++)
        {
            var selected = (indexMod == i).nonzero().squeeze(1);
            if (selected.numel() == 0) continue;
            var actualMean = actualTangents.index_select(0, selected).mean([0, 1]);
            var expectedMean = expectedTangents.index_select(0, selected).mean([0, 1]);
            var dotProduct = actualMean.dot(expectedMean);
            var crossProduct = actualMean[0] * expectedMean[1] - actualMean[1] * expectedMean[0];
            var angle = atan2(crossProduct, dotProduct);
            var c = cos(angle);
            var s = sin(angle);
            var rot = stack([stack([c, -s]), stack([s, c])]);                    // [[c, -s], [s, c]]

            var cur = uv.index_select(0, selected) * 2 - 1;                       // centred first
            var rotated = einsum("ij,nfj->nfi", rot, cur);
            rotated = (rotated - rotated.min()) / (rotated.max() - rotated.min());
            uv.index_copy_(0, selected, rotated);
        }
        return uv;
    }

    /// <summary><c>_find_slice_offset_and_scale</c>: where each slot of the atlas goes (a 3 x 2 grid of the first six, smaller grids for the overlap slots and a block for the rest) and how much it is shrunk.</summary>
    public static (float[] OffsetX, float[] OffsetY, float[] DivX, float[] DivY) FindSliceOffsetAndScale(long[] index)
    {
        const double off = 1.0 / 3.0, duplOff = 1.0 / 6.0;
        int[] offsetXVals = [0, 1, 2, 0, 1, 2], offsetYVals = [0, 0, 0, 1, 1, 1];
        double XOffset(int x, int i) => i / 6 == 0 ? off * x : duplOff * x + Math.Min(i / 6 - 1, 1) * 0.5;
        double YOffset(int x, int i) => i / 6 == 0 ? off * x : duplOff * x + off * 2;

        var n = index.Length;
        float[] ox = new float[n], oy = new float[n], dx = new float[n], dy = new float[n];
        for (var k = 0; k < n; k++)
        {
            var i = (int)index[k];
            ox[k] = (float)XOffset(offsetXVals[i % 6], i);
            oy[k] = (float)YOffset(offsetYVals[i % 6], i);
            dx[k] = i >= 6 ? 6 : 3;
            dy[k] = dx[k];
            if (i >= 12) { dx[k] = 2; dy[k] = 3; }
        }
        return (ox, oy, dx, dy);
    }

    private static Tensor HandleSliceUvs(Tensor uv, Tensor index, double islandPadding)
    {
        var uc = uv[TensorIndex.Ellipsis, 0].clone();
        var vc = uv[TensorIndex.Ellipsis, 1].clone();
        for (var slot = 6; slot < 12; slot++)
        {
            var sel = (index == slot).nonzero().squeeze(1);
            if (sel.numel() == 0) continue;
            // scale the slice but only up to a factor of 2, so its texel density stays in line with the first slice
            var u = uc.index_select(0, sel);
            var v = vc.index_select(0, sel);
            uc.index_copy_(0, sel, (u - u.min()) / (u.max() - u.min()).clip(0.5));
            vc.index_copy_(0, sel, (v - v.min()) / (v.max() - v.min()).clip(0.5));
        }
        var ucPadded = (uc * (1 - 2 * islandPadding) + islandPadding).clip(0, 1);
        var vcPadded = (vc * (1 - 2 * islandPadding) + islandPadding).clip(0, 1);
        return stack([ucPadded, vcPadded], -1);
    }

    private static Tensor HandleRemainingUvs(Tensor uv, Tensor index, double islandPadding)
    {
        var remaining = (index >= 12).nonzero().squeeze(1);
        var squaresLeft = remaining.shape[0];
        if (squaresLeft == 0) return uv;
        var uc = uv[TensorIndex.Ellipsis, 0].index_select(0, remaining);
        var vc = uv[TensorIndex.Ellipsis, 1].index_select(0, remaining);

        // the leftovers go in a grid of squares in a block 0.5 wide and 1/3 high; the sizes are computed as Python does on float32 tensors
        var ratio = 0.5 * (1.0 / 3.0);
        var mult = Math.Sqrt((double)((float)squaresLeft / (float)ratio));
        var numSquareWidth = (int)Math.Ceiling(0.5 * mult);
        var numSquareHeight = (int)Math.Ceiling((double)((float)squaresLeft / (float)numSquareWidth));
        var width = 1.0 / numSquareWidth;
        var height = 1.0 / numSquareHeight;
        var clipVal = Math.Min(width, height) * 1.5;

        uc = (uc - uc.min(1, keepdim: true).values) / (uc.amax([1], keepdim: true) - uc.amin([1], keepdim: true)).clip(clipVal);
        vc = (vc - vc.min(1, keepdim: true).values) / (vc.amax([1], keepdim: true) - vc.amin([1], keepdim: true)).clip(clipVal);

        uc = (uc * (1 - islandPadding * numSquareWidth * 0.5) + islandPadding * numSquareWidth * 0.25).clip(0, 1);
        vc = (vc * (1 - islandPadding * numSquareHeight * 0.5) + islandPadding * numSquareHeight * 0.25).clip(0, 1);
        uc = uc * width;
        vc = vc * height;

        var idx = arange(uc.shape[0], dtype: ScalarType.Int32, device: uc.device);
        var xIdx = idx % numSquareWidth;
        var yIdx = idx.floor_divide(numSquareWidth);
        uc = uc + xIdx.unsqueeze(1) * width;
        vc = vc + yIdx.unsqueeze(1) * height;
        uc = (uc * (1 - 2 * islandPadding * 0.5) + islandPadding * 0.5).clip(0, 1);
        vc = (vc * (1 - 2 * islandPadding * 0.5) + islandPadding * 0.5).clip(0, 1);

        var result = uv.clone();
        result.index_copy_(0, remaining, stack([uc, vc], -1));
        return result;
    }

    /// <summary><c>_distribute_individual_uvs_in_atlas</c>: places every triangle's UVs in its slot of the atlas; (Nf x 3, 2).</summary>
    public static Tensor DistributeIndividualUvsInAtlas(Tensor faceUv, Tensor assigned, float[] offsetX, float[] offsetY, float[] divX, float[] divY, double islandPadding)
    {
        using var _ = no_grad();
        var placed = HandleSliceUvs(faceUv, assigned, islandPadding);
        placed = HandleRemainingUvs(placed, assigned, islandPadding);
        using var ox = tensor(offsetX, device: faceUv.device).unsqueeze(1);
        using var oy = tensor(offsetY, device: faceUv.device).unsqueeze(1);
        using var dx = tensor(divX, device: faceUv.device).unsqueeze(1);
        using var dy = tensor(divY, device: faceUv.device).unsqueeze(1);
        var uc = placed[TensorIndex.Ellipsis, 0] / dx + ox;
        var vc = placed[TensorIndex.Ellipsis, 1] / dy + oy;
        return stack([uc, vc], -1).reshape(-1, 2);
    }

    /// <summary>
    /// The unwrapping (<c>Unwrapper.forward</c>): (the distinct UVs, Utex x 2, in sorted order; and for each triangle the index of the UV of each corner, Nf x 3).
    /// The mesh is aligned with its principal axes, projected on the faces of its bounding cube, each face's UVs are turned to a consistent space, the triangles that overlap in their
    /// face are moved to other slots of the atlas and everything is placed in it.
    /// </summary>
    public static (Tensor UniqueUv, Tensor TexIndex) Unwrap(Tensor positions, Tensor normals, Tensor triangles, double islandPadding = 0.02)
    {
        using var _ = no_grad();
        var (pos, nrm) = AlignMeshWithMainAxis(positions, normals);
        var bbox = stack([pos.min(0).values, pos.max(0).values], dim: 0);
        var (faceUv, faceIndex) = BoxAssignVertexToCubeFace(pos, nrm, triangles, bbox);
        faceUv = RotateUvSlicesConsistentSpace(pos, nrm, triangles, faceUv, faceIndex);
        var assigned = AssignAtlasIndex(pos, triangles, faceUv, faceIndex);
        var (ox, oy, dx, dy) = FindSliceOffsetAndScale(assigned.data<long>().ToArray());
        var placed = DistributeIndividualUvsInAtlas(faceUv, assigned, ox, oy, dx, dy, islandPadding);
        var (unique, inverse, _) = placed.unique(sorted: true, return_inverse: true, dim: 0);
        return (unique, inverse!.reshape(-1, 3));
    }

    /// <summary>The atlas slot of each triangle (the C++ op of upstream, ported in <see cref="AtlasIndex"/>), as a tensor.</summary>
    public static Tensor AssignAtlasIndex(Tensor positions, Tensor triangles, Tensor faceUv, Tensor faceIndex)
    {
        var slots = AtlasIndex.Assign(
            positions.cpu().contiguous().data<float>().ToArray(), triangles.cpu().contiguous().data<long>().ToArray(),
            faceUv.cpu().contiguous().data<float>().ToArray(), faceIndex.cpu().contiguous().data<long>().ToArray());
        return tensor(slots, device: positions.device);
    }
}
