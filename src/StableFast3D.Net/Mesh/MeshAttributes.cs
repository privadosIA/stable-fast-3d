using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Mesh;

/// <summary>The per-vertex attributes <c>sf3d.models.mesh.Mesh</c> computes: normals (the area-weighted mean of the faces around a vertex) and tangents (from the UVs).</summary>
public static class MeshAttributes
{
    /// <summary><c>_compute_vertex_normal</c>: the face normals (not normalised: their length is twice the area) added to their three vertices, then normalised; a vertex whose sum is nearly zero gets (0, 0, 1).</summary>
    public static Tensor VertexNormals(Tensor vertices, Tensor triangles)
    {
        using var _ = no_grad();
        var i0 = triangles[TensorIndex.Colon, 0];
        var i1 = triangles[TensorIndex.Colon, 1];
        var i2 = triangles[TensorIndex.Colon, 2];
        var v0 = vertices.index_select(0, i0);
        var v1 = vertices.index_select(0, i1);
        var v2 = vertices.index_select(0, i2);
        var faceNormals = linalg.cross(v1 - v0, v2 - v0, dim: -1);

        var sum = zeros_like(vertices);
        sum.scatter_add_(0, i0.unsqueeze(1).repeat(1, 3), faceNormals);
        sum.scatter_add_(0, i1.unsqueeze(1).repeat(1, 3), faceNormals);
        sum.scatter_add_(0, i2.unsqueeze(1).repeat(1, 3), faceNormals);

        var fallback = tensor(new float[] { 0, 0, 1 }, device: vertices.device).to(sum.dtype);
        sum = where((sum * sum).sum(-1, keepdim: true) > 1e-20, sum, fallback);
        return F.normalize(sum, dim: 1);
    }

    /// <summary>
    /// <c>_compute_vertex_tangent</c>: per triangle, the direction of increasing u (from the position and UV differences; the denominator is clamped to at least 1e-6), averaged over
    /// the triangles of each vertex, normalised and made perpendicular to the vertex normal.
    /// </summary>
    /// <param name="vertices">(Nv, 3)</param>
    /// <param name="triangles">(Nf, 3)</param>
    /// <param name="uv">(Nv, 2): one per vertex (the unwrapped mesh has one vertex per corner)</param>
    /// <param name="normals">(Nv, 3)</param>
    public static Tensor VertexTangents(Tensor vertices, Tensor triangles, Tensor uv, Tensor normals)
    {
        using var _ = no_grad();
        var pos = new Tensor[3];
        var tex = new Tensor[3];
        var idx = new Tensor[3];
        for (var i = 0; i < 3; i++)
        {
            idx[i] = triangles[TensorIndex.Colon, i];
            pos[i] = vertices.index_select(0, idx[i]);
            tex[i] = uv.index_select(0, idx[i]);
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
}
