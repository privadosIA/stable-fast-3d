using StableFast3D.Mesh;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Texture;

/// <summary>
/// <c>Mesh.unwrap_uv</c>: the mesh with one vertex per triangle corner (so every corner can have its own UV, and the seams are free), the UVs, and the normals and tangents
/// recomputed for it. Its normals are those of the flat triangles: no vertex is shared.
/// </summary>
public sealed record UnwrappedMesh(Tensor Vertices, Tensor Triangles, Tensor Uv, Tensor Normals, Tensor Tangents)
{
    public static UnwrappedMesh Unwrap(Tensor vertices, Tensor triangles, double islandPadding = 0.02)
    {
        using var _ = no_grad();
        var normals = MeshAttributes.VertexNormals(vertices, triangles);
        var (uv, indices) = UvUnwrapper.Unwrap(vertices, normals, triangles, islandPadding);

        var individualVertices = vertices.index_select(0, triangles.reshape(-1)).reshape(-1, 3);
        var individualFaces = arange(individualVertices.shape[0], dtype: triangles.dtype, device: vertices.device).reshape(-1, 3);
        var uvFlat = uv.index_select(0, indices.reshape(-1)).reshape(-1, 2);

        var flatNormals = MeshAttributes.VertexNormals(individualVertices, individualFaces);
        var tangents = MeshAttributes.VertexTangents(individualVertices, individualFaces, uvFlat, flatNormals);
        return new UnwrappedMesh(individualVertices, individualFaces, uvFlat, flatNormals, tangents);
    }
}
