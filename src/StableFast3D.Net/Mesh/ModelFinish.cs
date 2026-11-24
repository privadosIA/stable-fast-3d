using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Mesh;

/// <summary>The engine made no surface (the message is the sentence the caller sees).</summary>
public sealed class ModelRefusedException(string message) : Exception(message);

/// <summary>More triangles than a scene takes: the container fails the job with this sentence (the mesh of Stable Fast 3D does not depend on the detail, so there is nothing to make again).</summary>
public sealed class TooManyTrianglesException(long triangles, int limit) : Exception($"{triangles} triangles, more than the {limit} of a scene")
{
    public long Triangles { get; } = triangles;
}

/// <summary>
/// What every model leaves with (the contract of the container): it stands upright in glTF's axes (Y up), its origin is the middle of its base, it has surface and
/// no more triangles than a scene takes. For Stable Fast 3D the turn to Y up is part of <c>SF3D.generate_mesh</c> (-90 degrees about X, then 90 about Y, then the
/// faces are inverted), and the container only adds the grounding.
/// </summary>
public static class ModelFinish
{
    /// <summary>
    /// <c>rotation(-90, X)</c> then <c>rotation(90, Y)</c> is the permutation (x, y, z) to (-y, z, -x), which is exact here (numpy's matrices carry 6e-17 where
    /// a cosine should be 0; they vanish when the vertices are written as float32). The triangles' winding is reversed, as <c>trimesh.invert()</c> does.
    /// </summary>
    public static TriMesh Upright(TriMesh mesh)
    {
        var v = mesh.Vertices;
        var x = v[TensorIndex.Colon, 0];
        var y = v[TensorIndex.Colon, 1];
        var z = v[TensorIndex.Colon, 2];
        var turned = stack([-y, z, -x], 1);
        return new TriMesh(turned, mesh.Triangles.flip(1));
    }

    /// <summary>The origin at the middle of the base: the lowest point of the bounding box (Y), centred on the other two axes (the arithmetic in float64, as trimesh does).</summary>
    public static TriMesh Grounded(TriMesh mesh)
    {
        using var _ = no_grad();
        var v = mesh.Vertices.to(ScalarType.Float64);
        var lo = v.min(0).values;
        var hi = v.max(0).values;
        var shift = stack([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2]);
        return new TriMesh((v + shift).to(ScalarType.Float32), mesh.Triangles);
    }

    /// <summary>A model with no surface is refused; one with more triangles than a scene takes is <see cref="TooManyTrianglesException"/>. Returns the triangles count.</summary>
    public static long Check(TriMesh mesh, int maxTriangles = EngineInfo.MaxTriangles)
    {
        var n = mesh.TriangleCount;
        if (n == 0 || mesh.VertexCount == 0) throw new ModelRefusedException("the engine made no surface: the picture may not show an object on its own");
        if (n > maxTriangles) throw new TooManyTrianglesException(n, maxTriangles);
        return n;
    }
}
