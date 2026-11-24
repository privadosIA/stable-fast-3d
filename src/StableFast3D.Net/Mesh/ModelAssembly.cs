using StableFast3D.Texture;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Mesh;

/// <summary>The finished geometry of a model, as it is written: float32 positions (x, y, z), float32 texture coordinates (u, v) with v counted from the top as glTF does, and uint32 triangle indices.</summary>
public sealed record ModelMesh(float[] Positions, float[] Uv, uint[] Indices)
{
    public int VertexCount => Positions.Length / 3;
    public int TriangleCount => Indices.Length / 3;
}

/// <summary>
/// What <c>SF3D.generate_mesh</c> and the container do to the unwrapped mesh before it is exported: <c>trimesh</c> merges the vertices that have the same position (to 1e-8) and the same
/// texture coordinates (to 1e-4), keeping the order in which they first appear; the mesh is turned to Y up and its faces inverted; the origin is put at the middle of the base; and glTF's
/// texture coordinates count v from the top. The arithmetic is float64, as trimesh's, and what is written is float32.
/// </summary>
public static class ModelAssembly
{
    public static ModelMesh Assemble(UnwrappedMesh mesh)
    {
        var vertices = mesh.Vertices.cpu().contiguous().data<float>().ToArray();
        var uv = mesh.Uv.cpu().contiguous().data<float>().ToArray();
        var triangles = mesh.Triangles.cpu().contiguous().data<long>().ToArray();
        return Assemble(vertices, uv, triangles);
    }

    /// <param name="vertices">(n x 3) float32, in the space of the triplane</param>
    /// <param name="uv">(n x 2) float32</param>
    /// <param name="triangles">(m x 3) vertex indices</param>
    public static ModelMesh Assemble(float[] vertices, float[] uv, long[] triangles)
    {
        // merge: the key is the position and the UV rounded to integers (half to even, as numpy), the index the order of first appearance
        var count = vertices.Length / 3;
        var map = new Dictionary<(long, long, long, long, long), int>(count);
        var remap = new int[count];
        var kept = new List<int>(count);
        for (var i = 0; i < count; i++)
        {
            var key = (Round(vertices[i * 3] * 1e8), Round(vertices[i * 3 + 1] * 1e8), Round(vertices[i * 3 + 2] * 1e8), Round(uv[i * 2] * 1e4), Round(uv[i * 2 + 1] * 1e4));
            if (!map.TryGetValue(key, out var index)) { index = kept.Count; map[key] = index; kept.Add(i); }
            remap[i] = index;
        }

        // turned to Y up: (x, y, z) to (-y, z, -x); the origin at the middle of the base
        var n = kept.Count;
        var pos = new double[n * 3];
        for (var k = 0; k < n; k++)
        {
            double x = vertices[kept[k] * 3], y = vertices[kept[k] * 3 + 1], z = vertices[kept[k] * 3 + 2];
            pos[k * 3] = -y; pos[k * 3 + 1] = z; pos[k * 3 + 2] = -x;
        }
        double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue, maxX = double.MinValue, maxZ = double.MinValue;
        for (var k = 0; k < n; k++)
        {
            minX = Math.Min(minX, pos[k * 3]); maxX = Math.Max(maxX, pos[k * 3]);
            minY = Math.Min(minY, pos[k * 3 + 1]);
            minZ = Math.Min(minZ, pos[k * 3 + 2]); maxZ = Math.Max(maxZ, pos[k * 3 + 2]);
        }
        var (shiftX, shiftY, shiftZ) = (-(minX + maxX) / 2, -minY, -(minZ + maxZ) / 2);
        var positions = new float[n * 3];
        var texcoords = new float[n * 2];
        for (var k = 0; k < n; k++)
        {
            positions[k * 3] = (float)(pos[k * 3] + shiftX);
            positions[k * 3 + 1] = (float)(pos[k * 3 + 1] + shiftY);
            positions[k * 3 + 2] = (float)(pos[k * 3 + 2] + shiftZ);
            texcoords[k * 2] = uv[kept[k] * 2];
            texcoords[k * 2 + 1] = (float)(1.0 - uv[kept[k] * 2 + 1]);       // glTF counts v from the top
        }

        // the triangles with their winding reversed (the faces of the turned mesh are inverted)
        var indices = new uint[triangles.Length];
        for (var f = 0; f < triangles.Length / 3; f++)
        {
            indices[f * 3] = (uint)remap[triangles[f * 3 + 2]];
            indices[f * 3 + 1] = (uint)remap[triangles[f * 3 + 1]];
            indices[f * 3 + 2] = (uint)remap[triangles[f * 3]];
        }
        return new ModelMesh(positions, texcoords, indices);
    }

    private static long Round(double v) => (long)Math.Round(v, MidpointRounding.ToEven);
}
