namespace StableFast3D.Texture;

/// <summary>
/// <c>texture_baker</c>: which triangle of the UV layout covers each pixel of the texture, and what a per-vertex attribute is there. The same two operations as
/// upstream's CPU path (<c>rasterize</c> and <c>interpolate</c>), in float32 and pixel by pixel in parallel; the result is the same numbers, not just similar ones.
/// </summary>
public static class TextureBaker
{
    /// <summary>
    /// Rasterizes a UV layout: per pixel (row by row, 4 floats) the barycentric coordinates of the pixel's point in the first triangle that holds it and the index of that
    /// triangle, or (0, 0, 0, -1) where none does. The point of a pixel is <c>(column / size, 1 - row / size)</c> (its corner, not its centre).
    /// </summary>
    /// <param name="uv">(vertices x 2) floats</param>
    /// <param name="indices">(triangles x 3) vertex indices</param>
    public static float[] Rasterize(float[] uv, int[] indices, int size)
    {
        var bvh = new UvBvh(uv, indices);
        var rast = new float[size * size * 4];
        Parallel.For(0, size, x =>      // x is the row of the output (the slow index of upstream's pixel loop)
        {
            for (var y = 0; y < size; y++)
            {
                var px = Math.Clamp((float)y / size, 0.0f, 1.0f);
                var py = 1.0f - Math.Clamp((float)x / size, 0.0f, 1.0f);
                var o = (x * size + y) * 4;
                if (bvh.Intersect(px, py, out var u, out var v, out var w, out var tri))
                {
                    rast[o] = u; rast[o + 1] = v; rast[o + 2] = w; rast[o + 3] = tri;
                }
                else
                {
                    rast[o] = 0; rast[o + 1] = 0; rast[o + 2] = 0; rast[o + 3] = -1.0f;
                }
            }
        });
        return rast;
    }

    /// <summary>Which pixels are covered by a triangle.</summary>
    public static bool[] Mask(float[] rast)
    {
        var mask = new bool[rast.Length / 4];
        for (var i = 0; i < mask.Length; i++) mask[i] = rast[i * 4 + 3] >= 0;
        return mask;
    }

    /// <summary>
    /// A per-vertex attribute of 3 floats (a position, a normal, a tangent) at every pixel: the barycentric blend of the three vertices of its triangle, zero where
    /// no triangle covers it. Returns (pixels x 3) floats.
    /// </summary>
    /// <param name="attr">(vertices x 3) floats</param>
    public static float[] Interpolate(float[] attr, int[] indices, float[] rast)
    {
        var pixels = rast.Length / 4;
        var o = new float[pixels * 3];
        Parallel.For(0, pixels, idx =>
        {
            var tri = (int)rast[idx * 4 + 3];
            if (tri < 0) return;                                   // already zero
            float b0 = rast[idx * 4], b1 = rast[idx * 4 + 1], b2 = rast[idx * 4 + 2];
            int a = indices[tri * 3] * 3, b = indices[tri * 3 + 1] * 3, c = indices[tri * 3 + 2] * 3;
            for (var k = 0; k < 3; k++) o[idx * 3 + k] = attr[a + k] * b0 + attr[b + k] * b1 + attr[c + k] * b2;
        });
        return o;
    }
}
