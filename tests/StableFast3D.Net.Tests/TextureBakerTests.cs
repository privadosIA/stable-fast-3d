using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using StableFast3D.Texture;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

/// <summary>The texture baker over the UV layouts of the five pictures: the same pixels as upstream's <c>texture_baker</c>, at the three bake sizes.</summary>
[Collection(Heavy.Name)]
public class TextureBakerTests
{
    private const string Bench = "reference/baker_bench.json";

    private static string Hash(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private static (float[] Uv, int[] Faces, float[] Pos, float[] Nrm) Layout(string name)
    {
        using var uv = Fixtures.ReadStage(name, "uv_v_tex");
        using var faces = Fixtures.ReadStage(name, "uv_t_pos_idx");
        using var pos = Fixtures.ReadStage(name, "uv_v_pos");
        using var nrm = Fixtures.ReadStage(name, "uv_v_nrm");
        return (uv.data<float>().ToArray(), faces.to(ScalarType.Int32).data<int>().ToArray(), pos.data<float>().ToArray(), nrm.data<float>().ToArray());
    }

    // ---- by hand ----------------------------------------------------------------------------------------------------------------------------

    [Fact]
    public void ATriangleCoversTheCornersOfThePixelsInsideIt()
    {
        // the triangle (0,0) (1,0) (0,1) of the UV square: the pixel point is (column / 4, 1 - row / 4)
        float[] uv = [0, 0, 1, 0, 0, 1];
        var rast = TextureBaker.Rasterize(uv, [0, 1, 2], 4);
        var mask = TextureBaker.Mask(rast);
        // row 0 is v = 1: only the point (0, 1), a corner of the triangle
        Assert.True(mask[0 * 4 + 0]);
        Assert.False(mask[0 * 4 + 1]);
        // row 2 is v = 0.5: (0, .5) (.25, .5) inside, (.5, .5) on the hypotenuse (the test is inclusive), (.75, .5) beyond it
        Assert.Equal(new[] { true, true, true, false }, mask[(2 * 4)..(2 * 4 + 4)]);
        Assert.Equal(new[] { true, true, true, true }, mask[(3 * 4)..(3 * 4 + 4)]);      // row 3 is v = .25: (.75, .25) is on the hypotenuse too
        var o = 3 * 4 + 1;
        Assert.Equal(new[] { 0.5f, 0.25f, 0.25f, 0f }, new[] { rast[o * 4], rast[o * 4 + 1], rast[o * 4 + 2], rast[o * 4 + 3] });   // u, v, w of (.25, .25), triangle 0
    }

    [Fact]
    public void WhereNothingCoversThePixelTheIndexIsMinusOneAndTheAttributeZero()
    {
        float[] uv = [0, 0, 0.1f, 0, 0, 0.1f];
        var rast = TextureBaker.Rasterize(uv, [0, 1, 2], 8);
        Assert.Equal(-1f, rast[(7 * 8 + 7) * 4 + 3]);                   // the far corner
        var attr = TextureBaker.Interpolate([1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2], rast);
        Assert.Equal(new float[] { 0, 0, 0 }, attr[((7 * 8 + 7) * 3)..((7 * 8 + 7) * 3 + 3)]);
    }

    [Fact]
    public void TheFirstTriangleOfTheTraversalWinsAPointThatTwoShare()
    {
        // two triangles that share the edge x = 0.5; the pixel point (0.5, 0.25) is on it: both hold it, one is chosen and it is always the same
        float[] uv = [0, 0, 0.5f, 0, 0.5f, 1, 1, 0, 1, 1];
        int[] faces = [0, 1, 2, 1, 3, 4];
        var a = TextureBaker.Rasterize(uv, faces, 4);
        var b = TextureBaker.Rasterize(uv, faces, 4);
        Assert.Equal(a, b);
        Assert.True(a[(3 * 4 + 2) * 4 + 3] >= 0);
    }

    [Fact]
    public void AnAttributeIsTheBarycentricBlendOfTheVertices()
    {
        float[] uv = [0, 0, 1, 0, 0, 1];
        var rast = TextureBaker.Rasterize(uv, [0, 1, 2], 4);
        var attr = TextureBaker.Interpolate([0, 0, 0, 4, 0, 0, 0, 8, 0], [0, 1, 2], rast);
        var p = 3 * 4 + 1;                                              // (.25, .25): u .5, v .25, w .25
        Assert.Equal(new[] { 1f, 2f, 0f }, attr[(p * 3)..(p * 3 + 3)]);
    }

    // ---- against upstream -------------------------------------------------------------------------------------------------------------------

    [RequiresPictureFixtures(Bench)]
    public void TheRasterizedMapIsUpstreamsBitForBitAtThreeSizes()
    {
        using var bench = JsonDocument.Parse(Fixtures.Dir.ReadAllText(Bench));
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            var (uv, faces, pos, nrm) = Layout(name);
            var reference = bench.RootElement.GetProperty("pictures").GetProperty(name);
            foreach (var size in new[] { 512, 1024, 2048 })
            {
                var expected = reference.GetProperty("sizes").GetProperty(size.ToString());
                float[] rast = [];
                var times = new List<double>();
                for (var i = 0; i < 6; i++)
                {
                    var sw = Stopwatch.StartNew();
                    rast = TextureBaker.Rasterize(uv, faces, size);
                    sw.Stop();
                    if (i > 0) times.Add(sw.Elapsed.TotalMilliseconds);   // the first run is the warm-up
                }
                var interpolateTimes = new List<double>();
                float[] p = [];
                for (var i = 0; i < 6; i++)
                {
                    var sw = Stopwatch.StartNew();
                    p = TextureBaker.Interpolate(pos, faces, rast);
                    sw.Stop();
                    if (i > 0) interpolateTimes.Add(sw.Elapsed.TotalMilliseconds);
                }
                var nrmMap = TextureBaker.Interpolate(nrm, faces, rast);
                var covered = TextureBaker.Mask(rast).Count(m => m);
                var rastOk = Hash(rast) == expected.GetProperty("rast_sha256").GetString();
                Console.WriteLine($"{name} {size}: rasterize {times.Order().ElementAt(times.Count / 2):F1} ms (upstream {expected.GetProperty("rasterize_ms_median").GetDouble():F1}), " +
                                  $"interpolate {interpolateTimes.Order().ElementAt(interpolateTimes.Count / 2):F1} ms (upstream {expected.GetProperty("interpolate_ms_median").GetDouble():F1}), " +
                                  $"covered {covered} (upstream {expected.GetProperty("mask_pixels").GetInt32()}), identical {rastOk}");
                if (!rastOk) failures.Add($"{name} {size}: the rasterized map differs from upstream's (covered {covered} vs {expected.GetProperty("mask_pixels").GetInt32()})");
                if (Hash(p) != expected.GetProperty("pos_sha256").GetString()) failures.Add($"{name} {size}: the interpolated positions differ");
                if (Hash(nrmMap) != expected.GetProperty("nrm_sha256").GetString()) failures.Add($"{name} {size}: the interpolated normals differ");
            }
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    // ---- the unwrapper's random step ---------------------------------------------------------------------------------------------------------

    [RequiresPictureFixtures]
    public void TheAlignmentWithThePrincipalAxesIsUpstreamsOnTheFiveMeshes()
    {
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            using var scope = NewDisposeScope();
            using var pos = Fixtures.ReadStage(name, "mesh_v_pos");
            using var nrm = Fixtures.ReadUnwrap(name, "vertex_normals");
            using var alignedPos = Fixtures.ReadUnwrap(name, "aligned_pos");
            using var alignedNrm = Fixtures.ReadUnwrap(name, "aligned_nrm");
            var (gotPos, gotNrm) = StableFast3D.Texture.UvUnwrapper.AlignMeshWithMainAxis(pos, nrm);
            var p = MusicGen.Core.Testing.Parity.Compare(gotPos, alignedPos);
            var n = MusicGen.Core.Testing.Parity.Compare(gotNrm, alignedNrm);
            Console.WriteLine($"{name}: aligned positions {p}, aligned normals {n}");
            if (!Criterion.Passes(p)) failures.Add($"{name} positions: {p}");
            if (!Criterion.Passes(n)) failures.Add($"{name} normals: {n}");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
