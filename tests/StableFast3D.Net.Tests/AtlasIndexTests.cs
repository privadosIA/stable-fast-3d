using System.Diagnostics;
using StableFast3D.Texture;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

/// <summary>The slot of the atlas of every triangle: upstream's C++ op, ported, on the five meshes of the golden run.</summary>
[Collection(Heavy.Name)]
public class AtlasIndexTests
{
    [Fact]
    public void TwoTrianglesThatOverlapMoveTheOneBehindToTheNextSlot()
    {
        // two coincident-ish triangles in the UV square, projected on the +x face (slot 0): the one with the larger x centroid is behind (use_max is false for slot 0)
        float[] positions = [0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 0, 0, 6, 0, 0, 5, 1, 0];
        long[] triangles = [0, 1, 2, 3, 4, 5];
        float[] faceUv = [0.1f, 0.1f, 0.9f, 0.1f, 0.1f, 0.9f, 0.15f, 0.12f, 0.8f, 0.1f, 0.1f, 0.8f];
        var slots = AtlasIndex.Assign(positions, triangles, faceUv, [0, 0]);
        Assert.Equal(new long[] { 0, 6 }, slots);
    }

    [Fact]
    public void TrianglesThatDoNotOverlapStayWhereTheyAre()
    {
        float[] positions = [0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 0, 0, 6, 0, 0, 5, 1, 0];
        float[] faceUv = [0.1f, 0.1f, 0.4f, 0.1f, 0.1f, 0.4f, 0.6f, 0.6f, 0.9f, 0.6f, 0.6f, 0.9f];
        Assert.Equal(new long[] { 2, 2 }, AtlasIndex.Assign(positions, [0, 1, 2, 3, 4, 5], faceUv, [2, 2]));
        // sharing an edge is not overlapping
        float[] touching = [0.1f, 0.1f, 0.5f, 0.1f, 0.1f, 0.5f, 0.5f, 0.1f, 0.5f, 0.5f, 0.1f, 0.5f];
        Assert.Equal(new long[] { 2, 2 }, AtlasIndex.Assign(positions, [0, 1, 2, 3, 4, 5], touching, [2, 2]));
    }

    [Fact]
    public void ADegenerateTriangleOverlapsNothing() =>
        Assert.False(TriangleOverlap.TriangleTriangle(new V2(0, 0), new V2(1, 1), new V2(2, 2), new V2(0, 0), new V2(1, 0), new V2(0, 1)));

    [Fact]
    public void ATriangleOverlapsItselfAndAContainedOneButNotAFarOne()
    {
        var (a, b, c) = (new V2(0, 0), new V2(1, 0), new V2(0, 1));
        Assert.True(TriangleOverlap.TriangleTriangle(a, b, c, a, b, c));
        Assert.True(TriangleOverlap.TriangleTriangle(a, b, c, new V2(0.1f, 0.1f), new V2(0.3f, 0.1f), new V2(0.1f, 0.3f)));
        Assert.False(TriangleOverlap.TriangleTriangle(a, b, c, new V2(2, 2), new V2(3, 2), new V2(2, 3)));
    }

    [RequiresPictureFixtures]
    public void TheAtlasIndicesAreUpstreamsOnTheFiveMeshes()
    {
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            using var scope = NewDisposeScope();
            using var pos = Fixtures.ReadUnwrap(name, "aligned_pos");
            using var faces = Fixtures.ReadStage(name, "mesh_t_pos_idx");
            using var uv = Fixtures.ReadUnwrap(name, "face_uv_rotated");
            using var index = Fixtures.ReadUnwrap(name, "face_index");
            using var expected = Fixtures.ReadUnwrap(name, "atlas_index");
            var p = pos.data<float>().ToArray();
            var f = faces.data<long>().ToArray();
            var u = uv.data<float>().ToArray();
            var i = index.data<long>().ToArray();
            var e = expected.data<long>().ToArray();

            var sw = Stopwatch.StartNew();
            var got = AtlasIndex.Assign(p, f, u, i);
            sw.Stop();
            var different = got.Zip(e, (g, x) => g != x).Count(d => d);
            var moved = got.Count(s => s >= 6);
            Console.WriteLine($"{name}: {got.Length} triangles, {moved} moved to slots 6..12, {different} different from upstream, {sw.Elapsed.TotalMilliseconds:F0} ms");
            if (different != 0) failures.Add($"{name}: {different} of {got.Length} atlas indices differ");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
