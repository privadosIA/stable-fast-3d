using MusicGen.Core.Testing;
using StableFast3D.Mesh;
using StableFast3D.Texture;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

/// <summary>The UV unwrapper, stage by stage against the steps the golden run recorded (<c>tools/dump_unwrap.py</c>): each stage takes the reference's own input.</summary>
[Collection(Heavy.Name)]
public class UnwrapperTests
{
    private static string Describe(Parity.Result r) => r.ToString();

    private static void Check(List<string> failures, string name, string stage, Tensor got, Tensor expected, bool exact = false)
    {
        var r = Parity.Compare(got.to(ScalarType.Float64), expected.to(ScalarType.Float64));
        Console.WriteLine($"{name} {stage}: {r}");
        if (exact ? r.MaxAbs != 0 : !Criterion.Passes(r)) failures.Add($"{name} {stage}: {r}");
    }

    // ---- by hand ----------------------------------------------------------------------------------------------------------------------------

    [Fact]
    public void VertexNormalsAreTheAreaWeightedMeanOfTheFacesAroundAVertex()
    {
        // a unit right triangle in the xy plane facing +z and a lone vertex that no face uses
        var v = tensor(new float[] { 0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 5, 5 }, [4, 3]);
        var n = MeshAttributes.VertexNormals(v, tensor(new long[] { 0, 1, 2 }, [1, 3]));
        Assert.Equal(new float[] { 0, 0, 1 }, n[0].data<float>().ToArray());
        Assert.Equal(new float[] { 0, 0, 1 }, n[3].data<float>().ToArray());       // no face: (0, 0, 1)
        // the opposite winding faces the other way
        var m = MeshAttributes.VertexNormals(v, tensor(new long[] { 0, 2, 1 }, [1, 3]));
        Assert.Equal(new float[] { 0, 0, -1 }, m[1].data<float>().ToArray());
    }

    [Fact]
    public void ASliceOfTheAtlasGoesWhereItsNumberSays()
    {
        var (ox, oy, dx, dy) = UvUnwrapper.FindSliceOffsetAndScale([0, 1, 2, 3, 4, 5, 6, 11, 12]);
        Assert.Equal(new[] { 0f, 1f / 3, 2f / 3, 0f, 1f / 3, 2f / 3 }, ox[..6], new FloatComparer(1e-7f));
        Assert.Equal(new[] { 0f, 0f, 0f, 1f / 3, 1f / 3, 1f / 3 }, oy[..6], new FloatComparer(1e-7f));
        Assert.Equal(new float[] { 3, 3, 3, 3, 3, 3, 6, 6, 2 }, dx);
        Assert.Equal(new float[] { 3, 3, 3, 3, 3, 3, 6, 6, 3 }, dy);
        Assert.Equal(1f / 6 + 2f / 3, oy[7], 5);       // slot 11, an overlap slot: its row (1) times 1/6, in the lowest third
    }

    private sealed class FloatComparer(float tolerance) : IEqualityComparer<float>
    {
        public bool Equals(float a, float b) => Math.Abs(a - b) <= tolerance;
        public int GetHashCode(float x) => 0;
    }

    // ---- against upstream -------------------------------------------------------------------------------------------------------------------

    [RequiresPictureFixtures]
    public void EveryStageMatchesUpstreamOnTheFiveMeshes()
    {
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            using var _ = no_grad();
            using var scope = NewDisposeScope();
            using var pos0 = Fixtures.ReadStage(name, "mesh_v_pos");
            using var tri = Fixtures.ReadStage(name, "mesh_t_pos_idx");
            using var nrm0 = Fixtures.ReadUnwrap(name, "vertex_normals");
            using var pos = Fixtures.ReadUnwrap(name, "aligned_pos");
            using var nrm = Fixtures.ReadUnwrap(name, "aligned_nrm");
            using var faceUv = Fixtures.ReadUnwrap(name, "face_uv");
            using var faceIndex = Fixtures.ReadUnwrap(name, "face_index");
            using var rotated = Fixtures.ReadUnwrap(name, "face_uv_rotated");
            using var atlas = Fixtures.ReadUnwrap(name, "atlas_index");
            using var placed = Fixtures.ReadUnwrap(name, "placed_uv");
            using var uniqueUv = Fixtures.ReadUnwrap(name, "unique_uv");
            using var texIndex = Fixtures.ReadUnwrap(name, "vtex_idx");

            Check(failures, name, "vertex normals", MeshAttributes.VertexNormals(pos0, tri), nrm0);

            var bbox = stack([pos.min(0).values, pos.max(0).values], dim: 0);
            var (gotUv, gotIndex) = UvUnwrapper.BoxAssignVertexToCubeFace(pos, nrm, tri, bbox);
            Check(failures, name, "cube face uv", gotUv, faceUv);
            Check(failures, name, "cube face index", gotIndex, faceIndex, exact: true);

            Check(failures, name, "rotated uv", UvUnwrapper.RotateUvSlicesConsistentSpace(pos, nrm, tri, faceUv, faceIndex), rotated);

            var (ox, oy, dx, dy) = UvUnwrapper.FindSliceOffsetAndScale(atlas.data<long>().ToArray());
            Check(failures, name, "placed uv", UvUnwrapper.DistributeIndividualUvsInAtlas(rotated, atlas, ox, oy, dx, dy, 0.02), placed);

            var (unique, inverse, _) = placed.unique(sorted: true, return_inverse: true, dim: 0);
            Check(failures, name, "unique uv", unique, uniqueUv, exact: true);
            Check(failures, name, "uv indices", inverse!.reshape(-1, 3), texIndex, exact: true);
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    /// <summary>
    /// The whole unwrapping from the raw mesh. Every stage above is exact when it is given the reference's input, but the chain cannot be bit for bit: the rotation of each cube face's
    /// UVs uses <c>atan2</c>, <c>sin</c> and <c>cos</c> of one float32 per face, and PyTorch 2.10's CPU kernels (Sleef, 1 ulp) do not round the way TorchSharp's libtorch does
    /// (for atan2 of 1574.643 and -1141.984 PyTorch gives 2.1982577 and the correctly rounded value is 2.1982574). A 1e-7 difference in the UVs is enough for the overlap tests,
    /// which compare against thresholds of that size, to decide otherwise for a handful of triangles, which then sit in another slot of the atlas.
    /// </summary>
    [RequiresPictureFixtures]
    public void TheWholeUnwrappingIsUpstreamsExceptForAHandfulOfBorderlineTriangles()
    {
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            using var _ = no_grad();
            using var scope = NewDisposeScope();
            using var pos = Fixtures.ReadStage(name, "mesh_v_pos");
            using var tri = Fixtures.ReadStage(name, "mesh_t_pos_idx");
            using var expectedPos = Fixtures.ReadStage(name, "uv_v_pos");
            using var expectedTri = Fixtures.ReadStage(name, "uv_t_pos_idx");
            using var expectedTex = Fixtures.ReadStage(name, "uv_v_tex");
            using var expectedNrm = Fixtures.ReadStage(name, "uv_v_nrm");

            var unwrapped = UnwrappedMesh.Unwrap(pos, tri);
            Check(failures, name, "vertices", unwrapped.Vertices, expectedPos, exact: true);
            Check(failures, name, "triangles", unwrapped.Triangles, expectedTri, exact: true);
            Check(failures, name, "normals", unwrapped.Normals, expectedNrm, exact: true);

            // the triangles whose UVs are not the reference's (more than rounding noise)
            var different = (unwrapped.Uv - expectedTex).abs().reshape(-1, 3, 2).amax([1, 2]).gt(1e-4).sum().item<long>();
            var share = different / (double)tri.shape[0];
            Console.WriteLine($"{name} uv: {different} of {tri.shape[0]} triangles ({share:P2}) are placed differently");
            if (share > 0.02) failures.Add($"{name}: {different} triangles ({share:P2}) are placed differently");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [RequiresPictureFixtures]
    public void TheTangentsAreUpstreamsOnTheReferencesOwnUvs()
    {
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            using var _ = no_grad();
            using var scope = NewDisposeScope();
            using var pos = Fixtures.ReadStage(name, "uv_v_pos");
            using var tri = Fixtures.ReadStage(name, "uv_t_pos_idx");
            using var uv = Fixtures.ReadStage(name, "uv_v_tex");
            using var nrm = Fixtures.ReadStage(name, "uv_v_nrm");
            using var expected = Fixtures.ReadStage(name, "uv_v_tng");
            Check(failures, name, "tangents", MeshAttributes.VertexTangents(pos, tri, uv, nrm), expected);
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
