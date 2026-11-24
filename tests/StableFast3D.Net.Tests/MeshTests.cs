using System.IO.Compression;
using System.Text;
using System.Text.Json;
using MusicGen.Core.Testing;
using StableFast3D.Mesh;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

[Collection(Heavy.Name)]
public class MeshTests
{
    // ---- .npy and .npz ----------------------------------------------------------------------------------------------------------------------

    /// <summary>A <c>.npy</c> as numpy writes it (version 1.0, the header padded with spaces to a multiple of 64 bytes).</summary>
    private static byte[] Npy(string descr, string shape, byte[] data, string fortran = "False", byte major = 1)
    {
        var dict = $"{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': {shape}, }}";
        var total = 10 + dict.Length + 1;
        dict += new string(' ', (64 - total % 64) % 64) + "\n";
        var header = Encoding.Latin1.GetBytes(dict);
        var o = new List<byte> { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', major, 0 };
        o.AddRange(BitConverter.GetBytes((ushort)header.Length));
        o.AddRange(header);
        o.AddRange(data);
        return o.ToArray();
    }

    private static byte[] Doubles(params double[] v) => v.SelectMany(BitConverter.GetBytes).ToArray();
    private static byte[] Longs(params long[] v) => v.SelectMany(BitConverter.GetBytes).ToArray();

    private static string Npz(TempDir dir, params (string Name, byte[] Npy)[] entries)
    {
        var path = dir.File("a.npz");
        using (var zip = ZipFile.Open(path, ZipArchiveMode.Create))
            foreach (var (name, bytes) in entries)
            {
                using var s = zip.CreateEntry(name + ".npy", CompressionLevel.Optimal).Open();
                s.Write(bytes);
            }
        return path;
    }

    [Fact]
    public void ANpzIsReadByArrayNameWithItsTypeAndShape()
    {
        using var dir = new TempDir();
        var arrays = NpzReader.Read(Npz(dir, ("vertices", Npy("<f8", "(2, 3)", Doubles(1, 2, 3, 4, 5, 6))), ("indices", Npy("<i8", "(3,)", Longs(7, 8, 9)))));
        Assert.Equal(new[] { "indices", "vertices" }, arrays.Keys.Order().ToArray());
        Assert.Equal("<f8", arrays["vertices"].Descr);
        Assert.Equal(new long[] { 2, 3 }, arrays["vertices"].Shape);
        Assert.Equal(new double[] { 1, 2, 3, 4, 5, 6 }, arrays["vertices"].ToTensor().data<double>().ToArray());
        Assert.Equal(new long[] { 7, 8, 9 }, arrays["indices"].ToTensor().data<long>().ToArray());
    }

    [Fact]
    public void ANpyThatIsNotWhatItClaimsIsRefused()
    {
        Assert.Contains("not a .npy", Assert.Throws<InvalidDataException>(() => NpzReader.ParseNpy("not numpy at all"u8.ToArray(), "x")).Message);
        Assert.Contains("Fortran", Assert.Throws<InvalidDataException>(() => NpzReader.ParseNpy(Npy("<f8", "(1, 2)", Doubles(1, 2), fortran: "True"), "x")).Message);
        Assert.Contains("holds 8 bytes", Assert.Throws<InvalidDataException>(() => NpzReader.ParseNpy(Npy("<f8", "(2,)", Doubles(1)), "x")).Message);
        Assert.Contains("version 9", Assert.Throws<InvalidDataException>(() => NpzReader.ParseNpy(Npy("<f8", "(1,)", Doubles(1), major: 9), "x")).Message);
        Assert.Throws<NotSupportedException>(() => NpzReader.ParseNpy(Npy("<c16", "(1,)", new byte[16]), "x").ToTensor());
    }

    [Fact]
    public void TheGridIsReadAsFloat32VerticesAndInt64Tetrahedra()
    {
        using var dir = new TempDir();
        using var grid = new TetrahedraGrid(Npz(dir, ("vertices", Npy("<f8", "(4, 3)", Doubles(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1))), ("indices", Npy("<i8", "(1, 4)", Longs(0, 1, 2, 3)))));
        Assert.Equal((ScalarType.Float32, ScalarType.Int64), (grid.Vertices.dtype, grid.Indices.dtype));
        Assert.Equal(new long[] { 4, 3 }, grid.Vertices.shape);
    }

    [Fact]
    public void AGridThatIsNotOneIsRefusedByWhatIsWrong()
    {
        using var dir = new TempDir();
        var path = Npz(dir, ("vertices", Npy("<f8", "(4, 3)", Doubles(new double[12]))));
        Assert.Contains("'vertices' and 'indices'", Assert.Throws<InvalidDataException>(() => new TetrahedraGrid(path)).Message);
        using var dir2 = new TempDir();
        var bad = Npz(dir2, ("vertices", Npy("<f8", "(4, 3)", Doubles(new double[12]))), ("indices", Npy("<i8", "(1, 4)", Longs(0, 1, 2, 9))));
        Assert.Contains("vertex that is not there", Assert.Throws<InvalidDataException>(() => new TetrahedraGrid(bad)).Message);
    }

    // ---- marching tetrahedra, by hand -------------------------------------------------------------------------------------------------------

    private static TetrahedraGrid Grid(TempDir dir, double[] vertices, long[] tets) =>
        new(Npz(dir, ("vertices", Npy("<f8", $"({vertices.Length / 3}, 3)", Doubles(vertices))), ("indices", Npy("<i8", $"({tets.Length / 4}, 4)", Longs(tets)))));

    private static readonly double[] UnitTet = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1];

    [Fact]
    public void OneInsideVertexMakesOneTriangleAtTheMiddleOfItsThreeEdges()
    {
        using var dir = new TempDir();
        using var grid = Grid(dir, UnitTet, [0, 1, 2, 3]);
        var mesh = new MarchingTetrahedra(grid).Extract(grid.Vertices, tensor(new float[] { 1, -1, -1, -1 }, [4, 1]));
        Assert.Equal(new long[] { 3, 3 }, mesh.Vertices.shape);
        // vertices in the order of the sorted edges 01, 02, 03; the triangle is the table's (edge 02, edge 01, edge 03)
        Assert.Equal(new float[] { 0.5f, 0, 0, 0, 0.5f, 0, 0, 0, 0.5f }, mesh.Vertices.data<float>().ToArray());
        Assert.Equal(new long[] { 1, 0, 2 }, mesh.Triangles.data<long>().ToArray());
    }

    [Fact]
    public void TheVertexIsWhereTheValueCrossesZeroOnTheEdge()
    {
        using var dir = new TempDir();
        using var grid = Grid(dir, UnitTet, [0, 1, 2, 3]);
        var mesh = new MarchingTetrahedra(grid).Extract(grid.Vertices, tensor(new float[] { 3, -1, -1, -1 }, [4, 1]));
        Assert.Equal(0.75f, mesh.Vertices[0, 0].item<float>(), 6);   // 3 at one end and -1 at the other: zero three quarters of the way
    }

    [Fact]
    public void TwoInsideVerticesMakeTwoTrianglesAndNoneOrAllMakeNothing()
    {
        using var dir = new TempDir();
        using var grid = Grid(dir, UnitTet, [0, 1, 2, 3]);
        var march = new MarchingTetrahedra(grid);
        Assert.Equal(2, march.Extract(grid.Vertices, tensor(new float[] { 1, 1, -1, -1 }, [4, 1])).TriangleCount);
        Assert.Equal(0, march.Extract(grid.Vertices, tensor(new float[] { 1, 1, 1, 1 }, [4, 1])).TriangleCount);
        Assert.Equal(0, march.Extract(grid.Vertices, tensor(new float[] { -1, -1, -1, -1 }, [4, 1])).TriangleCount);
        Assert.Equal(0, march.Extract(grid.Vertices, tensor(new float[] { 0, 0, 0, 0 }, [4, 1])).TriangleCount);   // above zero is inside: zero is not
    }

    [Fact]
    public void TetrahedraThatShareAnEdgeShareItsVertex()
    {
        using var dir = new TempDir();
        // two tetrahedra on the face 1-2-3; vertex 0 is inside the first one's corner, vertex 4 is outside, and the other three are outside
        using var grid = Grid(dir, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 1, 2, 3, 4]);
        var mesh = new MarchingTetrahedra(grid).Extract(grid.Vertices, tensor(new float[] { 1, -1, -1, -1, -1 }, [5, 1]));
        Assert.Equal(1, mesh.TriangleCount);      // only the first tetrahedron has both signs
        Assert.Equal(3, mesh.VertexCount);
        // a second inside vertex makes the edges of the shared face cross once for both tetrahedra
        var two = new MarchingTetrahedra(grid).Extract(grid.Vertices, tensor(new float[] { 1, -1, -1, -1, 1 }, [5, 1]));
        Assert.Equal(6, two.VertexCount);         // 01 02 03 (first tet) and 14 24 34 (second tet): six distinct edges, none counted twice
        Assert.Equal(2, two.TriangleCount);
    }

    [Fact]
    public void TheOffsetMovesAVertexAtMostHalfATetrahedron()
    {
        using var dir = new TempDir();
        using var grid = Grid(dir, UnitTet, [0, 1, 2, 3]);
        var march = new MarchingTetrahedra(grid);
        Assert.True(march.Deform(zeros(4, 3)).allclose(grid.Vertices));
        var moved = march.Deform(full(new long[] { 4, 3 }, 1000f));        // tanh saturates: 1 / resolution
        Assert.True((moved - grid.Vertices).abs().max().item<float>() <= 1f / 160 + 1e-7f);
        Assert.Equal(1f / 160, (moved - grid.Vertices)[0, 0].item<float>(), 7);
    }

    // ---- what the model leaves with -----------------------------------------------------------------------------------------------------------

    [Fact]
    public void UprightTurnsXyzToMinusYZMinusXAndReversesTheTriangles()
    {
        var mesh = new TriMesh(tensor(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, [3, 3]), tensor(new long[] { 0, 1, 2 }, [1, 3]));
        var up = ModelFinish.Upright(mesh);
        Assert.Equal(new float[] { -2, 3, -1, -5, 6, -4, -8, 9, -7 }, up.Vertices.data<float>().ToArray());
        Assert.Equal(new long[] { 2, 1, 0 }, up.Triangles.data<long>().ToArray());
    }

    [Fact]
    public void GroundedPutsTheOriginAtTheMiddleOfTheBase()
    {
        var mesh = new TriMesh(tensor(new float[] { 1, 2, 3, 5, 10, 9, 3, 6, 4 }, [3, 3]), tensor(new long[] { 0, 1, 2 }, [1, 3]));
        var g = ModelFinish.Grounded(mesh).Vertices;
        Assert.Equal(new[] { 1f, 2f, 3f, 5f, 10f, 9f }.Length, 6);
        Assert.Equal(0f, g.min(0).values[1].item<float>());                    // the base on the floor
        Assert.Equal(-g.min(0).values[0].item<float>(), g.max(0).values[0].item<float>());   // centred on x
        Assert.Equal(-g.min(0).values[2].item<float>(), g.max(0).values[2].item<float>());   // and on z
        Assert.Equal(8f, g.max(0).values[1].item<float>());                    // the height is kept
    }

    [Fact]
    public void ANoSurfaceIsRefusedAndTooManyTrianglesFailWithTheirCount()
    {
        var empty = new TriMesh(zeros(0, 3), zeros(0, 3, dtype: ScalarType.Int64));
        Assert.Contains("made no surface", Assert.Throws<ModelRefusedException>(() => ModelFinish.Check(empty)).Message);
        var one = new TriMesh(zeros(3, 3), tensor(new long[] { 0, 1, 2 }, [1, 3]));
        Assert.Equal(1, ModelFinish.Check(one));
        var ex = Assert.Throws<TooManyTrianglesException>(() => ModelFinish.Check(one, maxTriangles: 0));
        Assert.Equal("1 triangles, more than the 0 of a scene", ex.Message);
        Assert.Equal(500_000, EngineInfo.MaxTriangles);
    }

    // ---- the real grid and the reference ----------------------------------------------------------------------------------------------------

    private const string Tets = "160_tets.npz";

    [RequiresFixtures(Tets)]
    public void TheRealGridHasTheTetrahedraOfTheReference()
    {
        using var grid = new TetrahedraGrid(Fixtures.Dir.PathOf(Tets));
        Assert.Equal(new long[] { 535882, 3 }, grid.Vertices.shape);
        Assert.Equal(new long[] { 2971452, 4 }, grid.Indices.shape);
        Assert.True(grid.Vertices.max().item<float>() == 1f && grid.Vertices.min().item<float>() > -1e-6f);
    }

    [RequiresCheckpoint]
    public void TheGridOfTheWeightsFolderIsTheSameFile()
    {
        var path = Checkpoint.Weights.TetsPath;
        Assert.Equal(new FileInfo(path).Length, 15408790);
        if (Fixtures.Has(Tets)) Assert.Equal(File.ReadAllBytes(Fixtures.Dir.PathOf(Tets)), File.ReadAllBytes(path));
    }

    /// <summary>The Chamfer distance of two point sets: the mean distance of each point to the nearest of the other set, both ways, added.</summary>
    private static double Chamfer(Tensor a, Tensor b)
    {
        using var _ = no_grad();
        double OneWay(Tensor from, Tensor to)
        {
            double sum = 0;
            for (long i = 0; i < from.shape[0]; i += 2048)
            {
                using var chunk = from.narrow(0, i, Math.Min(2048, from.shape[0] - i));
                using var d = cdist(chunk, to, compute_mode: compute_mode.donot_use_mm_for_euclid_dist);   // the matrix-product shortcut is not exact for points that coincide
                sum += d.min(1).values.sum().item<float>();
            }
            return sum / from.shape[0];
        }
        return OneWay(a, b) + OneWay(b, a);
    }

    [RequiresCheckpointAndPictureFixtures(Tets)]
    public void TheMeshOfTheSceneCodesIsTheReferencesOnTheFivePictures()
    {
        using var grid = new TetrahedraGrid(Fixtures.Dir.PathOf(Tets));
        var mesher = new TriplaneMesher(grid, RealStages.Decoder, new Sf3dConfig());
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            using var _ = no_grad();
            using var scope = NewDisposeScope();
            using var scene = Fixtures.Read(Fixtures.Big(name), "scene_codes");                      // (1, 3, 40, 384, 384)
            using var densityRef = Fixtures.Read(Fixtures.Big(name), "grid_density");
            using var offsetRef = Fixtures.Read(Fixtures.Big(name), "grid_vertex_offset");
            using var vRef = Fixtures.Read(Fixtures.Small(name), "mesh_v_pos");
            using var fRef = Fixtures.Read(Fixtures.Small(name), "mesh_t_pos_idx");

            // the decoder over the grid, on the reference's own scene codes
            var (density, offset) = mesher.DecodeGrid(scene[0]);
            var scale = densityRef.abs().max().item<float>();
            var dResult = Parity.Compare(density, densityRef);
            var oResult = Parity.Compare(offset, offsetRef);
            Console.WriteLine($"{name}: density {dResult} (largest {scale:F1}), offset {oResult}");
            if (!Criterion.Passes(dResult, scale)) failures.Add($"{name} density: {dResult}, largest value {scale}");
            if (!Criterion.Passes(oResult)) failures.Add($"{name} offset: {oResult}");

            // and the whole mesh from it: the spec's limits (triangles within 1%, the box within 1% of its size, Chamfer below 1e-3 of the size)
            var mesh = mesher.Extract(scene[0]);
            var size = (vRef.max(0).values - vRef.min(0).values).norm().item<float>();
            var chamfer = Chamfer(mesh.Vertices, vRef) / size;
            var boxError = ((mesh.Vertices.max(0).values - vRef.max(0).values).abs().max().item<float>() + (mesh.Vertices.min(0).values - vRef.min(0).values).abs().max().item<float>()) / size;
            var triangles = Math.Abs(mesh.TriangleCount - fRef.shape[0]) / (double)fRef.shape[0];
            Console.WriteLine($"{name}: {mesh.TriangleCount} vs {fRef.shape[0]} triangles ({triangles:P3}), box {boxError:E2} of the size, Chamfer {chamfer:E2} of the size, same triangles {mesh.Triangles.shape.SequenceEqual(fRef.shape) && mesh.Triangles.eq(fRef).all().item<bool>()}");
            if (triangles > 0.01) failures.Add($"{name}: {mesh.TriangleCount} triangles vs {fRef.shape[0]}");
            if (boxError > 0.01) failures.Add($"{name}: bounding box off by {boxError:E2} of the size");
            if (chamfer > 1e-3) failures.Add($"{name}: Chamfer {chamfer:E2} of the size");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [RequiresPictureFixtures(Tets)]
    public void MarchingTetrahedraGiveTheReferencesMeshIndexByIndexOnTheReferencesDensity()
    {
        using var grid = new TetrahedraGrid(Fixtures.Dir.PathOf(Tets));
        var mesher = new TriplaneMesher(grid, new StableFast3D.Field.MaterialMlp(), new Sf3dConfig());
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            using var scope = NewDisposeScope();
            using var density = Fixtures.Read(Fixtures.Big(name), "grid_density");               // (1, Nv, 1)
            using var offset = Fixtures.Read(Fixtures.Big(name), "grid_vertex_offset");          // (1, Nv, 3)
            using var vRef = Fixtures.Read(Fixtures.Small(name), "mesh_v_pos");
            using var fRef = Fixtures.Read(Fixtures.Small(name), "mesh_t_pos_idx");
            var mesh = mesher.Extract(density, offset);
            var sameTriangles = mesh.Triangles.shape.SequenceEqual(fRef.shape) && mesh.Triangles.eq(fRef).all().item<bool>();
            var positions = mesh.Vertices.shape.SequenceEqual(vRef.shape) ? Parity.Compare(mesh.Vertices, vRef) : new Parity.Result(0, double.PositiveInfinity);
            Console.WriteLine($"{name}: {mesh.TriangleCount} triangles, {mesh.VertexCount} vertices, same triangles {sameTriangles}, positions {positions}");
            if (!sameTriangles) failures.Add($"{name}: triangles differ ({mesh.TriangleCount} vs {fRef.shape[0]})");
            if (!Criterion.Passes(positions)) failures.Add($"{name}: positions {positions}");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [RequiresPictureFixtures]
    public void TheFinishPutsTheBoundingBoxWhereTheReferenceHasIt()
    {
        foreach (var name in Fixtures.Pictures)
        {
            using var scope = NewDisposeScope();
            using var uvPos = Fixtures.ReadStage(name, "uv_v_pos");                    // the unwrapped mesh as it is before the turn
            using var uvIdx = Fixtures.ReadStage(name, "uv_t_pos_idx");
            var finished = ModelFinish.Grounded(ModelFinish.Upright(new TriMesh(uvPos, uvIdx)));
            using var stats = JsonDocument.Parse(Fixtures.Dir.ReadAllText($"reference/{name}/stats.json"));
            var bounds = stats.RootElement.GetProperty("bakes").GetProperty("512");
            var min = bounds.GetProperty("bounds_min").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
            var max = bounds.GetProperty("bounds_max").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
            var gotMin = finished.Vertices.min(0).values.data<float>().ToArray();
            var gotMax = finished.Vertices.max(0).values.data<float>().ToArray();
            for (var i = 0; i < 3; i++)
            {
                Assert.True(Math.Abs(gotMin[i] - min[i]) <= 1e-6, $"{name} min[{i}] {gotMin[i]} vs {min[i]}");
                Assert.True(Math.Abs(gotMax[i] - max[i]) <= 1e-6, $"{name} max[{i}] {gotMax[i]} vs {max[i]}");
            }
            Assert.Equal(0f, gotMin[1]);
        }
    }
}
