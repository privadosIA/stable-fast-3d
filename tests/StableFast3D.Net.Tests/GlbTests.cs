using System.Text.Json;
using StableFast3D.Mesh;
using StableFast3D.Texture;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

/// <summary>The finished model and its binary glTF: assembled as <c>trimesh</c> does and written with the structure of the container's file.</summary>
[Collection(Heavy.Name)]
public class GlbTests
{
    private static BakedTextures Gray(int size) => new(size, Enumerable.Repeat((byte)128, size * size * 3).ToArray(), Enumerable.Repeat((byte)200, size * size * 3).ToArray(), 0.4f, 0.1f);

    // ---- by hand ----------------------------------------------------------------------------------------------------------------------------

    [Fact]
    public void VerticesWithTheSamePositionAndUvAreMergedInTheOrderTheyFirstAppear()
    {
        // corners of two triangles that share the position of the vertex 1 and 3: the same uv merges them, another does not
        float[] v = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        float[] uv = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0.5f, 0.5f];
        var m = ModelAssembly.Assemble(v, uv, [0, 1, 2, 3, 4, 5]);
        Assert.Equal(5, m.VertexCount);                      // vertex 3 repeats vertex 1 (same position, same uv); vertex 5 has the position of 2 but another uv
        Assert.Equal(2, m.TriangleCount);
        Assert.Equal(new uint[] { 2, 1, 0, 4, 3, 1 }, m.Indices);   // the winding is reversed: (a, b, c) becomes (c, b, a)
    }

    [Fact]
    public void ThePositionsAreTurnedToYUpAndPutOnTheFloorAndTheVIsFlipped()
    {
        // one triangle: after (x, y, z) -> (-y, z, -x) the lowest point is on y = 0 and the middle of x and z is at the origin
        float[] v = [1, 2, 3, 5, 10, 9, 3, 6, 4];
        float[] uv = [0.25f, 0.25f, 0.5f, 0.75f, 1, 0];
        var m = ModelAssembly.Assemble(v, uv, [0, 1, 2]);
        var xs = Enumerable.Range(0, 3).Select(i => m.Positions[i * 3]).ToArray();
        var ys = Enumerable.Range(0, 3).Select(i => m.Positions[i * 3 + 1]).ToArray();
        var zs = Enumerable.Range(0, 3).Select(i => m.Positions[i * 3 + 2]).ToArray();
        Assert.Equal(0f, ys.Min());
        Assert.Equal(-xs.Min(), xs.Max());
        Assert.Equal(-zs.Min(), zs.Max());
        Assert.Equal(6f, ys.Max());                          // the height is the old z range, 9 - 3
        Assert.Equal(new[] { 0.75f, 0.25f, 1f }, new[] { m.Uv[1], m.Uv[3], m.Uv[5] });
    }

    [Fact]
    public void TheFileIsABinaryGltfWithItsPicturesAndDataInside()
    {
        var mesh = ModelAssembly.Assemble([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 1, 2]);
        var glb = GlbWriter.Write(mesh, Gray(16));

        Assert.Equal("glTF", System.Text.Encoding.ASCII.GetString(glb, 0, 4));
        Assert.Equal(2u, BitConverter.ToUInt32(glb, 4));
        Assert.Equal((uint)glb.Length, BitConverter.ToUInt32(glb, 8));
        Assert.Equal(0x4E4F534Au, BitConverter.ToUInt32(glb, 16));                       // the first chunk is the JSON
        Assert.Equal(0, glb.Length % 4);

        var read = new GlbReader(glb);
        var root = read.Json.RootElement;
        Assert.False(root.GetProperty("buffers")[0].TryGetProperty("uri", out _));       // the data is in the file
        foreach (var image in root.GetProperty("images").EnumerateArray()) Assert.False(image.TryGetProperty("uri", out _));
        Assert.Equal("image/jpeg", read.Image("baseColor")!.Value.Mime);
        Assert.Equal("image/jpeg", read.Image("normalTexture")!.Value.Mime);
        var (rough, metal) = read.MaterialFactors();
        Assert.Equal(0.4f, (float)rough);
        Assert.Equal(0.1f, (float)metal);
        Assert.Equal(3, read.Accessor(0).Count);                                          // indices: the triangles ParseModel counts are count / 3
        foreach (var view in root.GetProperty("bufferViews").EnumerateArray().Take(2)) Assert.Equal(0, view.GetProperty("byteOffset").GetInt32() % 4);
    }

    [Fact]
    public void ThePicturesComeBackAsTheyWentInUpToJpegNoise()
    {
        var mesh = ModelAssembly.Assemble([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 1, 2]);
        var glb = new GlbReader(GlbWriter.Write(mesh, Gray(32)));
        var d = StableFast3D.Picture.PictureDecoder.Decode(glb.Image("baseColor")!.Value.Bytes).Image;
        Assert.Equal((32, 32), (d.Width, d.Height));
        Assert.InRange((int)d.Pixels[0], 126, 130);
        Assert.InRange((int)StableFast3D.Picture.PictureDecoder.Decode(glb.Image("normalTexture")!.Value.Bytes).Image.Pixels[0], 198, 202);
    }

    // ---- against the container's file -------------------------------------------------------------------------------------------------------

    private static ModelMesh Assembled(string name) => ModelAssembly.Assemble(new UnwrappedMesh(
        Fixtures.ReadStage(name, "uv_v_pos"), Fixtures.ReadStage(name, "uv_t_pos_idx"), Fixtures.ReadStage(name, "uv_v_tex"),
        Fixtures.ReadStage(name, "uv_v_nrm"), Fixtures.ReadStage(name, "uv_v_tng")));

    [RequiresPictureFixtures]
    public void TheGeometryIsTheContainersFileIndexByIndexOnTheFivePictures()
    {
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            var reference = new GlbReader(File.ReadAllBytes(Fixtures.Dir.PathOf($"reference/{name}/big/glb/512.glb")));
            var mine = Assembled(name);
            var (_, nIdx, refIdx) = reference.Accessor(0);
            var (_, nPos, refPos) = reference.Accessor(1);
            var (_, nUv, refUv) = reference.Accessor(2);
            Console.WriteLine($"{name}: {mine.VertexCount} vertices ({nPos} in the container's), {mine.TriangleCount} triangles ({nIdx / 3})");
            if (mine.VertexCount != nPos || mine.TriangleCount != nIdx / 3) { failures.Add($"{name}: {mine.VertexCount} vertices and {mine.TriangleCount} triangles, the container's has {nPos} and {nIdx / 3}"); continue; }

            var badIdx = mine.Indices.Where((t, i) => t != (uint)refIdx[i]).Count();
            double MaxDiff(float[] a, double[] b) => a.Select((x, i) => Math.Abs(x - b[i])).Max();
            var dPos = MaxDiff(mine.Positions, refPos);
            var dUv = MaxDiff(mine.Uv, refUv);
            Console.WriteLine($"{name}: indices differing {badIdx}, positions max diff {dPos:E2}, uv max diff {dUv:E2}");
            if (badIdx != 0) failures.Add($"{name}: {badIdx} indices differ");
            // trimesh turns the mesh with matrices that carry 6e-17 where a cosine should be 0, so a position can sit one float32 step (3e-8 here) from the exact one
            if (dPos > 1e-7) failures.Add($"{name}: positions differ by up to {dPos:E2}");
            if (dUv != 0) failures.Add($"{name}: uv differ by up to {dUv:E2}");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [RequiresPictureFixtures]
    public void TheFileHasTheStructureOfTheContainersFile()
    {
        foreach (var name in Fixtures.Pictures)
        {
            var reference = new GlbReader(File.ReadAllBytes(Fixtures.Dir.PathOf($"reference/{name}/big/glb/512.glb")));
            using var rough = Fixtures.ReadStage(name, "final_roughness");
            using var metal = Fixtures.ReadStage(name, "final_metallic");
            var textures = new BakedTextures(512, Fixtures.Bytes(Fixtures.Small(name), "final_albedo_u8"), Fixtures.Bytes(Fixtures.Small(name), "final_normal_u8"), rough.item<float>(), metal.item<float>());
            var mine = new GlbReader(GlbWriter.Write(Assembled(name), textures));

            // the same material, the same nodes, the same accessors (types, counts and bounds), the same pictures and the same sizes of the data
            foreach (var key in new[] { "scene", "scenes", "meshes", "textures", "materials", "nodes", "accessors" })
                Assert.True(SameJson(reference.Json.RootElement.GetProperty(key), mine.Json.RootElement.GetProperty(key), 1e-6), $"{name}: {key} differs");
            Assert.Equal(reference.Json.RootElement.GetProperty("images").GetArrayLength(), mine.Json.RootElement.GetProperty("images").GetArrayLength());
            var refViews = reference.Json.RootElement.GetProperty("bufferViews");
            var myViews = mine.Json.RootElement.GetProperty("bufferViews");
            foreach (var i in new[] { 0, 1, 4 }) Assert.Equal(refViews[i].GetProperty("byteLength").GetInt32(), myViews[i].GetProperty("byteLength").GetInt32());
        }
    }

    /// <summary>The same JSON, numbers compared by value (0.0 and 0 are the same number) within a tolerance.</summary>
    private static bool SameJson(JsonElement a, JsonElement b, double tolerance)
    {
        if (a.ValueKind != b.ValueKind) return false;
        switch (a.ValueKind)
        {
            case JsonValueKind.Number: return Math.Abs(a.GetDouble() - b.GetDouble()) <= tolerance;
            case JsonValueKind.String: return a.GetString() == b.GetString();
            case JsonValueKind.Array: return a.GetArrayLength() == b.GetArrayLength() && a.EnumerateArray().Zip(b.EnumerateArray(), (x, y) => SameJson(x, y, tolerance)).All(r => r);
            case JsonValueKind.Object:
                var pa = a.EnumerateObject().ToDictionary(p => p.Name, p => p.Value);
                var pb = b.EnumerateObject().ToDictionary(p => p.Name, p => p.Value);
                return pa.Count == pb.Count && pa.All(kv => pb.TryGetValue(kv.Key, out var other) && SameJson(kv.Value, other, tolerance));
            default: return true;
        }
    }

    // ---- the whole model, with the real weights ---------------------------------------------------------------------------------------------

    [RequiresCheckpointAndPictureFixtures(Tets)]
    public void TheWholeModelIsReadByTheValidatorOfTheOrchestratorSizeAndAll()
    {
        const int size = 512;
        var decoder = RealStages.Decoder;
        var config = new Sf3dConfig();
        using var grid = new TetrahedraGrid(Fixtures.Dir.PathOf(Tets));
        var mesher = new TriplaneMesher(grid, decoder, config);
        var dump = Environment.GetEnvironmentVariable("SF3D_GLB_DIR");
        foreach (var name in Fixtures.Pictures)
        {
            using var _ = no_grad();
            using var scope = NewDisposeScope();
            using var scene = Fixtures.Read(Fixtures.Big(name), "scene_codes");
            using var rough = Fixtures.ReadStage(name, "image_estimator_decoder_roughness");
            using var metal = Fixtures.ReadStage(name, "image_estimator_decoder_metallic");

            var raw = mesher.Extract(scene[0]);
            var unwrapped = UnwrappedMesh.Unwrap(raw.Vertices, raw.Triangles);
            var baked = TextureBakery.Bake(unwrapped, scene[0], decoder, size, config.Radius, rough.item<float>(), metal.item<float>());
            var mesh = ModelAssembly.Assemble(unwrapped);
            var glb = GlbWriter.Write(mesh, baked);

            Assert.True(glb.Length < 32 << 20, "the orchestrator takes files up to 32 MiB");
            Assert.True(mesh.TriangleCount <= EngineInfo.MaxTriangles);
            var read = new GlbReader(glb);
            Assert.Equal(mesh.TriangleCount, read.Accessor(0).Count / 3);
            Console.WriteLine($"{name}: {glb.Length} bytes, {mesh.VertexCount} vertices, {mesh.TriangleCount} triangles");
            if (dump is { Length: > 0 }) { Directory.CreateDirectory(dump); File.WriteAllBytes(Path.Combine(dump, $"{name}-{size}.glb"), glb); }
        }
    }

    private const string Tets = "160_tets.npz";
}
