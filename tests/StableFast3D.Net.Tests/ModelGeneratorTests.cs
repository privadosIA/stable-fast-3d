using System.Diagnostics;
using StableFast3D.Mesh;
using StableFast3D.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

/// <summary>The whole engine: a picture in, a model out, with the real weights, against the container's models of the golden run.</summary>
[Collection(Heavy.Name)]
public class ModelGeneratorTests
{
    private static readonly Lazy<ModelGenerator> Generator_ = new(() => ModelGenerator.Load(Checkpoint.Dir!, Checkpoint.Env.DeviceName));

    internal static ModelGenerator Generator => Generator_.Value;

    private static byte[] Picture(string name) => File.ReadAllBytes(Fixtures.Dir.PathOf($"pictures/{name}.png"));

    private static (float[] Positions, int Triangles) Geometry(byte[] glb)
    {
        var read = new GlbReader(glb);
        var (_, indices, _) = read.Accessor(0);
        var (_, _, positions) = read.Accessor(1);
        return (positions.Select(x => (float)x).ToArray(), indices / 3);
    }

    [RequiresCheckpoint]
    public void TheWholeCheckpointLoadsStrictlyIntoTheNetwork()
    {
        var weights = Checkpoint.Weights;
        var model = new Sf3dModel(weights.Dinov2, new Sf3dConfig().Radius);
        var names = model.named_parameters().Select(p => p.name).Concat(model.named_buffers().Select(b => b.name)).Order(StringComparer.Ordinal).ToArray();
        var expected = weights.Index.Tensors.Keys.Where(k => !Sf3dCheckpoint.IsUnused(k)).Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(1276 - 162, expected.Length);
        Assert.Equal(expected, names);                    // every key of the checkpoint is a tensor of the network, and the other way round, but the 162 it does not use
        model.Dispose();
    }

    [RequiresCheckpointAndPictureFixtures]
    public void TheModelOfEveryPictureIsTheContainersWithinTheLimitsOfTheSpec()
    {
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            var sw = Stopwatch.StartNew();
            var made = Generator.Generate(Picture(name), 512);
            sw.Stop();
            var (positions, triangles) = Geometry(made.Glb);
            var reference = File.ReadAllBytes(Fixtures.Dir.PathOf($"reference/{name}/big/glb/512.glb"));
            var (refPositions, refTriangles) = Geometry(reference);

            using var scope = NewDisposeScope();
            using var mine = tensor(positions, [positions.Length / 3, 3]);
            using var theirs = tensor(refPositions, [refPositions.Length / 3, 3]);
            var size = (theirs.max(0).values - theirs.min(0).values).norm().item<float>();
            var chamfer = StableFast3D.Net.Tests.Geometry.Chamfer(mine, theirs) / size;
            var box = ((mine.max(0).values - theirs.max(0).values).abs().max().item<float>() + (mine.min(0).values - theirs.min(0).values).abs().max().item<float>()) / size;
            var triangleError = Math.Abs(triangles - refTriangles) / (double)refTriangles;
            using var rough = Fixtures.ReadStage(name, "final_roughness");
            using var metal = Fixtures.ReadStage(name, "final_metallic");
            Console.WriteLine($"{name}: {sw.Elapsed.TotalSeconds:F1} s, {made.Glb.Length} bytes, {triangles} vs {refTriangles} triangles ({triangleError:P2}), box {box:E2} and Chamfer {chamfer:E2} of the size, roughness {made.Roughness:F6} ({rough.item<float>():F6}), metallic {made.Metallic:F6} ({metal.item<float>():F6})");

            if (triangleError > 0.01) failures.Add($"{name}: {triangles} triangles, the container's has {refTriangles}");
            if (box > 0.01) failures.Add($"{name}: bounding box off by {box:E2} of its size");
            if (chamfer > 1e-3) failures.Add($"{name}: Chamfer {chamfer:E2} of the size");
            if (Math.Abs(made.Roughness - rough.item<float>()) > 1e-5 || Math.Abs(made.Metallic - metal.item<float>()) > 1e-5) failures.Add($"{name}: roughness or metallic differ");
            Assert.Equal(triangles, made.Triangles);
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [RequiresCheckpointAndPictureFixtures]
    public void TheSamePictureGivesTheSameModelByteForByte()
    {
        var a = Generator.Generate(Picture("axe"), 512);
        var b = Generator.Generate(Picture("axe"), 512);
        Assert.Equal(a.Glb, b.Glb);
    }

    [RequiresCheckpointAndPictureFixtures("reference/cutout/input.png")]
    public void APictureWithNoTransparencyIsCutOutFirstAndStillMakesAModel()
    {
        var made = Generator.Generate(File.ReadAllBytes(Fixtures.Dir.PathOf("reference/cutout/input.png")), 512);
        Assert.InRange(made.Triangles, 1_000, EngineInfo.MaxTriangles);
        Assert.Equal("glTF", System.Text.Encoding.ASCII.GetString(made.Glb, 0, 4));
    }

    [RequiresCheckpoint]
    public void WhatIsNotAPictureIsRefusedBeforeAnythingIsMade() =>
        Assert.Throws<InvalidDataException>(() => Generator.Generate("not a picture"u8.ToArray(), 512));

    [RequiresCheckpoint]
    public void TheSurfaceResolutionIsTheGridsWhateverTheDetail() => Assert.Equal(160, Generator.SurfaceResolution);
}
