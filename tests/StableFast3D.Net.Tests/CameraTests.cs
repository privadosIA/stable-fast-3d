using MusicGen.Core.Testing;
using MusicGen.Core.Weights;
using StableFast3D.Camera;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

public class CameraTests
{
    private static string Small => Fixtures.Small("axe");

    [RequiresFixtures("reference/axe/small.safetensors")]
    public void TheDefaultCameraIsTheReferencesBitForBit()
    {
        var config = new Sf3dConfig();
        using var c2w = DefaultCamera.CameraToWorld(config.DefaultDistance);
        var (intrinsic, normed) = DefaultCamera.Intrinsics(config.DefaultFovyDeg, config.CondImageSize, config.CondImageSize);
        using var _ = intrinsic;
        using var __ = normed;
        using var e1 = Fixtures.Read(Small, "c2w_cond");
        using var e2 = Fixtures.Read(Small, "intrinsic_cond");
        using var e3 = Fixtures.Read(Small, "intrinsic_normed_cond");
        Assert.Equal(e1.data<float>().ToArray(), c2w.data<float>().ToArray());
        Assert.Equal(e2.data<float>().ToArray(), intrinsic.data<float>().ToArray());
        Assert.Equal(e3.data<float>().ToArray(), normed.data<float>().ToArray());
    }

    [Fact]
    public void TheIntrinsicsOfAFortyDegreeFieldOfViewAre()
    {
        var (k, n) = DefaultCamera.Intrinsics(40, 512, 512);
        var f = k.data<float>().ToArray();
        Assert.Equal(512 * 0.5 / Math.Tan(Math.PI / 9), f[0], 2);   // 0.5 * H / tan(fov / 2)
        Assert.Equal(256f, f[2]);
        Assert.Equal(f[0] / 512, n.data<float>()[0]);
        Assert.Equal(0.5f, n.data<float>()[2]);
    }

    [Fact]
    public void TheEmbedderJoinsTheTwoMatricesAndAppliesOneLinearLayer()
    {
        using var _ = no_grad();
        var embedder = new LinearCameraEmbedder();
        embedder.linear.get_parameter("weight")!.copy_(ones(768, 25));
        embedder.linear.get_parameter("bias")!.copy_(full(new long[] { 768 }, 0.5f));
        var out_ = embedder.forward(full(new long[] { 2, 1, 4, 4 }, 1f), full(new long[] { 2, 1, 3, 3 }, 2f));
        Assert.Equal(new long[] { 2, 1, 768 }, out_.shape);
        Assert.Equal(16 * 1f + 9 * 2f + 0.5f, out_[1, 0, 767].item<float>());
    }

    [Fact]
    public void TheParametersCarryTheNamesOfTheCheckpoint()
    {
        var names = new LinearCameraEmbedder().named_parameters().Select(p => p.name).Order().ToArray();
        Assert.Equal(new[] { "linear.bias", "linear.weight" }, names);
    }

    [Fact]
    public void ACameraOfTheWrongSizeIsRefused()
    {
        var embedder = new LinearCameraEmbedder();
        Assert.Throws<ArgumentException>(() => embedder.forward(zeros(1, 1, 4, 4), zeros(1, 1, 2, 2)));
    }

    [RequiresCheckpointAndFixtures("reference/axe/small.safetensors")]
    public void TheCameraEmbeddingMatchesTheReference()
    {
        var embedder = new LinearCameraEmbedder();
        StrictLoader.Load(embedder, Checkpoint.Weights.Index, CPU, new StrictLoadOptions("The checkpoint does not match the camera embedder:") { Prefix = "camera_embedder." });
        embedder.eval();
        using var _ = no_grad();
        using var c2w = Fixtures.Read(Small, "c2w_cond");
        using var normed = Fixtures.Read(Small, "intrinsic_normed_cond");
        using var expected = Fixtures.Read(Small, "camera_embeds");
        using var got = embedder.forward(c2w.reshape(1, 1, 4, 4), normed.reshape(1, 1, 3, 3));
        var r = Parity.Compare(got, expected);
        Assert.True(r.MaxAbs <= 1e-4 && r.CosSim >= 0.9999, r.ToString());
    }

    [Fact]
    public void ConfigDefaultsAreThoseOfTheRealConfigFile()
    {
        var c = new Sf3dConfig();
        Assert.Equal((512, 160, 0.87f), (c.CondImageSize, c.IsosurfaceResolution, c.Radius));
        Assert.Equal([0.5f, 0.5f, 0.5f], c.BackgroundColor);
        Assert.Equal((40.0, 1.6f, 10.0f), (c.DefaultFovyDeg, c.DefaultDistance, c.IsosurfaceThreshold));
    }

    [RequiresCheckpoint]
    public void TheRealConfigFileSaysTheSame() =>
        Assert.Equal(new Sf3dConfig().CondImageSize, Sf3dConfig.FromFile(Path.Combine(Checkpoint.Dir!, "sf3d", "config.yaml")).CondImageSize);

    [Fact]
    public void ConfigReadsTheScalarsItKnowsAndRejectsOnesItCannotRead()
    {
        using var dir = new TempDir();
        File.WriteAllText(dir.File("c.yaml"), "cond_image_size: 256\nradius: 0.5\nbackbone:\n  num_blocks: 4\n");
        var c = Sf3dConfig.FromFile(dir.File("c.yaml"));
        Assert.Equal((256, 0.5f, 160), (c.CondImageSize, c.Radius, c.IsosurfaceResolution));
        File.WriteAllText(dir.File("bad.yaml"), "radius: 1.2.3\n");
        Assert.Throws<InvalidDataException>(() => Sf3dConfig.FromFile(dir.File("bad.yaml")));
    }
}
