using MusicGen.Core.Testing;
using MusicGen.Core.Weights;
using StableFast3D.Field;
using StableFast3D.Vision;
using StableFast3D.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

/// <summary>The decoder over the triplane, the triplane query and the CLIP estimator of roughness and metallic.</summary>
[Collection(Heavy.Name)]
public class FieldTests
{
    private static readonly ClipVisionConfig SmallClip = new() { ImageSize = 64, PatchSize = 32, Width = 16, Layers = 2, Heads = 2, EmbedDim = 512 };

    private static string[] Names(nn.Module m) => m.named_parameters().Select(p => p.name).Concat(m.named_buffers().Select(b => b.name)).Order(StringComparer.Ordinal).ToArray();

    private static string[] Shapes(nn.Module m) => m.named_parameters().OrderBy(p => p.name, StringComparer.Ordinal).Select(p => $"{p.name} {string.Join("x", p.parameter.shape)}").ToArray();

    // ---- the decoder, small -----------------------------------------------------------------------------------------------------------------

    [Fact]
    public void EveryHeadAppliesItsBiasAndItsActivation()
    {
        using var _ = no_grad();
        var mlp = new MaterialMlp();
        foreach (var (_, p) in mlp.named_parameters()) p.zero_();
        // zero weights: every head gives its last bias
        mlp.heads["density"].get_parameter("4.bias")!.copy_(full(new long[] { 1 }, 3f));
        mlp.heads["features"].get_parameter("6.bias")!.copy_(tensor(new float[] { 0f, 2f, -2f }));
        mlp.heads["perturb_normal"].get_parameter("6.bias")!.copy_(tensor(new float[] { 3f, 4f, 0f }));
        mlp.heads["vertex_offset"].get_parameter("4.bias")!.copy_(tensor(new float[] { 0.1f, -0.2f, 0.3f }));
        var o = mlp.Forward(zeros(2, 120));
        Assert.Equal(new[] { "density", "features", "perturb_normal", "vertex_offset" }, o.Keys.ToArray());
        Assert.Equal(Math.Exp(3 - 1), o["density"][1, 0].item<float>(), 3);                     // exp(x + out_bias), out_bias = -1
        var near = new FloatComparer(1e-6f);
        Assert.Equal(new[] { 0.5f, (float)(1 / (1 + Math.Exp(-2))), (float)(1 / (1 + Math.Exp(2))) }, o["features"][0].data<float>().ToArray(), near);
        Assert.Equal(new[] { 0.6f, 0.8f, 0f }, o["perturb_normal"][0].data<float>().ToArray(), near);       // unit length
        Assert.Equal(new[] { 0.1f, -0.2f, 0.3f }, o["vertex_offset"][1].data<float>().ToArray(), near);     // no activation
    }

    [Fact]
    public void AllZeroNormalsStayZeroInsteadOfNaN()
    {
        using var _ = no_grad();
        var mlp = new MaterialMlp();
        foreach (var (_, p) in mlp.named_parameters()) p.zero_();
        Assert.True(mlp.Forward(zeros(1, 120))["perturb_normal"].eq(0).all().item<bool>());   // normalize's eps keeps a zero vector at zero
    }

    [Fact]
    public void HeadsCanBeChosenButNotBothWays()
    {
        var mlp = new MaterialMlp();
        using var x = zeros(3, 120);
        Assert.Equal(new[] { "density", "vertex_offset" }, mlp.Forward(x, include: ["density", "vertex_offset"]).Keys.ToArray());
        Assert.Equal(new[] { "features", "perturb_normal" }, mlp.Forward(x, exclude: ["density", "vertex_offset"]).Keys.ToArray());
        Assert.Throws<ArgumentException>(() => mlp.Forward(x, include: ["density"], exclude: ["features"]));
    }

    [Fact]
    public void TheDecoderParametersCarryTheNamesOfTheCheckpoint()
    {
        var names = Names(new MaterialMlp());
        Assert.Equal(28, names.Length);
        Assert.Contains("heads.density.0.weight", names);
        Assert.Contains("heads.density.4.bias", names);
        Assert.Contains("heads.features.6.weight", names);       // three hidden layers: 0, 2, 4 and the output at 6
        Assert.Contains("heads.perturb_normal.6.bias", names);
        Assert.Contains("heads.vertex_offset.4.weight", names);
        Assert.DoesNotContain("heads.density.6.weight", names);
    }

    // ---- the query, small -------------------------------------------------------------------------------------------------------------------

    private static Tensor Ramp(int size, bool alongWidth, int other)
    {
        // (H, W) whose value is the coordinate in [-1, 1] along one axis
        var line = linspace(-1, 1, size);
        return alongWidth ? line.unsqueeze(0).expand(other, size).contiguous() : line.unsqueeze(1).expand(size, other).contiguous();
    }

    [Fact]
    public void EachPlaneIsSampledAtTheRightTwoCoordinates()
    {
        // channel 0 and 1 of each plane are ramps along its width and its height: the sample says which coordinate it was given
        var s = 9;
        var planes = stack([
            stack([Ramp(s, true, s), Ramp(s, false, s)]),      // xy: x along the width, y along the height
            stack([Ramp(s, true, s), Ramp(s, false, s)]),      // xz
            stack([Ramp(s, true, s), Ramp(s, false, s)]),      // yz
        ]);                                                    // (3, 2, H, W)
        var q = TriplaneQuery.Query(tensor(new float[] { 0.3f, -0.4f, 0.6f }, [1, 3]), planes, radius: 1.0);
        Assert.Equal(new long[] { 1, 6 }, q.shape);
        var got = q.data<float>().ToArray();
        Assert.Equal(new[] { 0.3f, -0.4f, 0.3f, 0.6f, -0.4f, 0.6f }, got, new FloatComparer(1e-5f));   // x y | x z | y z
    }

    private sealed class FloatComparer(float tolerance) : IEqualityComparer<float>
    {
        public bool Equals(float a, float b) => Math.Abs(a - b) <= tolerance;
        public int GetHashCode(float x) => 0;
    }

    [Fact]
    public void ThePositionIsScaledByTheRadiusAndAPointOutsideSamplesZero()
    {
        var planes = ones(3, 2, 5, 5);
        var inside = TriplaneQuery.Query(tensor(new float[] { 0.87f, -0.87f, 0f }, [1, 3]), planes, radius: 0.87);   // on the border of the scene
        Assert.True(inside.allclose(ones(1, 6), atol: 1e-5));
        var outside = TriplaneQuery.Query(tensor(new float[] { 5f, 5f, 5f }, [1, 3]), planes, radius: 0.87);
        Assert.True(outside.eq(0).all().item<bool>());
    }

    [Fact]
    public void ABatchIsQueriedPictureByPicture()
    {
        manual_seed(2);
        var planes = randn([2, 3, 4, 6, 6]);
        var points = rand(2, 7, 3) - 0.5;
        var both = TriplaneQuery.Query(points, planes, radius: 0.5);
        Assert.Equal(new long[] { 2, 7, 12 }, both.shape);
        var second = TriplaneQuery.Query(points[1], planes[1], radius: 0.5);
        Assert.Equal(new long[] { 7, 12 }, second.shape);
        Assert.True(both[1].allclose(second, atol: 1e-6));
    }

    // ---- CLIP, small ------------------------------------------------------------------------------------------------------------------------

    [Theory]
    [InlineData(3.0, 2.0, 2.0 / 3.0)]       // (a - 1) / (a + b - 2)
    [InlineData(1.5, 0.5, 1.0)]             // b below 1: all the weight on 1
    [InlineData(0.5, 3.0, 0.0)]
    [InlineData(0.5, 0.5, 1.0)]             // both below 1: torch's arg-max of two NaNs is the first
    public void TheModeOfABetaIsTheOneTorchComputes(double a, double b, double expected)
    {
        var got = ClipHeadEstimator.BetaMode(tensor(new float[] { (float)a }), tensor(new float[] { (float)b }));
        Assert.Equal(expected, got[0].item<float>(), 6);
    }

    [Fact]
    public void TheVisionTowerGivesOneFeatureVectorPerPicture()
    {
        using var _ = no_grad();
        manual_seed(4);
        var visual = new ClipVisionTransformer(SmallClip);
        foreach (var (_, p) in visual.named_parameters()) p.copy_(randn_like(p) * 0.1);
        var x = visual.call(randn(2, 3, 64, 64));
        Assert.Equal(new long[] { 2, 512 }, x.shape);
        var again = visual.call(randn(1, 3, 64, 64));
        Assert.False(x[0].allclose(again[0]));
    }

    [Fact]
    public void ClipAttentionIsSoftmaxOfScaledDotProductsPerHead()
    {
        using var _ = no_grad();
        manual_seed(6);
        var attn = new ClipAttention(width: 8, heads: 2);
        attn.in_proj_weight.copy_(randn(24, 8) * 0.3);
        attn.in_proj_bias.copy_(randn(24) * 0.1);
        var x = randn(1, 5, 8);
        var got = attn.call(x);
        var qkv = nn.functional.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1);
        Tensor H(Tensor t) => t.reshape(5, 2, 4).permute(1, 0, 2);
        var probs = softmax(matmul(H(qkv[0]), H(qkv[1]).transpose(1, 2)) / Math.Sqrt(4), -1);
        var expected = attn.out_proj.call(matmul(probs, H(qkv[2])).permute(1, 0, 2).reshape(1, 5, 8));
        Assert.True(got.allclose(expected, rtol: 1e-4, atol: 1e-5));
    }

    [Fact]
    public void TheEstimatorParametersCarryTheNamesOfTheCheckpoint()
    {
        var names = Names(new ClipHeadEstimator(SmallClip)).ToHashSet();
        foreach (var n in new[]
        {
            "model.visual.class_embedding", "model.visual.positional_embedding", "model.visual.proj", "model.visual.conv1.weight", "model.visual.ln_pre.bias", "model.visual.ln_post.weight",
            "model.visual.transformer.resblocks.1.attn.in_proj_weight", "model.visual.transformer.resblocks.1.attn.out_proj.bias", "model.visual.transformer.resblocks.0.mlp.c_fc.weight",
            "model.visual.transformer.resblocks.0.mlp.c_proj.bias", "model.visual.transformer.resblocks.0.ln_2.weight",
            "heads.roughness.0.0.weight", "heads.roughness.0.4.bias", "heads.metallic.1.0.weight", "heads.metallic.1.2.bias", "heads.roughness.2.2.weight",
        })
            Assert.Contains(n, names);
        Assert.DoesNotContain("model.visual.conv1.bias", names);
        Assert.Equal(2 * 14, names.Count(n => n.StartsWith("heads.")));
    }

    [Fact]
    public void TheEstimatorReturnsOneValuePerPictureAndHead()
    {
        using var _ = no_grad();
        manual_seed(8);
        var est = new ClipHeadEstimator(SmallClip);
        var e = est.Estimate(rand([2, 1, 80, 80, 3]));
        Assert.Equal(new long[] { 2, 512 }, e.Features.shape);
        Assert.Equal(new[] { "roughness", "metallic" }, e.Heads.Keys.ToArray());
        Assert.Equal(new long[] { 2, 1, 1 }, e.Heads["roughness"].Value.shape);
        Assert.Equal(new long[] { 2 }, e.Heads["metallic"].Alpha.shape);
    }

    // ---- the real checkpoint ----------------------------------------------------------------------------------------------------------------

    [RequiresCheckpoint]
    public void TheModulesHaveExactlyTheKeysAndShapesOfTheCheckpoint()
    {
        var t = Checkpoint.Weights.Index.Tensors;
        string[] Expected(string prefix, Func<string, bool>? skip = null) => t.Where(x => x.Key.StartsWith(prefix, StringComparison.Ordinal) && skip?.Invoke(x.Key) != true)
            .Select(x => $"{x.Key[prefix.Length..]} {string.Join("x", x.Value.Shape)}").Order(StringComparer.Ordinal).ToArray();

        var decoder = Expected("decoder.");
        Assert.Equal(28, decoder.Length);
        Assert.Equal(decoder, Shapes(new MaterialMlp()));

        var estimator = Expected("image_estimator.", Sf3dCheckpoint.IsClipTextTower);
        Assert.Equal(152 + 28, estimator.Length);
        Assert.Equal(estimator, Shapes(new ClipHeadEstimator()));
    }

    private static readonly string[] Stats = ["roughness", "metallic"];

    [RequiresCheckpointAndPictureFixtures]
    public void TheDecoderAndTheQueryMatchTheReferenceOnTheFivePictures()
    {
        var decoder = RealStages.Decoder;
        var config = new Sf3dConfig();
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            using var _ = no_grad();
            using var scope = NewDisposeScope();
            using var scene = Fixtures.Read(Fixtures.Big(name), "scene_codes");                      // (1, 3, 40, 384, 384)
            using var points = Fixtures.Read(Fixtures.Small(name), "heads_points");                  // (4096, 3): cube and surface
            using var queryRef = Fixtures.Read(Fixtures.Big(name), "heads_query");                   // (4096, 120)

            var results = new List<(string, Parity.Result)>();
            var query = TriplaneQuery.Query(points, scene[0], config.Radius);
            results.Add(("query", Parity.Compare(query, queryRef)));
            var decoded = decoder.Forward(queryRef);                                                  // the decoder takes the reference's own query
            foreach (var head in new[] { "density", "features", "perturb_normal", "vertex_offset" })
            {
                using var e = Fixtures.Read(Fixtures.Small(name), $"heads_{head}");
                results.Add((head, Parity.Compare(decoded[head], e)));
            }
            foreach (var (stage, r) in results.Where(r => !Criterion.Passes(r.Item2))) failures.Add($"{name} {stage}: {r}");
            Console.WriteLine($"{name}: " + string.Join("; ", results.Select(r => $"{r.Item1} {r.Item2}")));
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [RequiresCheckpointAndPictureFixtures]
    public void TheClipEstimateMatchesTheReferenceOnTheFivePictures()
    {
        var estimator = RealStages.Estimator;
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            using var _ = no_grad();
            using var scope = NewDisposeScope();
            using var rgb = Fixtures.Read(Fixtures.Big(name), "rgb_cond");                           // (1, 512, 512, 3)
            using var mask = Fixtures.Read(Fixtures.Small(name), "mask_cond");                       // (1, 512, 512, 1)
            var estimate = estimator.Estimate((rgb * mask).unsqueeze(1));

            var results = new List<(string, Parity.Result, double)>();
            using var features = Fixtures.Read(Fixtures.Small(name), "clip_features");
            results.Add(("features", Parity.Compare(estimate.Features, features), 0));
            foreach (var head in Stats)
            {
                using var alpha = Fixtures.Read(Fixtures.Small(name), $"image_estimator_{head}_dist_alpha");
                using var beta = Fixtures.Read(Fixtures.Small(name), $"image_estimator_{head}_dist_beta");
                using var value = Fixtures.Read(Fixtures.Small(name), $"image_estimator_decoder_{head}");
                // the Beta parameters are diagnostics (the value is their mode and is what is used): held to their own scale
                results.Add(($"{head} alpha", Parity.Compare(estimate.Heads[head].Alpha, alpha), alpha.abs().max().item<float>()));
                results.Add(($"{head} beta", Parity.Compare(estimate.Heads[head].Beta, beta), beta.abs().max().item<float>()));
                results.Add(($"{head}", Parity.Compare(estimate.Heads[head].Value, value), 0));
            }
            foreach (var (stage, r, scale) in results.Where(r => !Criterion.Passes(r.Item2, r.Item3))) failures.Add($"{name} {stage}: {r}");
            Console.WriteLine($"{name}: " + string.Join("; ", results.Select(r => $"{r.Item1} {r.Item2}")));
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
