using MusicGen.Core.Testing;
using MusicGen.Core.Weights;
using StableFast3D.Field;
using StableFast3D.Transformer;
using StableFast3D.Vision;
using StableFast3D.Triplane;
using StableFast3D.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

/// <summary>The triplane stage: the learned planes as tokens, the two-stream transformer over them and the image tokens, and the upsampling to the scene codes.</summary>
[Collection(Heavy.Name)]
public class BackboneTests
{
    private static readonly BackboneConfig Small = new()
    {
        NumAttentionHeads = 4, AttentionHeadDim = 8, RawTriplaneChannels = 8, TriplaneChannels = 8, RawImageChannels = 6,
        NumLatents = 5, NumBlocks = 2, NumBasicBlocks = 1, NormNumGroups = 2, CrossAttentionDim = 6,
    };

    private static string[] Names(nn.Module m) => m.named_parameters().Select(p => p.name).Concat(m.named_buffers().Select(b => b.name)).Order(StringComparer.Ordinal).ToArray();

    private static string[] KeysUnder(string prefix) =>
        Checkpoint.Weights.Index.Tensors.Keys.Where(k => k.StartsWith(prefix, StringComparison.Ordinal)).Select(k => k[prefix.Length..]).Order(StringComparer.Ordinal).ToArray();

    private static string[] Shapes(nn.Module m) => m.named_parameters().OrderBy(p => p.name, StringComparer.Ordinal).Select(p => $"{p.name} {string.Join("x", p.parameter.shape)}").ToArray();

    // ---- small, no weights ------------------------------------------------------------------------------------------------------------------

    [Fact]
    public void TheTokensAreThePlanesOneAfterTheOtherEachRowByRow()
    {
        using var _ = no_grad();
        var t = new TriplaneTokenizer(planeSize: 2, numChannels: 3);
        t.embeddings.copy_(arange(3 * 3 * 2 * 2, dtype: ScalarType.Float32).reshape(3, 3, 2, 2));
        var tokens = t.Tokens(2);
        Assert.Equal(new long[] { 2, 3, 12 }, tokens.shape);
        for (var plane = 0; plane < 3; plane++)
            for (var h = 0; h < 2; h++)
                for (var w = 0; w < 2; w++)
                    for (var c = 0; c < 3; c++)
                        Assert.Equal(t.embeddings[plane, c, h, w].item<float>(), tokens[1, c, plane * 4 + h * 2 + w].item<float>());
        Assert.True(t.Detokenize(tokens).allclose(t.embeddings.unsqueeze(0).expand(2, -1, -1, -1, -1)));
    }

    [Fact]
    public void TokensThatAreNotThreePlanesOfTheRightChannelsAreRefused()
    {
        var t = new TriplaneTokenizer(planeSize: 2, numChannels: 3);
        Assert.Throws<ArgumentException>(() => t.Detokenize(zeros(1, 3, 11)));
        Assert.Throws<ArgumentException>(() => t.Detokenize(zeros(1, 4, 12)));
    }

    [Fact]
    public void AttentionIsSoftmaxOfScaledDotProductsPerHead()
    {
        using var _ = no_grad();
        manual_seed(5);
        var attn = new CrossAttention(dim: 8, kvDim: 6, numHeads: 2);
        var xq = randn(1, 3, 8);
        var xkv = randn(1, 4, 6);
        var got = attn.call(xq, xkv);

        // by hand: heads of 4 channels
        var q = attn.wq.call(xq).reshape(3, 2, 4).permute(1, 0, 2);     // (head, token, channel)
        var k = attn.wk.call(xkv).reshape(4, 2, 4).permute(1, 0, 2);
        var v = attn.wv.call(xkv).reshape(4, 2, 4).permute(1, 0, 2);
        var probs = softmax(matmul(q, k.transpose(1, 2)) / Math.Sqrt(4), -1);
        var expected = attn.proj.call(matmul(probs, v).permute(1, 0, 2).reshape(1, 3, 8));
        Assert.True(got.allclose(expected, rtol: 1e-4, atol: 1e-5));
    }

    [Fact]
    public void TheGatedGeluMultipliesOneHalfByTheGeluOfTheOther()
    {
        using var _ = no_grad();
        var g = new GeGlu(2, 2);
        g.proj.get_parameter("weight")!.copy_(tensor(new float[] { 1, 0, 0, 1, 1, 1, 1, -1 }, [4, 2]));   // (a, b, a + b, a - b)
        g.proj.get_parameter("bias")!.copy_(zeros(4));
        var got = g.call(tensor(new float[] { 2f, 1f }, [1, 2]));
        var gelu = (double x) => x * 0.5 * (1 + Erf(x / Math.Sqrt(2)));
        Assert.Equal(2 * gelu(3), got[0, 0].item<float>(), 4);
        Assert.Equal(1 * gelu(1), got[0, 1].item<float>(), 4);
    }

    private static double Erf(double x)   // Abramowitz-Stegun 7.1.26 would not be precise enough: series
    {
        double sum = x, term = x;
        for (var n = 1; n < 60; n++) { term *= -x * x / n; sum += term / (2 * n + 1); }
        return 2 / Math.Sqrt(Math.PI) * sum;
    }

    [Fact]
    public void TheTwoStreamTransformerKeepsTheShapeOfTheTriplaneTokens()
    {
        using var _ = no_grad();
        manual_seed(1);
        var model = new TwoStreamInterleaveTransformer(Small);
        var tokens = randn(2, 8, 12);
        var image = randn(2, 7, 6);
        var got = model.call(tokens, image);
        Assert.Equal(new long[] { 2, 8, 12 }, got.shape);
        Assert.False(got.allclose(tokens));
        // each picture is its own: the second one's tokens do not depend on the first one's image
        var alone = model.call(tokens[TensorIndex.Slice(1, 2)], image[TensorIndex.Slice(1, 2)]);
        Assert.True(got[TensorIndex.Slice(1, 2)].allclose(alone, rtol: 1e-4, atol: 1e-5), $"max diff {(got[TensorIndex.Slice(1, 2)] - alone).abs().max().item<float>()} on values up to {got.abs().max().item<float>()}");
    }

    [Fact]
    public void TheTriplaneNormStartsAsTheIdentityNotAsGarbage()
    {
        // TorchSharp leaves a GroupNorm's parameters uninitialised: a model that is not fully loaded would give NaN
        var model = new TwoStreamInterleaveTransformer(Small);
        Assert.True(model.norm_triplane.weight!.eq(1).all().item<bool>());
        Assert.True(model.norm_triplane.bias!.eq(0).all().item<bool>());
    }

    [Fact]
    public void TheBackboneParametersCarryTheNamesOfTheCheckpoint()
    {
        var names = Names(new TwoStreamInterleaveTransformer(Small)).ToHashSet();
        foreach (var n in new[]
        {
            "latent_init", "norm_triplane.weight", "proj_triplane.bias", "norm_image.weight", "proj_image.weight", "norm_latent.bias", "proj_latent.weight", "proj_out.bias",
            "main_blocks.0.fuse_block_in.attn.wq.weight", "main_blocks.0.fuse_block_in.attn.proj.bias", "main_blocks.1.fuse_block_in.norm_z1.weight",
            "main_blocks.0.fuse_block_in.ff.net.0.proj.weight", "main_blocks.0.fuse_block_in.ff.net.2.bias",
            "main_blocks.1.transformer_block.0.attn2.wk.weight", "main_blocks.1.transformer_block.0.norm3.bias", "main_blocks.0.fuse_block_out.attn.wv.weight",
        })
            Assert.Contains(n, names);
        Assert.DoesNotContain(names, n => n.Contains("norm_x") || n.EndsWith("wq.bias"));   // norm_x_input is off; the q, k, v projections have no bias
        Assert.Equal(15 + 2 * 86 - 2 * (3 - 1) * 20, names.Count);   // 15 of its own, 86 per block with 3 basic blocks of 20 (here 1 each)
    }

    [Fact]
    public void ThePixelShuffleNetworkUpsamplesEachPlane()
    {
        using var _ = no_grad();
        var net = new PixelShuffleUpsampleNetwork(inChannels: 8, outChannels: 3, scaleFactor: 2, convLayers: 3);
        var got = net.call(randn([1, 3, 8, 5, 5]));
        Assert.Equal(new long[] { 1, 3, 3, 10, 10 }, got.shape);
        Assert.Equal(new[] { "upsample.0.bias", "upsample.0.weight", "upsample.2.bias", "upsample.2.weight", "upsample.4.bias", "upsample.4.weight" }, Names(net));
        // the planes are independent of each other
        var one = net.call(randn([1, 3, 8, 5, 5]).narrow(1, 1, 1));
        Assert.Equal(new long[] { 1, 1, 3, 10, 10 }, one.shape);
    }

    // ---- the real checkpoint ----------------------------------------------------------------------------------------------------------------

    [RequiresCheckpoint]
    public void TheModulesHaveExactlyTheKeysAndShapesOfTheCheckpoint()
    {
        foreach (var (prefix, module, count) in new (string, nn.Module, int)[]
        {
            ("backbone.", new TwoStreamInterleaveTransformer(new BackboneConfig()), 359),
            ("tokenizer.", new TriplaneTokenizer(), 1),
            ("post_processor.", new PixelShuffleUpsampleNetwork(), 8),
        })
        {
            var expected = Checkpoint.Weights.Index.Tensors.Where(t => t.Key.StartsWith(prefix, StringComparison.Ordinal))
                .Select(t => $"{t.Key[prefix.Length..]} {string.Join("x", t.Value.Shape)}").Order(StringComparer.Ordinal).ToArray();
            Assert.Equal(count, expected.Length);
            Assert.Equal(expected, Shapes(module));
        }
    }

    private sealed record Stage(string Name, Parity.Result Result)
    {
        public bool Passes => Result.CosSim >= 0.9999 && Result.MaxAbs <= 1e-4;
    }

    [RequiresCheckpointAndPictureFixtures]
    public void EachStageMatchesTheReferenceOnTheFivePictures()
    {
        var failures = new List<string>();
        var (triplaneModel, backboneModel, postModel) = (RealStages.Triplane, RealStages.Backbone, RealStages.PostProcessor);   // loaded outside the scope below, or it would free them
        foreach (var name in Fixtures.Pictures)
        {
            using var _ = no_grad();
            using var scope = NewDisposeScope();   // the intermediate tensors of the backbone are hundreds of MB: free them per picture, not whenever the GC gets to it
            // each stage takes the reference's own input, so a difference is found at the stage where it starts
            using var triplaneRef = Fixtures.Read(Fixtures.Big(name), "triplane_tokens");           // (1, 1024, 27648)
            using var imageRef = Fixtures.Read(Fixtures.Big(name), "image_tokens");                 // (1, 1, 1024, 1297)
            using var backboneRef = Fixtures.Read(Fixtures.Big(name), "backbone_out");
            using var directRef = Fixtures.Read(Fixtures.Big(name), "direct_codes");                // (1, 3, 1024, 96, 96)
            using var sceneRef = Fixtures.Read(Fixtures.Big(name), "scene_codes");                  // (1, 3, 40, 384, 384)

            var stages = new List<Stage>();
            using var tokens = triplaneModel.Tokens(1);
            stages.Add(new Stage("triplane tokens", Parity.Compare(tokens, triplaneRef)));

            using var imageTokens = imageRef.reshape(1, 1024, 1297).permute(0, 2, 1).contiguous();   // "B Nv C Nt -> B (Nv Nt) C"
            using var backbone = backboneModel.call(triplaneRef, imageTokens);
            stages.Add(new Stage("backbone", Parity.Compare(backbone, backboneRef)));

            using var direct = triplaneModel.Detokenize(backboneRef);
            stages.Add(new Stage("direct codes", Parity.Compare(direct, directRef)));

            using var scene = postModel.call(directRef);
            stages.Add(new Stage("scene codes", Parity.Compare(scene, sceneRef)));

            foreach (var s in stages.Where(s => !s.Passes)) failures.Add($"{name} {s.Name}: {s.Result}");
            Console.WriteLine($"{name}: " + string.Join("; ", stages.Select(s => $"{s.Name} {s.Result}")));
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}

/// <summary>The real triplane stage, loaded once from the checkpoint (1.7 GB: the tests that need it share it).</summary>
internal static class RealStages
{
    private static T Load<T>(T module, string prefix) where T : nn.Module
    {
        StrictLoader.Load(module, Checkpoint.Weights.Index, CPU, new StrictLoadOptions($"The checkpoint does not match {prefix.TrimEnd('.')}:") { Prefix = prefix, IncludeBuffers = true });
        module.eval();
        return module;
    }

    private static readonly Lazy<TwoStreamInterleaveTransformer> Backbone_ = new(() => Load(new TwoStreamInterleaveTransformer(new BackboneConfig()), "backbone."));
    private static readonly Lazy<TriplaneTokenizer> Triplane_ = new(() => Load(new TriplaneTokenizer(), "tokenizer."));
    private static readonly Lazy<PixelShuffleUpsampleNetwork> PostProcessor_ = new(() => Load(new PixelShuffleUpsampleNetwork(), "post_processor."));

    private static readonly Lazy<MaterialMlp> Decoder_ = new(() => Load(new MaterialMlp(), "decoder."));

    // the estimator is loaded without the CLIP text tower, which the checkpoint holds and the port never reads (the keys are the estimator's own, without its prefix)
    private static readonly Lazy<ClipHeadEstimator> Estimator_ = new(() =>
    {
        var model = new ClipHeadEstimator();
        StrictLoader.Load(model, Checkpoint.Weights.Index, CPU, new StrictLoadOptions("The checkpoint does not match the CLIP estimator:")
            { Prefix = "image_estimator.", Skip = rest => Sf3dCheckpoint.IsClipTextTower("image_estimator." + rest) });
        model.eval();
        return model;
    });

    public static MaterialMlp Decoder => Decoder_.Value;
    public static ClipHeadEstimator Estimator => Estimator_.Value;
    public static TwoStreamInterleaveTransformer Backbone => Backbone_.Value;
    public static TriplaneTokenizer Triplane => Triplane_.Value;
    public static PixelShuffleUpsampleNetwork PostProcessor => PostProcessor_.Value;
}
