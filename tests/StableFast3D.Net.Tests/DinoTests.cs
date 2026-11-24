using MusicGen.Core.Testing;
using MusicGen.Core.Weights;
using StableFast3D.Transformer;
using StableFast3D.Vision;
using StableFast3D.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

[Collection(Heavy.Name)]
public class DinoTests
{
    private static readonly Dinov2Config Tiny = new() { HiddenSize = 32, NumHiddenLayers = 2, NumAttentionHeads = 4, ImageSize = 56 };   // 4 x 4 patches trained, 14 pixels each

    private static (string Name, long[] Shape)[] StateOf(nn.Module m) =>
        m.named_parameters().Select(p => (p.name, p.parameter.shape)).Concat(m.named_buffers().Select(b => (b.name, b.buffer.shape))).OrderBy(x => x.Item1, StringComparer.Ordinal).ToArray();

    // ---- small, no weights ------------------------------------------------------------------------------------------------------------------

    [Fact]
    public void ModulationScalesAndShiftsEveryTokenByTheCondition()
    {
        using var _ = no_grad();
        var m = new Modulation(4, 3);
        // the weights: scale = 2 * c0, shift = c1 (per feature), by hand
        var w = zeros(8, 3);
        w[0, 0] = 1; w[1, 0] = 1; w[2, 0] = 1; w[3, 0] = 1;     // the scale rows read the first condition number (after SiLU)
        w[4, 1] = 1; w[5, 1] = 1; w[6, 1] = 1; w[7, 1] = 1;     // the shift rows read the second
        m.linear2.get_parameter("weight")!.copy_(w);
        m.linear2.get_parameter("bias")!.copy_(zeros(8));
        var cond = tensor(new float[] { 0f, 0f, 0f }, [1, 3]);
        var x = ones(1, 5, 4) * 3;
        Assert.True(m.call(x, cond).allclose(x));               // zero condition: SiLU(0) = 0, so scale 0 and shift 0, the identity

        cond = tensor(new float[] { 1f, -1f, 0f }, [1, 3]);
        var silu = (float)(1 / (1 + Math.Exp(-1)));            // SiLU(1) = 1 * sigmoid(1)
        var siluNeg = (float)(-1 / (1 + Math.Exp(1)));          // SiLU(-1)
        var got = m.call(x, cond);
        Assert.Equal(3 * (1 + silu) + siluNeg, got[0, 4, 3].item<float>(), 5);
    }

    [Fact]
    public void TheParametersCarryTheNamesOfTheCheckpoint()
    {
        var names = StateOf(new Dinov2ImageTokenizer(Tiny)).Select(x => x.Name).ToHashSet();
        Assert.Contains("model.embeddings.cls_token", names);
        Assert.Contains("model.embeddings.mask_token", names);
        Assert.Contains("model.embeddings.position_embeddings", names);
        Assert.Contains("model.embeddings.patch_embeddings.projection.weight", names);
        Assert.Contains("model.encoder.layer.1.attention.attention.query.bias", names);
        Assert.Contains("model.encoder.layer.1.attention.output.dense.weight", names);
        Assert.Contains("model.encoder.layer.0.layer_scale2.lambda1", names);
        Assert.Contains("model.encoder.layer.0.mlp.fc1.weight", names);
        Assert.Contains("model.encoder.layer.1.norm2_modulation.linear2.bias", names);
        Assert.Contains("model.layernorm.weight", names);
        // cls, mask, position, patch weight and bias; the final norm's weight and bias; and 22 tensors in each layer (the real one has 24: 5 + 2 + 24 * 22 = 535)
        Assert.Equal(5 + 2 + Tiny.NumHiddenLayers * 22, names.Count);
    }

    [Fact]
    public void ThePositionEmbeddingsAreInterpolatedToThePictureSize()
    {
        using var _ = no_grad();
        var model = new Dinov2ImageTokenizer(Tiny);
        // a picture of 6 x 6 patches where 4 x 4 were trained: 1 + 36 tokens
        var tokens = model.call(rand([2, 1, 3, 84, 84]), zeros(2, 1, 768));
        Assert.Equal(new long[] { 2, 1, 32, 37 }, tokens.shape);
        // and a picture of the trained size takes the embeddings as they are
        Assert.Equal(new long[] { 1, 1, 32, 17 }, model.call(rand([1, 1, 3, 56, 56]), zeros(1, 1, 768)).shape);
    }

    [Fact]
    public void ThePositionEmbeddingsAreAddedAsTheyAreAtTheTrainedSize()
    {
        using var _ = no_grad();
        var emb = new Dinov2Embeddings(Tiny);
        emb.get_parameter("position_embeddings")!.copy_(arange(17 * 32, dtype: ScalarType.Float32).reshape(1, 17, 32));
        emb.get_parameter("cls_token")!.copy_(zeros(1, 1, 32));
        foreach (var (_, p) in emb.named_parameters().Where(p => p.name.StartsWith("patch_embeddings"))) p.copy_(zeros_like(p));
        var got = emb.call(rand(1, 3, 56, 56));
        Assert.True(got.allclose(arange(17 * 32, dtype: ScalarType.Float32).reshape(1, 17, 32)));
    }

    [Fact]
    public void ABatchOfPicturesAreModulatedEachByItsOwnCamera()
    {
        using var _ = no_grad();
        manual_seed(3);
        var model = new Dinov2ImageTokenizer(Tiny);
        foreach (var (name, p) in model.named_parameters().Where(p => p.name.Contains("modulation"))) p.copy_(randn_like(p) * 0.5);
        var images = rand([2, 1, 3, 56, 56]);
        var cams = randn(2, 1, 768);
        var both = model.call(images, cams);
        var first = model.call(images[TensorIndex.Slice(0, 1)], cams[TensorIndex.Slice(0, 1)]);
        Assert.True(both[TensorIndex.Slice(0, 1)].allclose(first, rtol: 1e-4, atol: 1e-5));
        var other = model.call(images[TensorIndex.Slice(0, 1)], cams[TensorIndex.Slice(1, 2)]);
        Assert.False(other.allclose(first, rtol: 1e-4, atol: 1e-5));   // another camera, other tokens
    }

    // ---- the real checkpoint ----------------------------------------------------------------------------------------------------------------

    [RequiresCheckpoint]
    public void TheModulesHaveExactlyTheKeysAndShapesOfTheCheckpoint()
    {
        var config = Checkpoint.Weights.Dinov2;
        var expected = Checkpoint.Weights.Index.Tensors.Where(t => t.Key.StartsWith("image_tokenizer.", StringComparison.Ordinal))
            .Select(t => (Name: t.Key["image_tokenizer.".Length..], Shape: t.Value.Shape)).OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();
        var model = new Dinov2ImageTokenizer(config);
        var actual = StateOf(model);
        Assert.Equal(expected.Select(e => e.Name), actual.Select(a => a.Name));
        Assert.Equal(expected.Select(e => string.Join("x", e.Shape)), actual.Select(a => string.Join("x", a.Shape)));
        Assert.Equal(535, actual.Length);
    }

    /// <summary>
    /// The stage of the spec is the tokens: <c>max_abs &lt;= 1e-4</c>. The outputs of layers 0, 11 and 23 are only there to find where a drift would start; the last one is
    /// the residual stream before the final norm, whose largest values are around 380 (DINOv2's massive activations), so its error is held to a fraction of that
    /// scale (1e-5 of the largest value; fp32 has 1.2e-7) rather than to an absolute 1e-4, which would be a few ulps there.
    /// </summary>
    private sealed record Stage(string Name, Parity.Result Result, double Scale = 0)
    {
        public bool Passes => Result.CosSim >= 0.9999 && Result.MaxAbs <= (Scale > 0 ? 1e-5 * Scale : 1e-4);
    }

    [RequiresCheckpointAndPictureFixtures]
    public void TheTokensMatchTheReferenceOnTheFivePictures()
    {
        var tokenizer = RealModel.Tokenizer;
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            using var _ = no_grad();
            using var scope = NewDisposeScope();
            using var rgb = Fixtures.Read(Fixtures.Big(name), "rgb_cond");                          // (1, 512, 512, 3)
            using var camera = Fixtures.Read(Fixtures.Small(name), "camera_embeds");                // (1, 1, 768)
            var images = rgb.unsqueeze(1).permute(0, 1, 4, 2, 3).contiguous();                       // (B, Nv, C, H, W)
            var layers = new Dictionary<int, Tensor>();
            tokenizer.model.encoder.LayerHook = (i, t) => { if (i is 0 or 11 or 23) layers[i] = t.detach().clone(); };
            using var got = tokenizer.call(images, camera);
            tokenizer.model.encoder.LayerHook = null;

            var stages = new List<Stage>();
            foreach (var i in new[] { 0, 11, 23 })
            {
                using var e = Fixtures.Read(Fixtures.Big(name), $"dino_layer_{i}");
                stages.Add(new Stage($"layer {i}", Parity.Compare(layers[i], e), e.abs().max().item<float>()));
            }
            using var tokens = Fixtures.Read(Fixtures.Big(name), "image_tokens");
            stages.Add(new Stage("tokens", Parity.Compare(got, tokens)));
            foreach (var s in stages.Where(s => !s.Passes))
                failures.Add($"{name} {s.Name}: {s.Result}");
            Console.WriteLine($"{name}: " + string.Join("; ", stages.Select(s => $"{s.Name} {s.Result}")));
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}

/// <summary>The real DINOv2 tokenizer, loaded once from the checkpoint (it is 1.5 GB: the tests that need it share it).</summary>
internal static class RealModel
{
    private static readonly Lazy<Dinov2ImageTokenizer> Tokenizer_ = new(() =>
    {
        var model = new Dinov2ImageTokenizer(Checkpoint.Weights.Dinov2);
        StrictLoader.Load(model, Checkpoint.Weights.Index, CPU, new StrictLoadOptions("The checkpoint does not match the image tokenizer:") { Prefix = "image_tokenizer.", IncludeBuffers = true });
        model.eval();
        return model;
    });

    public static Dinov2ImageTokenizer Tokenizer => Tokenizer_.Value;
}
