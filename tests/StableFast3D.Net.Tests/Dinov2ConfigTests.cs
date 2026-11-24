using StableFast3D.Weights;

namespace StableFast3D.Net.Tests;

public class Dinov2ConfigTests
{
    private static string Write(TempDir dir, string json)
    {
        var path = dir.File("dinov2-large-config.json");
        File.WriteAllText(path, json);
        return path;
    }

    [Fact]
    public void DefaultsDescribeDinov2Large()
    {
        var c = new Dinov2Config();
        Assert.Equal(64, c.HeadDim);
        Assert.Equal(4096, c.IntermediateSize);
        Assert.Equal(37, c.PositionGrid);          // 518 / 14
        Assert.Equal(1370, c.NumPositions);        // the rows of the position embedding in the checkpoint
        Assert.Equal(1297, c.TokensFor(512));      // the class token and 36 x 36 patches: the position embedding is interpolated
    }

    [Theory]
    [InlineData("{\"model_type\": \"vit\"}", "dinov2")]
    [InlineData("{\"hidden_act\": \"silu\"}", "gelu")]
    [InlineData("{\"use_swiglu_ffn\": true}", "swiglu")]
    [InlineData("{\"qkv_bias\": false}", "qkv_bias")]
    public void UnsupportedConfigurationsAreRejected(string json, string expectedInMessage)
    {
        using var dir = new TempDir();
        var ex = Assert.Throws<NotSupportedException>(() => Dinov2Config.FromFile(Write(dir, json)));
        Assert.Contains(expectedInMessage, ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AHiddenSizeThatTheHeadsDoNotDivideIsRejected()
    {
        using var dir = new TempDir();
        Assert.Throws<InvalidDataException>(() => Dinov2Config.FromFile(Write(dir, "{\"hidden_size\": 1000, \"num_attention_heads\": 16}")));
    }

    [Fact]
    public void MissingFieldsKeepTheDefaults() { using var dir = new TempDir(); Assert.Equal(new Dinov2Config(), Dinov2Config.FromFile(Write(dir, "{}"))); }

    [RequiresCheckpoint]
    public void DefaultsEqualTheRealConfigFile()
    {
        var real = Dinov2Config.FromFile(Path.Combine(Checkpoint.Dir!, Sf3dCheckpoint.Dinov2ConfigFile));
        Assert.Equal(new Dinov2Config(), real);
        Assert.Equal(1e-6, real.LayerNormEps);
    }
}
