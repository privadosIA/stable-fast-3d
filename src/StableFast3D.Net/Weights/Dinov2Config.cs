using System.Text.Json;

namespace StableFast3D.Weights;

/// <summary>
/// The configuration of the DINOv2 tower of Stable Fast 3D (<c>sf3d/dinov2-large-config.json</c>: the Hugging Face config of <c>facebook/dinov2-large</c>, which
/// the Python container needed a patch to read from disk instead of downloading). The defaults are that file's values; only what the inference path uses is
/// kept, and what the port does not implement (SwiGLU, another activation, no query/key/value bias) is rejected when a file is read.
/// </summary>
public sealed record Dinov2Config
{
    public int HiddenSize { get; init; } = 1024;
    public int NumHiddenLayers { get; init; } = 24;
    public int NumAttentionHeads { get; init; } = 16;
    public int MlpRatio { get; init; } = 4;
    public int PatchSize { get; init; } = 14;
    /// <summary>The size the position embedding was trained at (518 = 37 x 14); other sizes interpolate it.</summary>
    public int ImageSize { get; init; } = 518;
    public int NumChannels { get; init; } = 3;
    public double LayerNormEps { get; init; } = 1e-6;

    public int HeadDim => HiddenSize / NumAttentionHeads;
    public int IntermediateSize => HiddenSize * MlpRatio;

    /// <summary>Patches a side at the trained size (37): the position embedding is a 37 x 37 grid plus the class token.</summary>
    public int PositionGrid => ImageSize / PatchSize;

    /// <summary>Rows of the position embedding in the checkpoint (1 + 37 x 37 = 1370).</summary>
    public int NumPositions => 1 + PositionGrid * PositionGrid;

    /// <summary>Tokens for a picture of <paramref name="size"/> pixels a side: the class token and the patches (512 gives 1 + 36 x 36 = 1297).</summary>
    public int TokensFor(int size) => 1 + (size / PatchSize) * (size / PatchSize);

    public static Dinov2Config FromFile(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var r = doc.RootElement;
        if (r.ValueKind != JsonValueKind.Object) throw new InvalidDataException($"{path} is not a JSON object");

        if (Str(r, "model_type", "dinov2") != "dinov2")
            throw new NotSupportedException($"model_type must be dinov2, got {Str(r, "model_type", "")}");
        if (Str(r, "hidden_act", "gelu") != "gelu")
            throw new NotSupportedException("only hidden_act=gelu is supported");
        if (Bool(r, "use_swiglu_ffn", false))
            throw new NotSupportedException("use_swiglu_ffn is true: only the plain MLP of dinov2-large is supported");
        if (!Bool(r, "qkv_bias", true))
            throw new NotSupportedException("qkv_bias is false: the checkpoint has query, key and value biases");

        var d = new Dinov2Config();
        var c = new Dinov2Config
        {
            HiddenSize = Int(r, "hidden_size", d.HiddenSize),
            NumHiddenLayers = Int(r, "num_hidden_layers", d.NumHiddenLayers),
            NumAttentionHeads = Int(r, "num_attention_heads", d.NumAttentionHeads),
            MlpRatio = Int(r, "mlp_ratio", d.MlpRatio),
            PatchSize = Int(r, "patch_size", d.PatchSize),
            ImageSize = Int(r, "image_size", d.ImageSize),
            NumChannels = Int(r, "num_channels", d.NumChannels),
            LayerNormEps = Dbl(r, "layer_norm_eps", d.LayerNormEps),
        };
        if (c.HiddenSize % c.NumAttentionHeads != 0)
            throw new InvalidDataException($"hidden_size {c.HiddenSize} is not a multiple of num_attention_heads {c.NumAttentionHeads}");
        return c;
    }

    private static string Str(JsonElement e, string name, string fallback) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString()! : fallback;
    private static bool Bool(JsonElement e, string name, bool fallback) => e.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : fallback;
    private static int Int(JsonElement e, string name, int fallback) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
    private static double Dbl(JsonElement e, string name, double fallback) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : fallback;
}
