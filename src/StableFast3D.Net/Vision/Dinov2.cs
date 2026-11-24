using StableFast3D.Transformer;
using StableFast3D.Weights;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Vision;

// DINOv2-large as Stable Fast 3D carries it (sf3d/models/tokenizers/dinov2.py, a copy of Hugging Face's with the per-layer modulation added). Every field is
// named as the key of the checkpoint (image_tokenizer.model.embeddings.cls_token, ...encoder.layer.0.norm1_modulation.linear2.weight, ...), so loading needs
// no mapping. Inference only: no dropout, no drop path, no head mask, no hidden states or attentions returned.

/// <summary>The convolution that cuts the picture into patches and projects each one.</summary>
public sealed class Dinov2PatchEmbeddings : nn.Module<Tensor, Tensor>
{
    public readonly nn.Module<Tensor, Tensor> projection;

    public Dinov2PatchEmbeddings(Dinov2Config c) : base(nameof(Dinov2PatchEmbeddings))
    {
        projection = nn.Conv2d(c.NumChannels, c.HiddenSize, kernel_size: c.PatchSize, stride: c.PatchSize);
        RegisterComponents();
    }

    /// <param name="pixelValues">(B, 3, H, W)</param>
    /// <returns>(B, patches, hidden)</returns>
    public override Tensor forward(Tensor pixelValues) => projection.call(pixelValues).flatten(2).transpose(1, 2);
}

/// <summary>The class token, the patch embeddings and the position embeddings, interpolated to the size of the picture.</summary>
public sealed class Dinov2Embeddings : nn.Module<Tensor, Tensor>
{
    private readonly Dinov2Config _config;

    public readonly Parameter cls_token;
    public readonly Dinov2PatchEmbeddings patch_embeddings;
    public readonly Parameter position_embeddings;

    public Dinov2Embeddings(Dinov2Config c) : base(nameof(Dinov2Embeddings))
    {
        _config = c;
        cls_token = nn.Parameter(zeros(1, 1, c.HiddenSize));
        patch_embeddings = new Dinov2PatchEmbeddings(c);
        position_embeddings = nn.Parameter(zeros(1, c.NumPositions, c.HiddenSize));
        RegisterComponents();
        // a buffer of the checkpoint (never used at inference: it is for masked training)
        register_buffer("mask_token", zeros(1, c.HiddenSize));
    }

    public override Tensor forward(Tensor pixelValues)
    {
        var (batch, height, width) = (pixelValues.shape[0], (int)pixelValues.shape[2], (int)pixelValues.shape[3]);
        var patches = patch_embeddings.call(pixelValues);
        var embeddings = cat([cls_token.expand(batch, -1, -1), patches], 1);
        return embeddings + InterpolatePositionEncoding(embeddings, height, width);
    }

    /// <summary>
    /// The position embeddings for a picture of another size than the one they were trained at (37 x 37 patches): the grid is interpolated bicubically with the
    /// scale factor <c>(patches + 0.1) / 37</c> (the 0.1 is the reference's guard against rounding, and it is part of the coordinates the filter uses).
    /// </summary>
    private Tensor InterpolatePositionEncoding(Tensor embeddings, int height, int width)
    {
        var numPatches = embeddings.shape[1] - 1;
        var numPositions = position_embeddings.shape[1] - 1;
        if (numPatches == numPositions && height == width) return position_embeddings;

        var classPos = position_embeddings[TensorIndex.Colon, 0];
        var patchPos = position_embeddings[TensorIndex.Colon, TensorIndex.Slice(1)];
        var dim = embeddings.shape[^1];
        var h = height / _config.PatchSize + 0.1;
        var w = width / _config.PatchSize + 0.1;
        var grid = (long)Math.Sqrt(numPositions);
        var grid2d = patchPos.reshape(1, grid, grid, dim).permute(0, 3, 1, 2);
        var resized = F.interpolate(grid2d, scale_factor: [h / Math.Sqrt(numPositions), w / Math.Sqrt(numPositions)], mode: InterpolationMode.Bicubic, align_corners: false);
        if ((int)h != resized.shape[^2] || (int)w != resized.shape[^1])
            throw new InvalidOperationException("Width or height does not match with the interpolated position embeddings");
        var flat = resized.permute(0, 2, 3, 1).reshape(1, -1, dim);
        return cat([classPos.unsqueeze(0), flat], 1);
    }
}

public sealed class Dinov2SelfAttention : nn.Module<Tensor, Tensor>
{
    private readonly int _heads;
    private readonly int _headSize;

    public readonly nn.Module<Tensor, Tensor> query;
    public readonly nn.Module<Tensor, Tensor> key;
    public readonly nn.Module<Tensor, Tensor> value;

    public Dinov2SelfAttention(Dinov2Config c) : base(nameof(Dinov2SelfAttention))
    {
        _heads = c.NumAttentionHeads;
        _headSize = c.HeadDim;
        query = nn.Linear(c.HiddenSize, c.HiddenSize);
        key = nn.Linear(c.HiddenSize, c.HiddenSize);
        value = nn.Linear(c.HiddenSize, c.HiddenSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor hiddenStates)
    {
        var (b, n) = (hiddenStates.shape[0], hiddenStates.shape[1]);
        var q = query.call(hiddenStates).reshape(b, n, _heads, _headSize).transpose(1, 2);
        var k = key.call(hiddenStates).reshape(b, n, _heads, _headSize).transpose(1, 2);
        var v = value.call(hiddenStates).reshape(b, n, _heads, _headSize).transpose(1, 2);
        var context = F.scaled_dot_product_attention(q, k, v);
        return context.transpose(1, 2).reshape(b, n, -1);
    }
}

public sealed class Dinov2SelfOutput : nn.Module<Tensor, Tensor>
{
    public readonly nn.Module<Tensor, Tensor> dense;

    public Dinov2SelfOutput(Dinov2Config c) : base(nameof(Dinov2SelfOutput))
    {
        dense = nn.Linear(c.HiddenSize, c.HiddenSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor hiddenStates) => dense.call(hiddenStates);
}

public sealed class Dinov2Attention : nn.Module<Tensor, Tensor>
{
    public readonly Dinov2SelfAttention attention;
    public readonly Dinov2SelfOutput output;

    public Dinov2Attention(Dinov2Config c) : base(nameof(Dinov2Attention))
    {
        attention = new Dinov2SelfAttention(c);
        output = new Dinov2SelfOutput(c);
        RegisterComponents();
    }

    public override Tensor forward(Tensor hiddenStates) => output.call(attention.call(hiddenStates));
}

/// <summary>A learned scale per channel (<c>lambda1</c>).</summary>
public sealed class Dinov2LayerScale : nn.Module<Tensor, Tensor>
{
    public readonly Parameter lambda1;

    public Dinov2LayerScale(Dinov2Config c) : base(nameof(Dinov2LayerScale))
    {
        lambda1 = nn.Parameter(ones(c.HiddenSize));
        RegisterComponents();
    }

    public override Tensor forward(Tensor hiddenState) => hiddenState * lambda1;
}

public sealed class Dinov2Mlp : nn.Module<Tensor, Tensor>
{
    public readonly nn.Module<Tensor, Tensor> fc1;
    public readonly nn.Module<Tensor, Tensor> fc2;

    public Dinov2Mlp(Dinov2Config c) : base(nameof(Dinov2Mlp))
    {
        fc1 = nn.Linear(c.HiddenSize, c.IntermediateSize);
        fc2 = nn.Linear(c.IntermediateSize, c.HiddenSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor hiddenState) => fc2.call(F.gelu(fc1.call(hiddenState)));
}

/// <summary>One block: pre-norm attention and pre-norm MLP, each scaled per channel and each with the camera modulation applied right after its norm.</summary>
public sealed class Dinov2Layer : nn.Module<Tensor, Tensor, Tensor>
{
    public readonly nn.Module<Tensor, Tensor> norm1;
    public readonly Modulation norm1_modulation;
    public readonly Dinov2Attention attention;
    public readonly Dinov2LayerScale layer_scale1;
    public readonly nn.Module<Tensor, Tensor> norm2;
    public readonly Modulation norm2_modulation;
    public readonly Dinov2Mlp mlp;
    public readonly Dinov2LayerScale layer_scale2;

    public Dinov2Layer(Dinov2Config c, int modulationDim) : base(nameof(Dinov2Layer))
    {
        norm1 = nn.LayerNorm(c.HiddenSize, c.LayerNormEps);
        norm1_modulation = new Modulation(c.HiddenSize, modulationDim);
        attention = new Dinov2Attention(c);
        layer_scale1 = new Dinov2LayerScale(c);
        norm2 = nn.LayerNorm(c.HiddenSize, c.LayerNormEps);
        norm2_modulation = new Modulation(c.HiddenSize, modulationDim);
        mlp = new Dinov2Mlp(c);
        layer_scale2 = new Dinov2LayerScale(c);
        RegisterComponents();
    }

    /// <param name="hiddenStates">(B, N, C)</param>
    /// <param name="modulationCond">(B, Cc)</param>
    public override Tensor forward(Tensor hiddenStates, Tensor modulationCond)
    {
        var attn = layer_scale1.call(attention.call(norm1_modulation.call(norm1.call(hiddenStates), modulationCond)));
        hiddenStates = attn + hiddenStates;
        var mlpOut = layer_scale2.call(mlp.call(norm2_modulation.call(norm2.call(hiddenStates), modulationCond)));
        return mlpOut + hiddenStates;
    }
}

public sealed class Dinov2Encoder : nn.Module<Tensor, Tensor, Tensor>
{
    public readonly ModuleList<Dinov2Layer> layer;

    /// <summary>Called with the index and the output of every layer, when set (to find where a drift from the reference starts).</summary>
    public Action<int, Tensor>? LayerHook { get; set; }

    public Dinov2Encoder(Dinov2Config c, int modulationDim) : base(nameof(Dinov2Encoder))
    {
        layer = nn.ModuleList(Enumerable.Range(0, c.NumHiddenLayers).Select(_ => new Dinov2Layer(c, modulationDim)).ToArray());
        RegisterComponents();
    }

    public override Tensor forward(Tensor hiddenStates, Tensor modulationCond)
    {
        for (var i = 0; i < layer.Count; i++)
        {
            hiddenStates = layer[i].call(hiddenStates, modulationCond);
            LayerHook?.Invoke(i, hiddenStates);
        }
        return hiddenStates;
    }
}

/// <summary><c>Dinov2Model</c> with the modulation: embeddings, the 24 blocks and the final layer norm; the output is every token (the class token first).</summary>
public sealed class Dinov2Model : nn.Module<Tensor, Tensor, Tensor>
{
    public readonly Dinov2Embeddings embeddings;
    public readonly Dinov2Encoder encoder;
    public readonly nn.Module<Tensor, Tensor> layernorm;

    public Dinov2Model(Dinov2Config c, int modulationDim = 768) : base(nameof(Dinov2Model))
    {
        embeddings = new Dinov2Embeddings(c);
        encoder = new Dinov2Encoder(c, modulationDim);
        layernorm = nn.LayerNorm(c.HiddenSize, c.LayerNormEps);
        RegisterComponents();
    }

    /// <param name="pixelValues">(B, 3, H, W), already normalised</param>
    /// <param name="modulationCond">(B, 768): the camera embedding of each picture</param>
    /// <returns>(B, 1 + patches, hidden)</returns>
    public override Tensor forward(Tensor pixelValues, Tensor modulationCond) => layernorm.call(encoder.call(embeddings.call(pixelValues), modulationCond));
}

/// <summary>
/// <c>DINOV2SingleImageTokenizer</c>: the pictures are normalised with ImageNet's mean and standard deviation, go through DINOv2 modulated by their camera, and
/// the tokens come out channel first. The weights are <c>image_tokenizer.model.*</c> of the checkpoint.
/// </summary>
public sealed class Dinov2ImageTokenizer : nn.Module<Tensor, Tensor, Tensor>
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    public readonly Dinov2Model model;

    public Dinov2ImageTokenizer(Dinov2Config c, int modulationDim = 768) : base(nameof(Dinov2ImageTokenizer))
    {
        model = new Dinov2Model(c, modulationDim);
        RegisterComponents();
    }

    /// <param name="images">(B, Nv, 3, H, W), 0..1</param>
    /// <param name="modulationCond">(B, Nv, 768)</param>
    /// <returns>(B, Nv, hidden, tokens)</returns>
    public override Tensor forward(Tensor images, Tensor modulationCond)
    {
        var (b, nv) = (images.shape[0], images.shape[1]);
        using var mean = tensor(Mean, [1, 1, 3, 1, 1], device: images.device);
        using var std = tensor(Std, [1, 1, 3, 1, 1], device: images.device);
        var normalised = (images - mean) / std;
        var tokens = model.call(normalised.reshape(b * nv, images.shape[2], images.shape[3], images.shape[4]), modulationCond.reshape(b * nv, -1));
        return tokens.permute(0, 2, 1).reshape(b, nv, tokens.shape[2], tokens.shape[1]);
    }
}
