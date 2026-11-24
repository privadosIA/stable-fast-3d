using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Vision;

// The vision tower of open_clip's ViT-B/32 (laion2b_s34b_b79k), as the checkpoint of Stable Fast 3D holds it under image_estimator.model.visual.*: the class
// token and the 49 patch tokens of a 224 x 224 picture through 12 pre-norm blocks, the class token's final norm and a projection to 512. Fields carry the
// checkpoint's key names. The text tower of the same CLIP model is in the checkpoint too and is never read (Sf3dCheckpoint.IsClipTextTower).

/// <summary>Multi-head self attention with one packed projection (<c>torch.nn.MultiheadAttention</c>'s <c>in_proj_*</c> and <c>out_proj</c>).</summary>
public sealed class ClipAttention : nn.Module<Tensor, Tensor>
{
    private readonly int _heads;

    public readonly Parameter in_proj_weight;
    public readonly Parameter in_proj_bias;
    public readonly nn.Module<Tensor, Tensor> out_proj;

    public ClipAttention(int width, int heads) : base(nameof(ClipAttention))
    {
        _heads = heads;
        in_proj_weight = nn.Parameter(zeros(width * 3, width));
        in_proj_bias = nn.Parameter(zeros(width * 3));
        out_proj = nn.Linear(width, width);
        RegisterComponents();
    }

    /// <param name="x">(B, N, width)</param>
    public override Tensor forward(Tensor x)
    {
        var (b, n, w) = (x.shape[0], x.shape[1], x.shape[2]);
        var qkv = F.linear(x, in_proj_weight, in_proj_bias).chunk(3, dim: -1);
        var hd = w / _heads;
        Tensor Heads(Tensor t) => t.reshape(b, n, _heads, hd).permute(0, 2, 1, 3);
        var context = F.scaled_dot_product_attention(Heads(qkv[0]), Heads(qkv[1]), Heads(qkv[2])).permute(0, 2, 1, 3).reshape(b, n, w);
        return out_proj.call(context);
    }
}

public sealed class ClipMlp : nn.Module<Tensor, Tensor>
{
    public readonly nn.Module<Tensor, Tensor> c_fc;
    public readonly nn.Module<Tensor, Tensor> c_proj;

    public ClipMlp(int width, int ratio = 4) : base(nameof(ClipMlp))
    {
        c_fc = nn.Linear(width, width * ratio);
        c_proj = nn.Linear(width * ratio, width);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => c_proj.call(F.gelu(c_fc.call(x)));
}

public sealed class ClipResidualBlock : nn.Module<Tensor, Tensor>
{
    public readonly nn.Module<Tensor, Tensor> ln_1;
    public readonly ClipAttention attn;
    public readonly nn.Module<Tensor, Tensor> ln_2;
    public readonly ClipMlp mlp;

    public ClipResidualBlock(int width, int heads) : base(nameof(ClipResidualBlock))
    {
        ln_1 = nn.LayerNorm(width);
        attn = new ClipAttention(width, heads);
        ln_2 = nn.LayerNorm(width);
        mlp = new ClipMlp(width);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = x + attn.call(ln_1.call(x));
        return x + mlp.call(ln_2.call(x));
    }
}

public sealed class ClipTransformer : nn.Module<Tensor, Tensor>
{
    public readonly ModuleList<ClipResidualBlock> resblocks;

    public ClipTransformer(int width, int layers, int heads) : base(nameof(ClipTransformer))
    {
        resblocks = nn.ModuleList(Enumerable.Range(0, layers).Select(_ => new ClipResidualBlock(width, heads)).ToArray());
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        foreach (var block in resblocks) x = block.call(x);
        return x;
    }
}

/// <summary>The sizes of the vision tower (<c>ViT-B-32</c> of open_clip).</summary>
public sealed record ClipVisionConfig
{
    public int ImageSize { get; init; } = 224;
    public int PatchSize { get; init; } = 32;
    public int Width { get; init; } = 768;
    public int Layers { get; init; } = 12;
    public int Heads { get; init; } = 12;
    public int EmbedDim { get; init; } = 512;

    public int Patches => (ImageSize / PatchSize) * (ImageSize / PatchSize);
}

public sealed class ClipVisionTransformer : nn.Module<Tensor, Tensor>
{
    public readonly Parameter class_embedding;
    public readonly Parameter positional_embedding;
    public readonly Parameter proj;
    public readonly nn.Module<Tensor, Tensor> conv1;
    public readonly nn.Module<Tensor, Tensor> ln_pre;
    public readonly ClipTransformer transformer;
    public readonly nn.Module<Tensor, Tensor> ln_post;

    public ClipVisionTransformer(ClipVisionConfig c) : base(nameof(ClipVisionTransformer))
    {
        class_embedding = nn.Parameter(zeros(c.Width));
        positional_embedding = nn.Parameter(zeros(c.Patches + 1, c.Width));
        proj = nn.Parameter(zeros(c.Width, c.EmbedDim));
        conv1 = nn.Conv2d(3, c.Width, kernel_size: c.PatchSize, stride: c.PatchSize, bias: false);
        ln_pre = nn.LayerNorm(c.Width);
        transformer = new ClipTransformer(c.Width, c.Layers, c.Heads);
        ln_post = nn.LayerNorm(c.Width);
        RegisterComponents();
    }

    /// <param name="x">(B, 3, 224, 224), normalised</param>
    /// <returns>(B, 512): the class token's feature, not normalised to unit length</returns>
    public override Tensor forward(Tensor x)
    {
        x = conv1.call(x);                                                    // (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1);           // (B, grid^2, width)
        x = cat([class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x], 1) + positional_embedding.to(x.dtype);
        x = transformer.call(ln_pre.call(x));
        var pooled = ln_post.call(x)[TensorIndex.Colon, 0];                   // the class token
        return pooled.matmul(proj);
    }
}

/// <summary>The CLIP model as the checkpoint names it: only its vision tower is carried (<c>image_estimator.model.visual.*</c>).</summary>
public sealed class ClipModel : nn.Module
{
    public readonly ClipVisionTransformer visual;

    public ClipModel(ClipVisionConfig c) : base(nameof(ClipModel))
    {
        visual = new ClipVisionTransformer(c);
        RegisterComponents();
    }
}
