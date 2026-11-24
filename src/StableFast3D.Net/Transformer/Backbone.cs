using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Transformer;

// The two-stream interleave transformer of sf3d/models/transformers/backbone.py: a stream of latents (the image tokens and 1792 learned ones) and the triplane
// stream, each fused into the other with cross attention around a few basic blocks that cross-attend to the raw image tokens. Fields carry the checkpoint's
// key names (backbone.main_blocks.0.fuse_block_in.attn.wq.weight, ...ff.net.0.proj.weight, ...), so loading needs no mapping. Inference only: no dropout.

/// <summary>A gated GELU: one linear layer to twice the width, one half gating the other.</summary>
public sealed class GeGlu : nn.Module<Tensor, Tensor>
{
    public readonly nn.Module<Tensor, Tensor> proj;

    public GeGlu(int dimIn, int dimOut) : base(nameof(GeGlu))
    {
        proj = nn.Linear(dimIn, dimOut * 2);
        RegisterComponents();
    }

    public override Tensor forward(Tensor hiddenStates)
    {
        var halves = proj.call(hiddenStates).chunk(2, dim: -1);
        return halves[0] * F.gelu(halves[1]);
    }
}

/// <summary>Multi-head attention of <c>x_q</c> over <c>x_kv</c> (the same tensor for self attention), with no bias on the query, key and value projections.</summary>
public sealed class CrossAttention : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly int _heads;

    public readonly nn.Module<Tensor, Tensor> wq;
    public readonly nn.Module<Tensor, Tensor> wk;
    public readonly nn.Module<Tensor, Tensor> wv;
    public readonly nn.Module<Tensor, Tensor> proj;

    public CrossAttention(int dim, int kvDim, int numHeads) : base(nameof(CrossAttention))
    {
        _heads = numHeads;
        wq = nn.Linear(dim, dim, hasBias: false);
        wk = nn.Linear(kvDim, dim, hasBias: false);
        wv = nn.Linear(kvDim, dim, hasBias: false);
        proj = nn.Linear(dim, dim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor xQ, Tensor xKv)
    {
        var (b, nq, c) = (xQ.shape[0], xQ.shape[1], xQ.shape[2]);
        var nkv = xKv.shape[1];
        var hd = c / _heads;
        var q = wq.call(xQ).reshape(b, nq, _heads, hd).permute(0, 2, 1, 3);
        var k = wk.call(xKv).reshape(b, nkv, _heads, hd).permute(0, 2, 1, 3);
        var v = wv.call(xKv).reshape(b, nkv, _heads, hd).permute(0, 2, 1, 3);
        // upstream passes scale = head_dim ** -0.5, which is what SDPA uses by default (TorchSharp has no scale argument)
        var x = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3);
        return proj.call(x.reshape(b, nq, c));
    }
}

/// <summary>GEGLU, a (no-op at inference) dropout and a linear layer back: <c>net.0.proj</c> and <c>net.2</c> in the checkpoint.</summary>
public sealed class FeedForward : nn.Module<Tensor, Tensor>
{
    public readonly ModuleList<nn.Module<Tensor, Tensor>> net;

    public FeedForward(int dim, int mult = 4) : base(nameof(FeedForward))
    {
        var inner = dim * mult;
        net = nn.ModuleList<nn.Module<Tensor, Tensor>>(new GeGlu(dim, inner), nn.Identity(), nn.Linear(inner, dim));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        foreach (var m in net) x = m.call(x);
        return x;
    }
}

/// <summary>Self attention, attention over <c>x</c> (or over itself when there is none) and a feed-forward, each with its pre-norm and its residual.</summary>
public sealed class BasicBlock : nn.Module<Tensor, Tensor, Tensor>
{
    public readonly nn.Module<Tensor, Tensor> norm1;
    public readonly CrossAttention attn1;
    public readonly nn.Module<Tensor, Tensor> norm2;
    public readonly CrossAttention attn2;
    public readonly nn.Module<Tensor, Tensor> norm3;
    public readonly FeedForward ff;

    public BasicBlock(int dim, int kvDim, int numHeads) : base(nameof(BasicBlock))
    {
        norm1 = nn.LayerNorm(dim);
        attn1 = new CrossAttention(dim, dim, numHeads);
        norm2 = nn.LayerNorm(dim);
        attn2 = new CrossAttention(dim, kvDim, numHeads);
        norm3 = nn.LayerNorm(dim);
        ff = new FeedForward(dim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor z, Tensor x)
    {
        var zNorm = norm1.call(z);
        z = z + attn1.call(zNorm, zNorm);
        zNorm = norm2.call(z);
        z = z + attn2.call(zNorm, x);
        zNorm = norm3.call(z);
        return z + ff.call(zNorm);
    }
}

/// <summary>Fuses <c>x</c> into <c>z</c> with cross attention and a feed-forward (the reference's <c>norm_x_input</c> is off in this checkpoint: no <c>norm_x</c>).</summary>
public sealed class FuseBlock : nn.Module<Tensor, Tensor, Tensor>
{
    public readonly CrossAttention attn;
    public readonly nn.Module<Tensor, Tensor> norm_z1;
    public readonly nn.Module<Tensor, Tensor> norm_z2;
    public readonly FeedForward ff;

    public FuseBlock(int dimZ, int dimX, int numHeads) : base(nameof(FuseBlock))
    {
        attn = new CrossAttention(dimZ, dimX, numHeads);
        norm_z1 = nn.LayerNorm(dimZ);
        norm_z2 = nn.LayerNorm(dimZ);
        ff = new FeedForward(dimZ);
        RegisterComponents();
    }

    public override Tensor forward(Tensor z, Tensor x)
    {
        z = z + attn.call(norm_z1.call(z), x);
        return z + ff.call(norm_z2.call(z));
    }
}

public sealed class TwoStreamBlock : nn.Module<Tensor, Tensor, Tensor, (Tensor Latent, Tensor Input)>
{
    public readonly FuseBlock fuse_block_in;
    public readonly ModuleList<BasicBlock> transformer_block;
    public readonly FuseBlock fuse_block_out;

    public TwoStreamBlock(int dimLatent, int dimInput, int numBasicBlocks, int numHeads, int dimCross) : base(nameof(TwoStreamBlock))
    {
        fuse_block_in = new FuseBlock(dimLatent, dimInput, numHeads);
        transformer_block = nn.ModuleList(Enumerable.Range(0, numBasicBlocks).Select(_ => new BasicBlock(dimLatent, dimCross, numHeads)).ToArray());
        fuse_block_out = new FuseBlock(dimInput, dimLatent, numHeads);
        RegisterComponents();
    }

    public override (Tensor Latent, Tensor Input) forward(Tensor latent, Tensor input, Tensor crossInput)
    {
        latent = fuse_block_in.call(latent, input);
        foreach (var block in transformer_block) latent = block.call(latent, crossInput);
        input = fuse_block_out.call(input, latent);
        return (latent, input);
    }
}

/// <summary>The sizes of the backbone (<c>backbone:</c> of <c>sf3d/config.yaml</c>; the rest of upstream's config is fixed by the checkpoint).</summary>
public sealed record BackboneConfig
{
    public int NumAttentionHeads { get; init; } = 16;
    public int AttentionHeadDim { get; init; } = 64;
    public int RawTriplaneChannels { get; init; } = 1024;
    public int TriplaneChannels { get; init; } = 1024;
    public int RawImageChannels { get; init; } = 1024;
    public int NumLatents { get; init; } = 1792;
    public int NumBlocks { get; init; } = 4;
    public int NumBasicBlocks { get; init; } = 3;
    public int NormNumGroups { get; init; } = 32;
    public int CrossAttentionDim { get; init; } = 1024;

    public int LatentDim => NumAttentionHeads * AttentionHeadDim;
}

/// <summary><c>TwoStreamInterleaveTransformer</c>: (triplane tokens, image tokens) to the new triplane tokens. The weights are <c>backbone.*</c> of the checkpoint.</summary>
public sealed class TwoStreamInterleaveTransformer : nn.Module<Tensor, Tensor, Tensor>
{
    public readonly GroupNorm norm_triplane;
    public readonly nn.Module<Tensor, Tensor> proj_triplane;
    public readonly nn.Module<Tensor, Tensor> norm_image;
    public readonly nn.Module<Tensor, Tensor> proj_image;
    public readonly nn.Module<Tensor, Tensor> norm_latent;
    public readonly nn.Module<Tensor, Tensor> proj_latent;
    public readonly Parameter latent_init;
    public readonly ModuleList<TwoStreamBlock> main_blocks;
    public readonly nn.Module<Tensor, Tensor> proj_out;

    public TwoStreamInterleaveTransformer(BackboneConfig c) : base(nameof(TwoStreamInterleaveTransformer))
    {
        norm_triplane = nn.GroupNorm(c.NormNumGroups, c.RawTriplaneChannels, eps: 1e-6);
        // TorchSharp leaves a GroupNorm's weight and bias uninitialised (garbage, even NaN) until something loads them: start from the identity
        using (no_grad())
        {
            norm_triplane.weight!.fill_(1);
            norm_triplane.bias!.fill_(0);
        }
        proj_triplane = nn.Linear(c.RawTriplaneChannels, c.TriplaneChannels);
        norm_image = nn.LayerNorm(c.RawImageChannels);
        proj_image = nn.Linear(c.RawImageChannels, c.LatentDim);
        norm_latent = nn.LayerNorm(c.LatentDim);
        proj_latent = nn.Linear(c.LatentDim, c.LatentDim);
        latent_init = nn.Parameter(zeros(1, c.NumLatents, c.LatentDim));
        main_blocks = nn.ModuleList(Enumerable.Range(0, c.NumBlocks)
            .Select(_ => new TwoStreamBlock(c.LatentDim, c.TriplaneChannels, c.NumBasicBlocks, c.NumAttentionHeads, c.CrossAttentionDim)).ToArray());
        proj_out = nn.Linear(c.TriplaneChannels, c.RawTriplaneChannels);
        RegisterComponents();
    }

    /// <param name="hiddenStates">the triplane tokens, (B, Ct, Nt)</param>
    /// <param name="encoderHiddenStates">the image tokens, (B, N_image, Cit)</param>
    /// <returns>(B, Ct, Nt)</returns>
    public override Tensor forward(Tensor hiddenStates, Tensor encoderHiddenStates)
    {
        var triplane = proj_triplane.call(norm_triplane.call(hiddenStates).permute(0, 2, 1));
        var image = proj_image.call(norm_image.call(encoderHiddenStates));
        var initLatents = proj_latent.call(norm_latent.call(latent_init.expand(hiddenStates.shape[0], -1, -1)));
        var latent = cat([image, initLatents], 1);
        foreach (var block in main_blocks) (latent, triplane) = block.call(latent, triplane, encoderHiddenStates);
        return proj_out.call(triplane).permute(0, 2, 1).contiguous() + hiddenStates;
    }
}
