using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Transformer;

/// <summary>
/// <c>sf3d.models.transformers.attention.Modulation</c> in the single-layer form the image tokenizer uses (<c>single_layer=True</c>: no <c>linear1</c>):
/// the condition goes through SiLU and one linear layer (<c>linear2</c>, to twice the width of the features), which is split into a scale and a shift,
/// and the features become <c>x * (1 + scale) + shift</c>, the same for every token.
/// </summary>
public sealed class Modulation : nn.Module<Tensor, Tensor, Tensor>
{
    public readonly nn.Module<Tensor, Tensor> linear2;

    public Modulation(int embeddingDim, int conditionDim) : base(nameof(Modulation))
    {
        linear2 = nn.Linear(conditionDim, embeddingDim * 2);
        RegisterComponents();
    }

    /// <param name="x">(B, N, C)</param>
    /// <param name="condition">(B, Cc)</param>
    public override Tensor forward(Tensor x, Tensor condition)
    {
        var emb = linear2.call(F.silu(condition));
        var chunks = emb.chunk(2, dim: 1);
        return x * (1 + chunks[0].unsqueeze(1)) + chunks[1].unsqueeze(1);
    }
}
