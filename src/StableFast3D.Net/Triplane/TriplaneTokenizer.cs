using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StableFast3D.Triplane;

/// <summary>
/// <c>TriplaneLearnablePositionalEmbedding</c>: three learned planes (<c>tokenizer.embeddings</c>, 3 x 1024 x 96 x 96) laid out as tokens, and back.
/// Tokens are (B, channels, 3 x 96 x 96): the planes one after the other, each row by row.
/// </summary>
public sealed class TriplaneTokenizer : nn.Module
{
    public const int Planes = 3;

    private readonly int _planeSize;
    private readonly int _channels;

    public readonly Parameter embeddings;

    public TriplaneTokenizer(int planeSize = 96, int numChannels = 1024) : base(nameof(TriplaneTokenizer))
    {
        _planeSize = planeSize;
        _channels = numChannels;
        embeddings = nn.Parameter(zeros(Planes, numChannels, planeSize, planeSize));
        RegisterComponents();
    }

    /// <summary>The tokens of <paramref name="batchSize"/> pictures: (B, channels, 3 x plane x plane).</summary>
    public Tensor Tokens(long batchSize) =>
        embeddings.permute(1, 0, 2, 3).reshape(_channels, Planes * _planeSize * _planeSize).unsqueeze(0).expand(batchSize, -1, -1);

    /// <summary>Tokens (B, channels, 3 x plane x plane) back to the three planes: (B, 3, channels, plane, plane).</summary>
    public Tensor Detokenize(Tensor tokens)
    {
        var (b, c, n) = (tokens.shape[0], tokens.shape[1], tokens.shape[2]);
        if (n != Planes * _planeSize * _planeSize) throw new ArgumentException($"{n} tokens are not three planes of {_planeSize} x {_planeSize}");
        if (c != _channels) throw new ArgumentException($"{c} channels, the planes have {_channels}");
        return tokens.reshape(b, c, Planes, _planeSize, _planeSize).permute(0, 2, 1, 3, 4);
    }
}
