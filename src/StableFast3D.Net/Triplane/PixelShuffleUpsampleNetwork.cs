using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Triplane;

/// <summary>
/// <c>PixelShuffleUpsampleNetwork</c> (the post-processor): each plane through four 3 x 3 convolutions (ReLU between them; the last one to
/// <c>out_channels x scale^2</c> channels) and a pixel shuffle, so 1024 channels at 96 x 96 become 40 at 384 x 384. The weights are
/// <c>post_processor.upsample.{0,2,4,6}.{weight,bias}</c> (the sequence's indices, with the ReLUs between them).
/// </summary>
public sealed class PixelShuffleUpsampleNetwork : nn.Module<Tensor, Tensor>
{
    public readonly nn.Module<Tensor, Tensor> upsample;

    public PixelShuffleUpsampleNetwork(int inChannels = 1024, int outChannels = 40, int scaleFactor = 4, int convLayers = 4, int kernelSize = 3) : base(nameof(PixelShuffleUpsampleNetwork))
    {
        var layers = new List<nn.Module<Tensor, Tensor>>();
        var outputChannels = outChannels * scaleFactor * scaleFactor;
        for (var i = 0; i < convLayers; i++)
        {
            var cur = i != convLayers - 1 ? inChannels : outputChannels;
            layers.Add(nn.Conv2d(inChannels, cur, kernelSize, padding: (kernelSize - 1) / 2));
            if (i != convLayers - 1) layers.Add(nn.ReLU(inplace: true));
        }
        layers.Add(nn.PixelShuffle(scaleFactor));
        upsample = nn.Sequential(layers.ToArray());
        RegisterComponents();
    }

    /// <param name="triplanes">(B, 3, Ci, H, W)</param>
    /// <returns>(B, 3, Co, H x scale, W x scale)</returns>
    public override Tensor forward(Tensor triplanes)
    {
        var (b, planes) = (triplanes.shape[0], triplanes.shape[1]);
        var flat = triplanes.reshape(b * planes, triplanes.shape[2], triplanes.shape[3], triplanes.shape[4]);
        var up = upsample.call(flat);
        return up.reshape(b, planes, up.shape[1], up.shape[2], up.shape[3]);
    }
}
