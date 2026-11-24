using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Camera;

/// <summary>
/// <c>sf3d.models.camera.LinearCameraEmbedder</c>: the camera-to-world matrix (16 numbers) and the normalised intrinsics (9) of each view, flattened and
/// joined, through one linear layer (25 to 768). The weights are <c>camera_embedder.linear.{weight,bias}</c> of the checkpoint (the field is called
/// <c>linear</c> so the parameters carry that name).
/// </summary>
public sealed class LinearCameraEmbedder : nn.Module<Tensor, Tensor, Tensor>
{
    public const int InChannels = 25;
    public const int OutChannels = 768;

    public readonly nn.Module<Tensor, Tensor> linear;

    public LinearCameraEmbedder() : base(nameof(LinearCameraEmbedder))
    {
        linear = nn.Linear(InChannels, OutChannels);
        RegisterComponents();
    }

    /// <param name="c2wCond">(B, Nv, 4, 4)</param>
    /// <param name="intrinsicNormedCond">(B, Nv, 3, 3)</param>
    /// <returns>(B, Nv, 768)</returns>
    public override Tensor forward(Tensor c2wCond, Tensor intrinsicNormedCond)
    {
        var b = c2wCond.shape[0];
        var nv = c2wCond.shape[1];
        var cond = cat([c2wCond.reshape(b, nv, -1), intrinsicNormedCond.reshape(b, nv, -1)], -1);
        if (cond.shape[^1] != InChannels) throw new ArgumentException($"the camera has {cond.shape[^1]} numbers, the embedder takes {InChannels}");
        return linear.call(cond);
    }
}
