using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Texture;

/// <summary>What is done to a baked map before it is a texture: the filling of the borders of the islands (<c>dilate_fill</c>) and the quantization to 8 bits (<c>float32_to_uint8_np</c>).</summary>
public static class TextureFinish
{
    /// <summary>
    /// <c>sf3d.models.utils.dilate_fill</c>: grows the covered area <paramref name="iterations"/> pixels outwards, each new pixel taking the mean colour of the covered ones around it, so
    /// that bilinear filtering and mip-mapping do not drag in the empty background of the texture.
    /// </summary>
    /// <param name="img">(1, 3, H, W)</param>
    /// <param name="mask">(1, 1, H, W), true where covered</param>
    public static Tensor DilateFill(Tensor img, Tensor mask, int iterations)
    {
        using var _ = no_grad();
        var oldMask = mask.to(ScalarType.Float32);
        var oldImg = img;
        var (h, w) = (img.shape[2], img.shape[3]);
        using var maskKernel = ones([1, 1, 3, 3], dtype: oldMask.dtype, device: oldMask.device);

        for (var i = 0; i < iterations; i++)
        {
            var newMask = F.max_pool2d(oldMask, 3, 1, 1);

            var imgUnfold = F.unfold(oldImg, (3, 3)).view(1, 3, 3 * 3, -1);
            var maskUnfold = F.unfold(oldMask, (3, 3)).view(1, 1, 3 * 3, -1);
            var newMaskUnfold = F.unfold(newMask, (3, 3)).view(1, 1, 3 * 3, -1);

            // the mean colour of the covered pixels around, put on the new ones
            var meanColor = (imgUnfold.sum(2) / maskUnfold.sum(2).clip(1)).unsqueeze(2);
            var fillColor = (meanColor * newMaskUnfold).view(1, 3 * 3 * 3, -1);

            var maskConv = F.conv2d(newMask, maskKernel, padding: [1L, 1L]);
            var newImg = F.fold(fillColor, (h, w), (3, 3)) / maskConv.clamp(1);

            var diffMask = newMask - oldMask;
            oldMask = newMask;
            oldImg = lerp(oldImg, newImg, diffMask);
        }
        return oldImg;
    }

    /// <summary>
    /// <c>float32_to_uint8_np</c>: <c>floor(256 x + dither)</c> clipped to 0..255, the dither being <c>rand - 0.5</c> (optionally times a mask) with one draw per pixel from numpy's
    /// generator. float32 as numpy does it.
    /// </summary>
    /// <param name="x">(H x W x channels) floats</param>
    /// <param name="ditherMask">per pixel, 1 where the dither applies and 0 where it does not (all ones when null)</param>
    public static byte[] ToUint8(float[] x, int channels, NumpyRandom random, float[]? ditherMask = null)
    {
        var pixels = x.Length / channels;
        var dither = new float[pixels];
        for (var i = 0; i < pixels; i++)
        {
            var r = (float)random.NextDouble();                    // rand(...).astype(np.float32)
            dither[i] = 1.0f * r - 0.5f;
            if (ditherMask is not null) dither[i] *= ditherMask[i];
        }
        var o = new byte[x.Length];
        for (var p = 0; p < pixels; p++)
            for (var c = 0; c < channels; c++)
            {
                var v = MathF.Floor(256.0f * x[p * channels + c] + dither[p]);
                o[p * channels + c] = (byte)Math.Clamp(v, 0f, 255f);
            }
        return o;
    }
}
