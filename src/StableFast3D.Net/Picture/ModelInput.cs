using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Picture;

/// <summary>
/// From the cut-out to what the model sees, as <c>SF3D.prepare_image</c> does it: the RGBA picture resized to 512 x 512 (Pillow, bicubic, through
/// premultiplied alpha), as floats in 0..1, and the colour interpolated from the background grey to the picture's by the alpha.
/// </summary>
public static class ModelInput
{
    /// <summary>(rgb (H, W, 3), mask (H, W, 1)), float32, H = W = <paramref name="size"/>. The antialiased resize that follows in upstream (<c>ImageProcessor</c>) is the identity at the same size.</summary>
    public static (Tensor Rgb, Tensor Mask) Prepare(RasterImage rgba, int size, IReadOnlyList<float> background)
    {
        var resized = PilResample.ResizeRgba(rgba, size, size);
        var n = size * size;
        var all = new float[n * 4];
        for (var i = 0; i < all.Length; i++) all[i] = resized.Pixels[i] / 255.0f;
        using var img = tensor(all, [size, size, 4]);
        var mask = img[TensorIndex.Ellipsis, TensorIndex.Slice(3, 4)].contiguous().clamp(0, 1);
        var rgb = img[TensorIndex.Ellipsis, TensorIndex.Slice(0, 3)].contiguous().clamp(0, 1);
        using var bg = tensor(background.ToArray(), [1, 1, 3]);
        return (lerp(bg, rgb, mask), mask);
    }
}
