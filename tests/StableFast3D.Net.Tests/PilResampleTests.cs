using StableFast3D.Picture;

namespace StableFast3D.Net.Tests;

/// <summary>The resampling is Pillow's, pixel by pixel: the cases are Pillow's own output (<c>tools/dump_pil.py</c>) for the calls the reference makes.</summary>
public class PilResampleTests
{
    private const string Cases = "reference/pil/cases.safetensors";

    private static byte[] B(string key) => Fixtures.Bytes(Cases, key);

    [RequiresFixturesTheory(Cases)]
    [InlineData("down", 200, 150, 100, 100)]
    [InlineData("up", 60, 40, 100, 100)]
    [InlineData("wide", 97, 61, 50, 70)]
    public void RgbaIsResizedThroughPremultipliedAlphaAsPillowDoes(string name, int w, int h, int newW, int newH)
    {
        var got = PilResample.ResizeRgba(new RasterImage(w, h, 4, B($"resize_rgba_{name}_in")), newW, newH);
        Assert.Equal(B($"resize_rgba_{name}_out"), got.Pixels);
    }

    [RequiresFixturesTheory(Cases)]
    [InlineData("down", 200, 150, 64, 64)]
    [InlineData("up", 40, 30, 90, 70)]
    public void RgbAndOneChannelAreResizedWithLanczosAsPillowDoes(string name, int w, int h, int newW, int newH)
    {
        Assert.Equal(B($"resize_rgb_{name}_out"), PilResample.Resize(B($"resize_rgb_{name}_in"), w, h, 3, newW, newH, PilResample.Filter.Lanczos));
        Assert.Equal(B($"resize_l_{name}_out"), PilResample.Resize(B($"resize_l_{name}_in"), w, h, 1, newW, newH, PilResample.Filter.Lanczos));
    }

    [RequiresFixtures(Cases)]
    public void ThePictureIsPastedOverTransparentWithTheMaskAsPillowDoes()
    {
        var got = U2NetCutout.Paste(new RasterImage(50, 40, 3, B("composite_rgb")), B("composite_mask"));
        Assert.Equal(B("composite_out"), got.Pixels);
    }

    [Fact]
    public void ASizeThatIsAlreadyRightIsNotResampled()
    {
        var img = new RasterImage(4, 4, 4, Enumerable.Range(0, 64).Select(i => (byte)(i * 3)).ToArray());
        Assert.Same(img, PilResample.ResizeRgba(img, 4, 4));
    }

    [Fact]
    public void PremultiplyingKeepsTheExtremesAndRoundsTheRest()
    {
        byte[] rgba = [200, 100, 50, 255, 200, 100, 50, 0, 200, 100, 50, 128];
        var pre = PilResample.Premultiply(rgba);
        Assert.Equal(new byte[] { 200, 100, 50, 255, 0, 0, 0, 0, 100, 50, 25, 128 }, pre);
        var back = PilResample.Unpremultiply(pre);
        Assert.Equal(new byte[] { 200, 100, 50, 255, 0, 0, 0, 0, 199, 99, 49, 128 }, back);   // what the rounding loses is lost, as in Pillow
    }
}
