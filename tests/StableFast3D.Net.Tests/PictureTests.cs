using SkiaSharp;
using StableFast3D.Picture;
using TorchSharp;
using static TorchSharp.torch;
using MusicGen.Core.Testing;

namespace StableFast3D.Net.Tests;

public class PictureTests
{
    private static byte[] Encode(int w, int h, Func<int, int, SKColor> pixel, SKEncodedImageFormat format, int quality = 100)
    {
        using var bmp = new SKBitmap(new SKImageInfo(w, h, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++) bmp.SetPixel(x, y, pixel(x, y));
        using var img = SKImage.FromBitmap(bmp);
        return img.Encode(format, quality).ToArray();
    }

    [Fact]
    public void APngKeepsItsAlphaAndItsColourUnpremultiplied()
    {
        var png = Encode(3, 2, (x, y) => new SKColor((byte)(40 * x + 10), 200, 100, (byte)(y == 0 ? 255 : 60)), SKEncodedImageFormat.Png);
        var d = PictureDecoder.Decode(png);
        Assert.Equal((3, 2, 4, 1), (d.Image.Width, d.Image.Height, d.Image.Channels, d.ExifOrientation));
        Assert.Equal(new byte[] { 10, 200, 100, 255 }, d.Image.Pixels[..4]);
        Assert.Equal(new byte[] { 10, 200, 100, 60 }, d.Image.Pixels[12..16]);
    }

    [Theory]
    [InlineData(SKEncodedImageFormat.Png)]
    [InlineData(SKEncodedImageFormat.Jpeg)]
    [InlineData(SKEncodedImageFormat.Webp)]
    public void PngJpgAndWebpAreTheFormatsTaken(SKEncodedImageFormat format)
    {
        var d = PictureDecoder.Decode(Encode(8, 5, (x, y) => new SKColor(120, 60, 30, 255), format));
        Assert.Equal((8, 5), (d.Image.Width, d.Image.Height));
    }

    [Fact]
    public void WhatIsNotAPictureIsRefusedWithASentence()
    {
        var ex = Assert.Throws<InvalidDataException>(() => PictureDecoder.Decode("not a picture"u8));
        Assert.Contains("png, jpg or webp", ex.Message);
        // a 1 x 1 gif (Skia cannot encode one): a picture, but not of a kind the service takes
        byte[] gif = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 1, 0, 1, 0, 0x80, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0x21, 0xf9, 4, 1, 0, 0, 0, 0, 0x2c, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 2, 0x44, 1, 0, 0x3b];
        var ex2 = Assert.Throws<InvalidDataException>(() => PictureDecoder.Decode(gif));
        Assert.Contains("png, jpg or webp", ex2.Message);
    }

    [Fact]
    public void OnlyAnAlphaBelow250MakesATransparentPicture()
    {
        byte[] opaque = [1, 2, 3, 255, 1, 2, 3, 250];
        byte[] cut = [1, 2, 3, 255, 1, 2, 3, 249];
        Assert.False(U2NetCutout.HasTransparency(new RasterImage(2, 1, 4, opaque)));
        Assert.True(U2NetCutout.HasTransparency(new RasterImage(2, 1, 4, cut)));
    }

    [Theory]
    [InlineData(2, 3, 2, new byte[] { 3, 2, 1, 6, 5, 4 })]        // mirrored left to right
    [InlineData(3, 3, 2, new byte[] { 6, 5, 4, 3, 2, 1 })]        // turned 180
    [InlineData(4, 3, 2, new byte[] { 4, 5, 6, 1, 2, 3 })]        // mirrored top to bottom
    public void ExifOrientationsThatKeepTheSizeTurnThePicture(int orientation, int w, int h, byte[] expected)
    {
        var img = new RasterImage(w, h, 1, [1, 2, 3, 4, 5, 6]);
        Assert.Equal(expected, U2NetCutout.ExifTurn(img, orientation).Pixels);
    }

    [Theory]
    [InlineData(5, new byte[] { 1, 4, 2, 5, 3, 6 })]              // transposed
    [InlineData(6, new byte[] { 4, 1, 5, 2, 6, 3 })]              // 90 degrees clockwise
    [InlineData(7, new byte[] { 6, 3, 5, 2, 4, 1 })]              // transversed
    [InlineData(8, new byte[] { 3, 6, 2, 5, 1, 4 })]              // 90 degrees counterclockwise
    public void ExifOrientationsThatSwapTheSidesTurnThePicture(int orientation, byte[] expected)
    {
        var img = new RasterImage(3, 2, 1, [1, 2, 3, 4, 5, 6]);
        var turned = U2NetCutout.ExifTurn(img, orientation);
        Assert.Equal((2, 3), (turned.Width, turned.Height));
        Assert.Equal(expected, turned.Pixels);
    }

    [RequiresFixtures("reference/pil/cases.safetensors")]
    public void AJpegThatSaysItIsTurnedIsTurnedByTheCutoutAsTheReferenceDoes()
    {
        const string cases = "reference/pil/cases.safetensors";
        var d = PictureDecoder.Decode(Fixtures.Bytes(cases, "exif_jpeg"));
        Assert.Equal((30, 20, 6), (d.Image.Width, d.Image.Height, d.ExifOrientation));
        var turned = U2NetCutout.ExifTurn(d.Image, d.ExifOrientation);
        Assert.Equal((20, 30), (turned.Width, turned.Height));
        // Pillow's turned picture (RGB); the decoders of Pillow and Skia are both libjpeg-turbo
        var expected = Fixtures.Bytes(cases, "exif_turned");
        var rgb = new byte[expected.Length];
        for (var i = 0; i < turned.Width * turned.Height; i++) Array.Copy(turned.Pixels, i * 4, rgb, i * 3, 3);
        Assert.Equal(expected, rgb);
    }

    // ---- what the model sees: the five pictures of the golden run ------------------------------------------------------------------------------

    [RequiresPictureFixtures]
    public void TheModelInputMatchesTheReferenceOnTheFivePictures()
    {
        var config = new Sf3dConfig();
        foreach (var name in Fixtures.Pictures)
        {
            var picture = PictureDecoder.Decode(File.ReadAllBytes(Fixtures.Dir.PathOf($"pictures/{name}.png")));
            Assert.True(U2NetCutout.HasTransparency(picture.Image), $"{name} has its own alpha: the container uses it as it is");
            var (rgb, mask) = ModelInput.Prepare(picture.Image, config.CondImageSize, config.BackgroundColor);

            using var expectedMask = Fixtures.Read(Fixtures.Small(name), "mask_cond_pre");
            using var expectedRgb = Fixtures.Read(Fixtures.Big(name), "rgb_cond_pre");
            var m = Parity.Compare(mask, expectedMask);
            var c = Parity.Compare(rgb, expectedRgb);
            Assert.True(m.MaxAbs <= 1e-4 && m.CosSim >= 0.9999, $"{name} mask: {m}");
            Assert.True(c.MaxAbs <= 1e-4 && c.CosSim >= 0.9999, $"{name} rgb: {c}");

            // upstream then resizes 512 to 512 with an antialiased bilinear filter: the identity
            using var after = Fixtures.Read(Fixtures.Big(name), "rgb_cond");
            Assert.Equal(0.0, Parity.Compare(expectedRgb, after.reshape(expectedRgb.shape)).MaxAbs);
        }
    }

    [Fact]
    public void ObjectsAreCompositedOverGreyByTheirAlpha()
    {
        // one opaque red pixel, one transparent one, one half transparent
        var rgba = new RasterImage(3, 1, 4, [255, 0, 0, 255, 10, 20, 30, 0, 255, 255, 255, 128]);
        var (rgb, mask) = ModelInput.Prepare(rgba, 3, [0.5f, 0.5f, 0.5f]);
        Assert.Equal([1f, 0f, 0f], rgb[0, 0].data<float>().ToArray());
        Assert.Equal([0.5f, 0.5f, 0.5f], rgb[0, 1].data<float>().ToArray());
        var half = rgb[0, 2].data<float>().ToArray();
        Assert.Equal(0.5f + (128 / 255f) * 0.5f, half[0], 6);
        Assert.Equal(1f, mask[0, 0, 0].item<float>());
        Assert.Equal(0f, mask[0, 1, 0].item<float>());
        Assert.Equal(128 / 255f, mask[0, 2, 0].item<float>(), 6);
    }
}
