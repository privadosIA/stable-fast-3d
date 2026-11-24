using SkiaSharp;

namespace StableFast3D.Picture;

/// <summary>A decoded picture: 8-bit RGBA (not premultiplied) as the file holds it, and the EXIF orientation (1 to 8) the file asks for.</summary>
/// <param name="ExifOrientation">1 when the file says nothing (or is not a jpeg). It is NOT applied: whoever needs it (the cut-out) does.</param>
public sealed record DecodedPicture(RasterImage Image, int ExifOrientation);

/// <summary>Decodes a png, jpg or webp (what the service takes) into 8-bit RGBA, with no colour management and no premultiplying (as the reference's <c>Image.open</c>).</summary>
public static class PictureDecoder
{
    /// <exception cref="InvalidDataException">The bytes are not a picture of a supported kind.</exception>
    public static DecodedPicture Decode(ReadOnlySpan<byte> bytes)
    {
        using var data = SKData.CreateCopy(bytes.ToArray());
        using var codec = SKCodec.Create(data);
        if (codec is null) throw new InvalidDataException("the body is not a picture: png, jpg or webp are the formats taken");
        if (codec.EncodedFormat is not (SKEncodedImageFormat.Png or SKEncodedImageFormat.Jpeg or SKEncodedImageFormat.Webp))
            throw new InvalidDataException($"the picture is a {codec.EncodedFormat}: png, jpg or webp are the formats taken");
        var info = new SKImageInfo(codec.Info.Width, codec.Info.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        var pixels = new byte[info.Width * info.Height * 4];
        unsafe
        {
            fixed (byte* p = pixels)
            {
                var result = codec.GetPixels(info, (IntPtr)p);
                if (result is not (SKCodecResult.Success or SKCodecResult.IncompleteInput))
                    throw new InvalidDataException($"the picture cannot be decoded ({result})");
            }
        }
        // SKEncodedOrigin numbers its values as EXIF does (TopLeft = 1 ... LeftBottom = 8)
        var orientation = codec.EncodedFormat == SKEncodedImageFormat.Jpeg ? (int)codec.EncodedOrigin : 1;
        return new DecodedPicture(new RasterImage(info.Width, info.Height, 4, pixels), orientation is >= 1 and <= 8 ? orientation : 1);
    }
}
