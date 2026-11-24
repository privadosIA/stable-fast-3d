namespace StableFast3D.Picture;

/// <summary>
/// Pillow's <c>Image.resize</c> for 8-bit pictures, step by step: the same filter support, the same normalised coefficients turned into 22-bit
/// fixed point and the same rounding, a horizontal pass and then a vertical one. Stable Fast 3D resizes the picture with it (bicubic, through
/// premultiplied alpha) and the cut-out of the reference (rembg) with the Lanczos one, so the port gets the same pixels, not just similar ones.
/// </summary>
public static class PilResample
{
    private const int PrecisionBits = 32 - 8 - 2;

    public enum Filter { Bicubic, Lanczos }

    private static double Support(Filter f) => f == Filter.Bicubic ? 2.0 : 3.0;

    private static double Sinc(double x) => x == 0.0 ? 1.0 : Math.Sin(x * Math.PI) / (x * Math.PI);

    private static double Lanczos3(double x) => x >= -3.0 && x < 3.0 ? Sinc(x) * Sinc(x / 3.0) : 0.0;

    private static double Bicubic(double x)
    {
        const double a = -0.5;
        if (x < 0.0) x = -x;
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    }

    private static double Weight(Filter f, double x) => f == Filter.Bicubic ? Bicubic(x) : Lanczos3(x);

    /// <summary>Coefficients (fixed point) and first input index of each output pixel along one axis.</summary>
    private static (int ksize, int[] bounds, int[] kk) Coefficients(int inSize, int outSize, Filter filter)
    {
        var scale = (double)inSize / outSize;
        var filterScale = Math.Max(scale, 1.0);
        var support = Support(filter) * filterScale;
        var ksize = (int)Math.Ceiling(support) * 2 + 1;
        var kk = new int[outSize * ksize];
        var bounds = new int[outSize * 2];
        var k = new double[ksize];
        var ss = 1.0 / filterScale;
        for (var xx = 0; xx < outSize; xx++)
        {
            var center = (xx + 0.5) * scale;
            var xmin = (int)(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            var xmax = (int)(center + support + 0.5);
            if (xmax > inSize) xmax = inSize;
            xmax -= xmin;
            double ww = 0.0;
            Array.Clear(k);
            for (var x = 0; x < xmax; x++)
            {
                var w = Weight(filter, (x + xmin - center + 0.5) * ss);
                k[x] = w;
                ww += w;
            }
            for (var x = 0; x < xmax; x++)
                if (ww != 0.0) k[x] /= ww;
            for (var x = 0; x < ksize; x++)
                kk[xx * ksize + x] = k[x] < 0 ? (int)(-0.5 + k[x] * (1 << PrecisionBits)) : (int)(0.5 + k[x] * (1 << PrecisionBits));
            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        return (ksize, bounds, kk);
    }

    private static byte Clip8(long v) => (byte)Math.Clamp(v >> PrecisionBits, 0, 255);

    /// <summary>Resizes an interleaved 8-bit picture (<paramref name="channels"/> bytes per pixel) as Pillow does (a copy when the size is the same).</summary>
    public static byte[] Resize(byte[] src, int width, int height, int channels, int newWidth, int newHeight, Filter filter)
    {
        if (newWidth == width && newHeight == height) return (byte[])src.Clone();
        var cur = src;
        var w = width;
        if (newWidth != width)
        {
            var (ksize, bounds, kk) = Coefficients(width, newWidth, filter);
            var dst = new byte[newWidth * height * channels];
            for (var y = 0; y < height; y++)
                for (var xx = 0; xx < newWidth; xx++)
                {
                    var xmin = bounds[xx * 2];
                    var xmax = bounds[xx * 2 + 1];
                    for (var c = 0; c < channels; c++)
                    {
                        long ss = 1L << (PrecisionBits - 1);
                        for (var x = 0; x < xmax; x++) ss += (long)cur[(y * w + xmin + x) * channels + c] * kk[xx * ksize + x];
                        dst[(y * newWidth + xx) * channels + c] = Clip8(ss);
                    }
                }
            cur = dst;
            w = newWidth;
        }
        if (newHeight != height)
        {
            var (ksize, bounds, kk) = Coefficients(height, newHeight, filter);
            var dst = new byte[w * newHeight * channels];
            for (var yy = 0; yy < newHeight; yy++)
            {
                var ymin = bounds[yy * 2];
                var ymax = bounds[yy * 2 + 1];
                for (var x = 0; x < w; x++)
                    for (var c = 0; c < channels; c++)
                    {
                        long ss = 1L << (PrecisionBits - 1);
                        for (var y = 0; y < ymax; y++) ss += (long)cur[((ymin + y) * w + x) * channels + c] * kk[yy * ksize + y];
                        dst[(yy * w + x) * channels + c] = Clip8(ss);
                    }
            }
            cur = dst;
        }
        return cur;
    }

    /// <summary>
    /// <c>Image.resize</c> of an RGBA picture, which Pillow does through premultiplied alpha (<c>RGBA</c> to <c>RGBa</c>, resize, back): what
    /// <c>SF3D.prepare_image</c> calls with no filter given, that is bicubic. A picture already of that size is returned as it is.
    /// </summary>
    public static RasterImage ResizeRgba(RasterImage rgba, int newWidth, int newHeight, Filter filter = Filter.Bicubic)
    {
        if (rgba.Channels != 4) throw new ArgumentException("an RGBA picture is needed", nameof(rgba));
        if (newWidth == rgba.Width && newHeight == rgba.Height) return rgba;
        var pre = Premultiply(rgba.Pixels);
        var resized = Resize(pre, rgba.Width, rgba.Height, 4, newWidth, newHeight, filter);
        return new RasterImage(newWidth, newHeight, 4, Unpremultiply(resized));
    }

    // Pillow's MULDIV255: (a * b) / 255 rounded
    private static int MulDiv255(int a, int b)
    {
        var t = a * b + 128;
        return ((t >> 8) + t) >> 8;
    }

    /// <summary>Pillow's <c>RGBA</c> to <c>RGBa</c>.</summary>
    public static byte[] Premultiply(byte[] rgba)
    {
        var o = new byte[rgba.Length];
        for (var i = 0; i < rgba.Length; i += 4)
        {
            int a = rgba[i + 3];
            o[i] = (byte)MulDiv255(rgba[i], a);
            o[i + 1] = (byte)MulDiv255(rgba[i + 1], a);
            o[i + 2] = (byte)MulDiv255(rgba[i + 2], a);
            o[i + 3] = (byte)a;
        }
        return o;
    }

    /// <summary>Pillow's <c>RGBa</c> to <c>RGBA</c>: the colour is left as it is where the alpha is 0 or 255 and divided by it (integer division, clipped) elsewhere.</summary>
    public static byte[] Unpremultiply(byte[] rgba)
    {
        var o = new byte[rgba.Length];
        for (var i = 0; i < rgba.Length; i += 4)
        {
            int a = rgba[i + 3];
            for (var c = 0; c < 3; c++)
                o[i + c] = a is 0 or 255 ? rgba[i + c] : (byte)Math.Min(255 * rgba[i + c] / a, 255);
            o[i + 3] = (byte)a;
        }
        return o;
    }
}
