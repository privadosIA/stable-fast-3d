using System.Diagnostics;
using MusicGen.Core.Testing;
using StableFast3D.Texture;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

/// <summary>The bake of the textures: the generator of the dither, the filling of the borders, the quantization and the whole thing against the golden run at 512.</summary>
[Collection(Heavy.Name)]
public class TextureBakeryTests
{
    // ---- by hand ----------------------------------------------------------------------------------------------------------------------------

    [Fact]
    public void TheGeneratorGivesNumpysNumbersForTheSameSeed()
    {
        var r = new NumpyRandom(0);
        Assert.Equal(0.5488135039273248, r.NextDouble());       // np.random.seed(0); np.random.rand(3)
        Assert.Equal(0.7151893663724195, r.NextDouble());
        Assert.Equal(0.6027633760716439, r.NextDouble());
        Assert.Equal(0.3745401188473625, new NumpyRandom(42).NextDouble());
        // the same seed, the same stream, however far it goes (the twist happens every 624 words)
        var a = new NumpyRandom(7);
        var b = new NumpyRandom(7);
        for (var i = 0; i < 5000; i++) Assert.Equal(a.NextUInt32(), b.NextUInt32());
    }

    [Fact]
    public void AValueIsMadeEightBitWithADitherOfHalfALevelEitherWay()
    {
        var q = TextureFinish.ToUint8([0f, 1f, 0.5f, 0.25f], 1, new NumpyRandom(0));
        Assert.Equal(0, q[0]);                                   // below zero: clipped
        Assert.Equal(255, q[1]);                                 // 256 clipped to 255
        Assert.InRange((int)q[2], 127, 128);                     // 128 plus or minus half a level, floored
        Assert.InRange((int)q[3], 63, 64);
    }

    [Fact]
    public void TheDitherIsLeftOutWhereTheMaskSaysSo()
    {
        float[] x = [0.5f, 0.5f, 0.5f];
        var q = TextureFinish.ToUint8(x, 1, new NumpyRandom(3), ditherMask: [0f, 0f, 0f]);
        Assert.Equal(new byte[] { 128, 128, 128 }, q);          // the mask zeroes the dither: floor(256 * 0.5) = 128
    }

    [Fact]
    public void FillingGrowsTheCoveredAreaByTheMeanColourOfItsCoveredNeighbours()
    {
        // one covered pixel in the middle of 5 x 5, colour (0.9, 0.6, 0.3): one step puts that colour on its 8 neighbours and leaves the rest
        var img = zeros(1, 3, 5, 5);
        img[0, 0, 2, 2] = 0.9; img[0, 1, 2, 2] = 0.6; img[0, 2, 2, 2] = 0.3;
        var mask = zeros(1, 1, 5, 5, dtype: ScalarType.Bool);
        mask[0, 0, 2, 2] = true;
        var filled = TextureFinish.DilateFill(img, mask, iterations: 1);
        Assert.Equal(0.9f, filled[0, 0, 1, 1].item<float>(), 5);
        Assert.Equal(0.6f, filled[0, 1, 3, 3].item<float>(), 5);
        Assert.Equal(0.3f, filled[0, 2, 2, 1].item<float>(), 5);
        Assert.Equal(0f, filled[0, 0, 0, 0].item<float>());
        // two steps reach the corners, where the patches are fewer (unfold has no padding): the values are those of upstream's dilate_fill on the same map
        var two = TextureFinish.DilateFill(img, mask, iterations: 2);
        Assert.Equal(0.225f, two[0, 0, 0, 0].item<float>(), 5);
        Assert.Equal(0.3f, two[0, 0, 0, 1].item<float>(), 5);
        Assert.Equal(0.45f, two[0, 0, 0, 2].item<float>(), 5);
        Assert.Equal(0.9f, two[0, 0, 1, 1].item<float>(), 5);
    }

    [Fact]
    public void NoStepsLeaveTheMapAsItIs()
    {
        var img = rand([1, 3, 4, 4]);
        Assert.True(TextureFinish.DilateFill(img, ones(1, 1, 4, 4, dtype: ScalarType.Bool), 0).allclose(img));
    }

    // ---- against the golden run -------------------------------------------------------------------------------------------------------------

    private static UnwrappedMesh ReferenceMesh(string name) => new(
        Fixtures.ReadStage(name, "uv_v_pos"), Fixtures.ReadStage(name, "uv_t_pos_idx"), Fixtures.ReadStage(name, "uv_v_tex"),
        Fixtures.ReadStage(name, "uv_v_nrm"), Fixtures.ReadStage(name, "uv_v_tng"));

    [RequiresCheckpointAndPictureFixtures]
    public void TheTexturesAt512AreTheReferencesOnTheFivePictures()
    {
        var decoder = RealStages.Decoder;
        var config = new Sf3dConfig();
        var failures = new List<string>();
        foreach (var name in Fixtures.Pictures)
        {
            using var _ = no_grad();
            using var scope = NewDisposeScope();
            using var scene = Fixtures.Read(Fixtures.Big(name), "scene_codes");
            using var expectedAlbedoF = Fixtures.Read(Fixtures.Big(name), "tex_albedo");
            using var expectedNormalF = Fixtures.Read(Fixtures.Big(name), "tex_normal");
            using var expectedMask = Fixtures.Read(Fixtures.Small(name), "bake_mask");
            var expectedAlbedo = Fixtures.Bytes(Fixtures.Small(name), "final_albedo_u8");
            var expectedNormal = Fixtures.Bytes(Fixtures.Small(name), "final_normal_u8");

            var sw = Stopwatch.StartNew();
            var maps = TextureBakery.Render(ReferenceMesh(name), scene[0], decoder, 512, config.Radius);
            sw.Stop();
            var maskSame = maps.Mask.to(ScalarType.Byte).eq(expectedMask).all().item<bool>();
            var albedo = TextureBakery.Pad(maps.Albedo, maps.Mask, 512);
            var normal = TextureBakery.Pad(maps.Normal, maps.Mask, 512);
            var a = Parity.Compare(albedo, expectedAlbedoF);
            var n = Parity.Compare(normal, expectedNormalF);

            var (albedoU8, normalU8) = TextureBakery.Quantize(albedo, normal, new NumpyRandom(TextureBakery.DitherSeed));
            int Differences(byte[] x, byte[] y) { var d = 0; for (var i = 0; i < x.Length; i++) if (x[i] != y[i]) d++; return d; }
            var dA = Differences(albedoU8, expectedAlbedo);
            var dN = Differences(normalU8, expectedNormal);
            Console.WriteLine($"{name}: render {sw.Elapsed.TotalMilliseconds:F0} ms, mask same {maskSame}, albedo {a}, normal {n}, 8-bit albedo {dA} of {albedoU8.Length} bytes differ, normal {dN} of {normalU8.Length}");

            if (!maskSame) failures.Add($"{name}: the coverage differs");
            if (!Criterion.Passes(a)) failures.Add($"{name} albedo: {a}");
            if (!Criterion.Passes(n)) failures.Add($"{name} normal: {n}");
            if (dA != 0) failures.Add($"{name}: {dA} bytes of the 8-bit albedo differ");
            if (dN != 0) failures.Add($"{name}: {dN} bytes of the 8-bit normal map differ");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}

public static class TextureSizesTests
{
    /// <summary>Mean absolute difference per channel, in levels of 0..255, between two RGB images of the same size.</summary>
    private static double[] MeanAbsoluteError(byte[] a, byte[] b)
    {
        var sum = new double[3];
        for (var i = 0; i < a.Length; i++) sum[i % 3] += Math.Abs(a[i] - b[i]);
        var pixels = a.Length / 3;
        return sum.Select(s => s / pixels).ToArray();
    }

    /// <summary>An encoded picture as RGB bytes.</summary>
    private static (int Size, byte[] Rgb) Decode(byte[] encoded)
    {
        var d = StableFast3D.Picture.PictureDecoder.Decode(encoded).Image;
        var rgb = new byte[d.Width * d.Height * 3];
        for (var i = 0; i < d.Width * d.Height; i++) Array.Copy(d.Pixels, i * 4, rgb, i * 3, 3);
        return (d.Width, rgb);
    }

    private static void SavePng(string path, byte[] rgb, int size)
    {
        using var bitmap = new SkiaSharp.SKBitmap(new SkiaSharp.SKImageInfo(size, size, SkiaSharp.SKColorType.Rgba8888, SkiaSharp.SKAlphaType.Opaque));
        var pixels = new byte[size * size * 4];
        for (var i = 0; i < size * size; i++) { pixels[i * 4] = rgb[i * 3]; pixels[i * 4 + 1] = rgb[i * 3 + 1]; pixels[i * 4 + 2] = rgb[i * 3 + 2]; pixels[i * 4 + 3] = 255; }
        System.Runtime.InteropServices.Marshal.Copy(pixels, 0, bitmap.GetPixels(), pixels.Length);
        using var image = SkiaSharp.SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SkiaSharp.SKEncodedImageFormat.Png, 100);
        File.WriteAllBytes(path, data.ToArray());
    }

    [Collection(Heavy.Name)]
    public sealed class Real
    {
        [RequiresCheckpointAndPictureFixtures]
        public void TheTexturesAtTheThreeSizesAreTheReferencesUpToTheNoiseOfItsJpeg()
        {
            var decoder = RealStages.Decoder;
            var config = new Sf3dConfig();
            var failures = new List<string>();
            foreach (var name in Fixtures.Pictures)
            {
                using var _ = no_grad();
                using var scene = Fixtures.Read(Fixtures.Big(name), "scene_codes");
                var mesh = new UnwrappedMesh(
                    Fixtures.ReadStage(name, "uv_v_pos"), Fixtures.ReadStage(name, "uv_t_pos_idx"), Fixtures.ReadStage(name, "uv_v_tex"),
                    Fixtures.ReadStage(name, "uv_v_nrm"), Fixtures.ReadStage(name, "uv_v_tng"));
                double floorAlbedo = 0, floorNormal = 0;
                foreach (var size in new[] { 512, 1024, 2048 })
                {
                    using var scope = NewDisposeScope();
                    var glb = new GlbReader(File.ReadAllBytes(Fixtures.Dir.PathOf($"reference/{name}/big/glb/{size}.glb")));
                    var (refAlbedoSize, refAlbedo) = Decode(glb.Image("baseColor")!.Value.Bytes);
                    var (refNormalSize, refNormal) = Decode(glb.Image("normalTexture")!.Value.Bytes);

                    var sw = Stopwatch.StartNew();
                    var baked = TextureBakery.Bake(mesh, scene[0], decoder, size, config.Radius, 0, 0);
                    sw.Stop();
                    var a = MeanAbsoluteError(baked.AlbedoRgb, refAlbedo);
                    var n = MeanAbsoluteError(baked.NormalRgb, refNormal);
                    if (size == 512) { floorAlbedo = a.Max(); floorNormal = n.Max(); }
                    // the 512 bake is byte for byte the reference's (checked above), so its error against the reference's JPEG is the noise of that JPEG; bigger textures of the same
                    // content must not be further from theirs than that (a real difference would show here, and JPEG noise only gets smaller on a bigger image)
                    else
                    {
                        if (a.Max() > floorAlbedo) failures.Add($"{name} {size}: albedo error {a.Max():F3} is above the JPEG noise of the 512 one, {floorAlbedo:F3}");
                        if (n.Max() > floorNormal) failures.Add($"{name} {size}: normal error {n.Max():F3} is above the JPEG noise of the 512 one, {floorNormal:F3}");
                    }
                    if (Environment.GetEnvironmentVariable("SF3D_TEXTURE_DIR") is { Length: > 0 } dump && name == "chair1" && size == 1024)
                    {
                        Directory.CreateDirectory(dump);
                        SavePng(Path.Combine(dump, $"{name}-{size}-albedo-port.png"), baked.AlbedoRgb, size);
                        SavePng(Path.Combine(dump, $"{name}-{size}-albedo-reference.png"), refAlbedo, size);
                        SavePng(Path.Combine(dump, $"{name}-{size}-normal-port.png"), baked.NormalRgb, size);
                        SavePng(Path.Combine(dump, $"{name}-{size}-normal-reference.png"), refNormal, size);
                    }
                    Console.WriteLine($"{name} {size}: bake {sw.Elapsed.TotalMilliseconds:F0} ms, albedo MAE per channel {string.Join(", ", a.Select(x => x.ToString("F3")))}, normal {string.Join(", ", n.Select(x => x.ToString("F3")))}, sizes {refAlbedoSize}/{refNormalSize}");
                    if (refAlbedoSize != size || refNormalSize != size) failures.Add($"{name} {size}: the reference's textures are {refAlbedoSize} and {refNormalSize}");
                }
            }
            Assert.True(failures.Count == 0, string.Join("\n", failures));
        }
    }
}
