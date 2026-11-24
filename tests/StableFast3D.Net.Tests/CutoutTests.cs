using StableFast3D.Picture;

namespace StableFast3D.Net.Tests;

/// <summary>The cut-out of the container's service (<c>server.cutout</c>): u2net on ONNX Runtime, over the weights folder's <c>rembg/u2net.onnx</c>.</summary>
public class CutoutTests
{
    private const string Input = "reference/cutout/input.png";
    private const string Expected = "reference/cutout/big/rgba.safetensors";

    [RequiresCheckpointAndFixtures(Input, Expected)]
    public void APictureWithNoTransparencyIsCutOutAsRembgDoesIt()
    {
        using var cutout = new U2NetCutout(Path.Combine(Checkpoint.Dir!, "rembg", "u2net.onnx"));
        var got = cutout.Cutout(File.ReadAllBytes(Fixtures.Dir.PathOf(Input)));
        var expected = Fixtures.Bytes(Expected, "rgba");
        Assert.Equal((1024, 1024, 4), (got.Width, got.Height, got.Channels));
        Assert.Equal(expected.Length, got.Pixels.Length);

        // ONNX Runtime of the port is not the build the reference ran (other versions, other kernels): the mask may differ by a level in a few pixels
        int differing = 0, worst = 0;
        for (var i = 0; i < expected.Length; i++)
        {
            var d = Math.Abs(expected[i] - got.Pixels[i]);
            if (d > 0) differing++;
            worst = Math.Max(worst, d);
        }
        Assert.True(differing == 0, $"{differing} of {expected.Length} values differ, by up to {worst}");
    }

    [RequiresCheckpointAndFixtures(Input)]
    public void APictureThatHasItsOwnAlphaIsUsedAsItIs()
    {
        using var cutout = new U2NetCutout(Path.Combine(Checkpoint.Dir!, "rembg", "u2net.onnx"));
        var picture = new DecodedPicture(new RasterImage(2, 1, 4, [9, 8, 7, 255, 1, 2, 3, 100]), 1);
        Assert.Same(picture.Image, cutout.Cutout(picture));
    }

    [Fact]
    public void ACutoutModelThatIsNotThereIsReportedByName()
    {
        var ex = Assert.Throws<FileNotFoundException>(() => new U2NetCutout("/nowhere/u2net.onnx"));
        Assert.Contains("/nowhere/u2net.onnx", ex.Message);
    }
}
