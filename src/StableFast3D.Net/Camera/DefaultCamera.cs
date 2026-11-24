using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Camera;

/// <summary>The camera the picture is taken to have (<c>sf3d.utils.default_cond_c2w</c> and <c>create_intrinsic_from_fov_deg</c>): the same one for every picture.</summary>
public static class DefaultCamera
{
    /// <summary>The camera-to-world matrix, (4, 4): looks at the origin along -x from <paramref name="distance"/>, with z up.</summary>
    public static Tensor CameraToWorld(float distance) => tensor(
        new float[] { 0, 0, 1, distance, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1 }, [4, 4]);

    /// <summary>The intrinsics, (3, 3), in pixels and normalised by the picture's width and height (as the model is given them).</summary>
    public static (Tensor Intrinsic, Tensor Normed) Intrinsics(double fovyDeg, int height, int width)
    {
        // upstream computes the focal length in float64 (numpy) and stores it in a float32 matrix, then divides in float32
        var fov = fovyDeg * Math.PI / 180.0;
        var focal = (float)(0.5 * height / Math.Tan(0.5 * fov));
        var intrinsic = new float[] { focal, 0, width / 2.0f, 0, focal, height / 2.0f, 0, 0, 1 };
        var normed = (float[])intrinsic.Clone();
        normed[2] /= width;
        normed[5] /= height;
        normed[0] /= width;
        normed[4] /= height;
        return (tensor(intrinsic, [3, 3]), tensor(normed, [3, 3]));
    }
}
