using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

public static class Geometry
{
    /// <summary>The Chamfer distance of two point sets: the mean distance of each point to the nearest of the other set, both ways, added.</summary>
    public static double Chamfer(Tensor a, Tensor b)
    {
        using var _ = no_grad();
        double OneWay(Tensor from, Tensor to)
        {
            double sum = 0;
            for (long i = 0; i < from.shape[0]; i += 2048)
            {
                using var chunk = from.narrow(0, i, Math.Min(2048, from.shape[0] - i));
                using var d = cdist(chunk, to, compute_mode: compute_mode.donot_use_mm_for_euclid_dist);   // the matrix-product shortcut is not exact for points that coincide
                sum += d.min(1).values.sum().item<float>();
            }
            return sum / from.shape[0];
        }
        return OneWay(a, b) + OneWay(b, a);
    }

}
