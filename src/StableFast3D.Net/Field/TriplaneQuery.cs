using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Field;

/// <summary><c>SF3D.query_triplane</c>: what the three planes hold at some points of space.</summary>
public static class TriplaneQuery
{
    /// <summary>
    /// Each point is projected on the xy, xz and yz planes (its coordinates scaled from <c>[-radius, radius]</c> to <c>[-1, 1]</c>), the three planes are sampled
    /// bilinearly (corners aligned, zeros outside) and their channels are joined: the three planes one after the other.
    /// </summary>
    /// <param name="positions">(N, 3), or (B, N, 3)</param>
    /// <param name="triplanes">(3, C, H, W), or (B, 3, C, H, W)</param>
    /// <returns>(N, 3 x C), or (B, N, 3 x C)</returns>
    public static Tensor Query(Tensor positions, Tensor triplanes, double radius)
    {
        var batched = positions.ndim == 3;
        if (!batched)
        {
            triplanes = triplanes.unsqueeze(0);
            positions = positions.unsqueeze(0);
        }
        if (triplanes.ndim != 5 || positions.ndim != 3) throw new ArgumentException("positions are (N, 3) or (B, N, 3) and triplanes (3, C, H, W) or (B, 3, C, H, W)");

        // scale_tensor(positions, (-radius, radius), (-1, 1)): the same operations in the same order as upstream, with python-float scalars
        positions = (positions - (-radius)) / (radius - (-radius));
        positions = positions * (1.0 - (-1.0)) + (-1.0);

        var xy = positions.index_select(-1, tensor(new long[] { 0, 1 }, device: positions.device));
        var xz = positions.index_select(-1, tensor(new long[] { 0, 2 }, device: positions.device));
        var yz = positions.index_select(-1, tensor(new long[] { 1, 2 }, device: positions.device));
        var indices2d = stack([xy, xz, yz], dim: -3).to(triplanes.dtype);                              // (B, 3, N, 2)

        var (b, planes, c, h, w) = (triplanes.shape[0], triplanes.shape[1], triplanes.shape[2], triplanes.shape[3], triplanes.shape[4]);
        var n = positions.shape[1];
        var sampled = F.grid_sample(
            triplanes.reshape(b * planes, c, h, w).to(ScalarType.Float32),
            indices2d.reshape(b * planes, 1, n, 2).to(ScalarType.Float32),
            mode: GridSampleMode.Bilinear, padding_mode: GridSamplePaddingMode.Zeros, align_corners: true);   // (B x 3, C, 1, N)
        var joined = sampled.reshape(b, planes, c, n).permute(0, 3, 1, 2).reshape(b, n, planes * c);          // (B, N, 3 x C)
        return batched ? joined : joined.squeeze(0);
    }
}
