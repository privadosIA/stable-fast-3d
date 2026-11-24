using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Texture;

/// <summary>
/// <c>uv_unwrapper.Unwrapper</c> (spike 7.1: the first step, the alignment of the mesh with its principal axes, which is the one that depends on a random generator).
/// The rest of the unwrapper is task 7.2.
/// </summary>
public static class UvUnwrapper
{
    /// <summary>
    /// <c>torch.pca_lowrank(A, q)</c> of a (m, 3) cloud: Halko's randomized subspace iteration (a random (3, q) start from <paramref name="generator"/>, QR, two rounds of
    /// <c>A^T</c> and <c>A</c> products with a QR after each, then an SVD of the small matrix) on the centred cloud. Returns V, (3, q): the principal directions.
    /// </summary>
    public static Tensor PcaLowrank(Tensor a, Generator generator, int q = 2, int niter = 2)
    {
        using var _ = no_grad();
        var (m, n) = (a.shape[0], a.shape[1]);
        if (m < n) throw new NotSupportedException("a cloud with fewer points than coordinates is not unwrapped");
        var centred = a - a.mean([0], keepdim: true);
        var r = randn([n, q], dtype: a.dtype, device: a.device, generator: generator);
        var qm = linalg.qr(centred.matmul(r)).Q;
        for (var i = 0; i < niter; i++)
        {
            qm = linalg.qr(centred.t().matmul(qm)).Q;
            qm = linalg.qr(centred.matmul(qm)).Q;
        }
        var b = qm.t().matmul(centred);
        var (_, _, vh) = linalg.svd(b, fullMatrices: false);
        return vh.t();
    }

    /// <summary>
    /// <c>_align_mesh_with_main_axis</c>: rotates the mesh (positions and normals) so that its first principal axis lies along the canonical axis it is closest to, the second
    /// along the next one and the third (their cross product) along what is left. Upstream seeds the global generator with 0 first; this uses a generator of its own seeded
    /// with 0, which gives the same numbers (the same algorithm) without touching, or depending on, the state other threads share.
    /// </summary>
    public static (Tensor Positions, Tensor Normals) AlignMeshWithMainAxis(Tensor positions, Tensor normals)
    {
        using var _ = no_grad();
        using var generator = new Generator(0, positions.device);
        var v = PcaLowrank(positions, generator, q: 2);
        var mainAxis = F.normalize(v[TensorIndex.Colon, 0], p: 2, dim: -1, eps: 1e-6);
        var secondary = v[TensorIndex.Colon, 1];
        secondary = F.normalize(secondary - (secondary * mainAxis).sum(-1, keepdim: true) * mainAxis, p: 2, dim: -1, eps: 1e-6);
        var third = F.normalize(linalg.cross(mainAxis, secondary, dim: -1), p: 2, dim: -1, eps: 1e-6);

        int iMain = (int)mainAxis.abs().argmax().item<long>(), iSecondary = (int)secondary.abs().argmax().item<long>(), iThird = (int)third.abs().argmax().item<long>();
        var missing = new HashSet<int> { 0, 1, 2 };
        var current = 1;
        while (new HashSet<int> { iMain, iSecondary, iThird }.Count != 3)
        {
            missing = [0, 1, 2];
            missing.ExceptWith([iMain, iSecondary, iThird]);
            var free = missing.First();
            // the third axis contributes least to the shape: it takes the free one first, then the second
            if (current == 1) iThird = free;
            else if (current == 2) iSecondary = free;
            else throw new InvalidOperationException("Could not find 3 unique axis");
            current++;
        }
        var axes = new Tensor[3];
        axes[iMain] = mainAxis;
        axes[iSecondary] = secondary;
        axes[iThird] = third;
        var rot = stack(axes, dim: 1).t();
        return (einsum("ij,nj->ni", rot, positions), einsum("ij,nj->ni", rot, normals));
    }
}
