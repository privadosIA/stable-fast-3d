using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D;

/// <summary>
/// Bounded use of memory. TorchSharp frees a tensor when it is disposed (or, much later, when the garbage collector finalizes it) and a <c>NewDisposeScope</c> disposes everything created
/// inside it only when it ends, so a whole forward pass under one scope keeps every intermediate alive (the feed-forward of the backbone alone makes several (27,648 x 8,192) tensors a block).
/// Running the body of a module, of a stage or of one iteration in a scope of its own and moving only its result out keeps the peak to what one step needs, as reference counting does in PyTorch.
/// </summary>
internal static class Frugal
{
    /// <summary>Runs <paramref name="body"/> in a dispose scope and returns its result, which outlives the scope; every other tensor it made is freed when it ends.</summary>
    public static Tensor Run(Func<Tensor> body)
    {
        using var scope = NewDisposeScope();
        return body().MoveToOuterDisposeScope();
    }

    public static (Tensor, Tensor) Run(Func<(Tensor, Tensor)> body)
    {
        using var scope = NewDisposeScope();
        var (a, b) = body();
        return (a.MoveToOuterDisposeScope(), b.MoveToOuterDisposeScope());
    }
}
