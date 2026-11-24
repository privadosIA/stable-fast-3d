using MusicGen.Core.Testing;
using StableFast3D.Weights;

namespace StableFast3D.Net.Tests;

/// <summary>
/// The real weights folder, taken from SF3D_CKPT: the layout of the container's <c>/ckpt</c> (<c>sf3d/model.safetensors</c>, ...), never inside the repository.
/// Tests that need it are marked <see cref="RequiresCheckpointAttribute"/> and are reported as skipped, with the reason, without it.
/// </summary>
public static class Checkpoint
{
    public static readonly CheckpointEnvironment Env = new("SF3D_CKPT", deviceVariable: "SF3D_DEVICE");

    private static readonly Lazy<Sf3dCheckpoint> Opened = new(() => Sf3dCheckpoint.Open(Dir!));

    public static string? Dir => Env.Dir;

    public static Sf3dCheckpoint Weights => Opened.Value;
}

/// <summary>[Fact] that is skipped (with the reason) unless SF3D_CKPT points to an existing directory.</summary>
public sealed class RequiresCheckpointAttribute : RequiresCheckpointFactAttribute
{
    public RequiresCheckpointAttribute() : base(Checkpoint.Env) { }
}

/// <summary>A directory of its own for a test, removed with it.</summary>
public sealed class TempDir : IDisposable
{
    public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "sf3d-tests-" + Guid.NewGuid().ToString("N"));

    public TempDir() => Directory.CreateDirectory(Path);

    public string File(string name) => System.IO.Path.Combine(Path, name);

    public void Dispose() => Directory.Delete(Path, true);
}
