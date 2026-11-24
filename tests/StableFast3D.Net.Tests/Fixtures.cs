using MusicGen.Core.Testing;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

/// <summary>
/// The fixtures of the golden run (<c>tools/golden_all.sh</c>), checked against the sha256 table of <c>tests/fixtures/MANIFEST.md</c> before they are used.
/// What is big or not ours to commit (the activations, the pictures) is regenerated: a test that needs it is skipped, saying so, when it is not there.
/// </summary>
public static class Fixtures
{
    public static readonly string[] Pictures = ["chair1", "character1", "axe", "tree", "stylized-rocks"];

    private static readonly Lazy<FixtureDirectory> Directory_ = new(FixtureDirectory.Find);

    public static FixtureDirectory Dir => Directory_.Value;

    public static bool Has(string file) => File.Exists(Path.Combine(Dir.Dir, file));

    public static string? Missing(params string[] files) => files.FirstOrDefault(f => !Has(f)) is { } m ? $"the fixture {m} is not there: run tools/golden_all.sh" : null;

    /// <summary>One tensor of a safetensors fixture, read without loading the rest of the file.</summary>
    public static Tensor Read(string file, string key) => CheckpointIndex.Open(Dir.PathOf(file)).Read(key);

    public static byte[] Bytes(string file, string key)
    {
        using var t = Read(file, key);
        return t.data<byte>().ToArray();
    }

    public static string Small(string picture) => $"reference/{picture}/small.safetensors";

    public static string Big(string picture) => $"reference/{picture}/big/stages.safetensors";
}

/// <summary>[Fact] that is skipped (with the reason) unless every fixture it names is there.</summary>
public sealed class RequiresFixturesAttribute : FactAttribute
{
    public RequiresFixturesAttribute(params string[] files) => Skip = Fixtures.Missing(files);
}

/// <summary>[Fact] that needs the real weights folder (SF3D_CKPT) and the fixtures it names.</summary>
public sealed class RequiresCheckpointAndFixturesAttribute : FactAttribute
{
    public RequiresCheckpointAndFixturesAttribute(params string[] files) => Skip = Checkpoint.Env.NoCheckpointReason ?? Fixtures.Missing(files);
}

/// <summary>[Theory] that is skipped (with the reason) unless every fixture it names is there.</summary>
public sealed class RequiresFixturesTheoryAttribute : TheoryAttribute
{
    public RequiresFixturesTheoryAttribute(params string[] files) => Skip = Fixtures.Missing(files);
}

/// <summary>[Fact] that needs the five pictures of the golden run and the stages recorded over them.</summary>
public sealed class RequiresPictureFixturesAttribute : FactAttribute
{
    public RequiresPictureFixturesAttribute() =>
        Skip = Fixtures.Missing(Fixtures.Pictures.SelectMany(p => new[] { $"pictures/{p}.png", Fixtures.Small(p), Fixtures.Big(p) }).ToArray());
}

/// <summary>[Fact] that needs the real weights, the five pictures of the golden run and the stages recorded over them.</summary>
public sealed class RequiresCheckpointAndPictureFixturesAttribute : FactAttribute
{
    public RequiresCheckpointAndPictureFixturesAttribute() =>
        Skip = Checkpoint.Env.NoCheckpointReason
            ?? Fixtures.Missing(Fixtures.Pictures.SelectMany(p => new[] { $"pictures/{p}.png", Fixtures.Small(p), Fixtures.Big(p) }).ToArray());
}

/// <summary>The tests that load a real model (gigabytes of weights and activations) share this collection, so they run one at a time.</summary>
public static class Heavy
{
    public const string Name = "real model";
}

[CollectionDefinition(Heavy.Name)]
public sealed class HeavyCollection { }

/// <summary>The parity criterion of the spec (<c>max_abs &lt;= 1e-4</c>, <c>cos_sim &gt;= 0.9999</c>) and the one for values that are only diagnostics.</summary>
public static class Criterion
{
    /// <summary>
    /// A stage output must be within 1e-4 and have cos_sim of at least 0.9999. For an intermediate value of a large scale (<paramref name="scale"/> is the largest
    /// magnitude in the reference; the Beta parameters of a metallic head are in the thousands) the error is held to 1e-5 of that scale instead, since fp32 has
    /// 1.2e-7 of it. Two all-zero tensors have no cosine (NaN): they agree when their difference is 0.
    /// </summary>
    public static bool Passes(MusicGen.Core.Testing.Parity.Result r, double scale = 0) =>
        r.MaxAbs <= (scale > 0 ? 1e-5 * scale : 1e-4) && (r.CosSim >= 0.9999 || (double.IsNaN(r.CosSim) && r.MaxAbs == 0));
}
