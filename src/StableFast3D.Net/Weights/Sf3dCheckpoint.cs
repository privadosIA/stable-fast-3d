using MusicGen.Core.Weights;

namespace StableFast3D.Weights;

/// <summary>
/// The weights folder of the engine, laid out as the Python container's <c>/ckpt</c> (<see cref="EngineInfo.RequiredFiles"/>), read without Python and
/// without downloading anything: the index of <c>sf3d/model.safetensors</c> (headers only), the DINOv2 config and the path of the tetrahedra grid.
/// </summary>
public sealed class Sf3dCheckpoint
{
    public const string ModelFile = "sf3d/model.safetensors";
    public const string Dinov2ConfigFile = "sf3d/dinov2-large-config.json";
    /// <summary>The marching-tetrahedra grid of the upstream repository (<c>load/tets/160_tets.npz</c>), kept next to the weights: it is not one of them.</summary>
    public const string TetsFile = "sf3d/160_tets.npz";

    /// <summary>The top-level modules of the checkpoint (the first segment of every key) and nothing else.</summary>
    public static readonly IReadOnlyList<string> TopLevelModules = new[]
    {
        "backbone", "bbox", "camera_embedder", "decoder", "global_estimator", "image_estimator", "image_tokenizer", "post_processor", "tokenizer"
    };

    private Sf3dCheckpoint(string directory, CheckpointIndex index, Dinov2Config dinov2)
    {
        Directory = directory;
        Index = index;
        Dinov2 = dinov2;
    }

    public string Directory { get; }
    public CheckpointIndex Index { get; }
    public Dinov2Config Dinov2 { get; }
    public string TetsPath => System.IO.Path.Combine(Directory, TetsFile);

    /// <summary>
    /// Opens the weights folder. A file of <see cref="EngineInfo.RequiredFiles"/> that is not there is one <see cref="FileNotFoundException"/> that names all
    /// of them (the service says the same in <c>missing</c>); the checkpoint itself is only indexed, never read.
    /// </summary>
    public static Sf3dCheckpoint Open(string directory)
    {
        var missing = EngineInfo.RequiredFiles.Where(f => !File.Exists(System.IO.Path.Combine(directory, f))).ToList();
        if (missing.Count > 0)
            throw new FileNotFoundException($"The weights folder {directory} lacks: {string.Join(", ", missing)}");
        var index = CheckpointIndex.Open(System.IO.Path.Combine(directory, ModelFile));
        var dinov2 = Dinov2Config.FromFile(System.IO.Path.Combine(directory, Dinov2ConfigFile));
        return new Sf3dCheckpoint(directory, index, dinov2);
    }

    /// <summary>
    /// The CLIP text tower. The checkpoint holds all of <c>ViT-B/32</c> inside <c>image_estimator.model</c> (the Python code builds the whole of it), but
    /// the roughness and metallic heads only read the vision tower (<c>image_estimator.model.visual</c>): the 150 tensors of the text tower are never used.
    /// </summary>
    public static bool IsClipTextTower(string key) =>
        key.StartsWith("image_estimator.model.", StringComparison.Ordinal)
        && !key.StartsWith("image_estimator.model.visual.", StringComparison.Ordinal);

    /// <summary>
    /// The global illumination estimator (12 tensors): not ported, because the container runs with <c>estimate_illumination</c> off, but the checkpoint
    /// holds its weights.
    /// </summary>
    public static bool IsGlobalEstimator(string key) => key.StartsWith("global_estimator.", StringComparison.Ordinal);

    /// <summary>The keys of the checkpoint the port never reads (the only ones allowed to be left over): the CLIP text tower and the global estimator.</summary>
    public static bool IsUnused(string key) => IsClipTextTower(key) || IsGlobalEstimator(key);

    /// <summary>
    /// How the checkpoint maps onto the port's modules: every key by its own name (the C# modules carry the checkpoint's names, so there is no mapping),
    /// the persistent buffers included (<c>bbox</c>), only <see cref="IsUnused"/> keys allowed to be left over, and fp32 as it is in the file.
    /// </summary>
    public static StrictLoadOptions LoadOptions(string errorPrefix = "The Stable Fast 3D checkpoint does not match the model:") =>
        new(errorPrefix)
        {
            Skip = IsUnused,
            IncludeBuffers = true,
            UnexpectedLabel = "checkpoint keys"
        };
}
