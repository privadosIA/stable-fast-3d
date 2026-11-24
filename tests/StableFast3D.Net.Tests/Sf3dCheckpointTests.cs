using MusicGen.Core.Weights;
using StableFast3D.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Net.Tests;

/// <summary>The checkpoint is read strictly: a missing or left-over key is an error that names it, and only the unused CLIP text tower may be left over.</summary>
public class Sf3dCheckpointTests
{
    /// <summary>A model with the names of three of the checkpoint's modules, small enough to write a checkpoint for it in a test.</summary>
    private sealed class Tiny : nn.Module
    {
        public readonly nn.Module<Tensor, Tensor> camera_embedder = nn.Linear(25, 4);
        public readonly nn.Module<Tensor, Tensor> decoder = nn.Linear(4, 2);

        public Tiny() : base("tiny")
        {
            RegisterComponents();
            register_buffer("bbox", tensor(new float[] { -1, -1, -1, 1, 1, 1 }));
        }
    }

    // the model names the layers' parameters camera_embedder.weight, camera_embedder.bias, decoder.weight, decoder.bias and the buffer bbox
    private static Dictionary<string, Tensor> ValidTensors() => new()
    {
        ["camera_embedder.weight"] = ones(4, 25),
        ["camera_embedder.bias"] = full(new long[] { 4 }, 0.5f),
        ["decoder.weight"] = full(new long[] { 2, 4 }, 2f),
        ["decoder.bias"] = zeros(2),
        ["bbox"] = tensor(new float[] { -0.87f, -0.87f, -0.87f, 0.87f, 0.87f, 0.87f }),
    };

    private static CheckpointIndex Write(TempDir dir, Dictionary<string, Tensor> tensors)
    {
        SafeTensorsWriter.Save(dir.File("model.safetensors"), tensors);
        return CheckpointIndex.Open(dir.File("model.safetensors"));
    }

    [Fact]
    public void ACheckpointThatFitsLoadsEveryTensorByItsOwnName()
    {
        using var dir = new TempDir();
        var model = new Tiny();
        var result = StrictLoader.Load(model, Write(dir, ValidTensors()), CPU, Sf3dCheckpoint.LoadOptions());
        Assert.Equal(5, result.TensorsRead);
        Assert.Equal(0.5f, model.get_parameter("camera_embedder.bias")!.cpu().data<float>()[0]);
        Assert.Equal(0.87f, model.get_buffer("bbox")!.cpu().data<float>()[5]);
    }

    [Fact]
    public void AKeyTheCheckpointLacksIsNamed()
    {
        using var dir = new TempDir();
        var tensors = ValidTensors();
        tensors.Remove("decoder.bias");
        var ex = Assert.Throws<InvalidDataException>(() => StrictLoader.Load(new Tiny(), Write(dir, tensors), CPU, Sf3dCheckpoint.LoadOptions()));
        Assert.StartsWith("The Stable Fast 3D checkpoint does not match the model:", ex.Message);
        Assert.Contains("1 model keys are missing from the checkpoint: decoder.bias", ex.Message);
    }

    [Fact]
    public void AKeyTheModelLacksIsNamed()
    {
        using var dir = new TempDir();
        var tensors = ValidTensors();
        tensors["backbone.proj_out.weight"] = zeros(2, 2);
        var ex = Assert.Throws<InvalidDataException>(() => StrictLoader.Load(new Tiny(), Write(dir, tensors), CPU, Sf3dCheckpoint.LoadOptions()));
        Assert.Contains("1 checkpoint keys are not in the model: backbone.proj_out.weight", ex.Message);
    }

    [Fact]
    public void AShapeThatDiffersIsNamed()
    {
        using var dir = new TempDir();
        var tensors = ValidTensors();
        tensors["camera_embedder.weight"] = ones(4, 24);
        var ex = Assert.Throws<InvalidDataException>(() => StrictLoader.Load(new Tiny(), Write(dir, tensors), CPU, Sf3dCheckpoint.LoadOptions()));
        Assert.Contains("camera_embedder.weight: the model has [4, 25], the checkpoint [4, 24]", ex.Message);
    }

    [Fact]
    public void NothingIsModifiedWhenTheCheckpointDoesNotFit()
    {
        using var dir = new TempDir();
        var model = new Tiny();
        var before = model.get_parameter("decoder.weight")!.clone();
        var tensors = ValidTensors();
        tensors.Remove("bbox");
        Assert.Throws<InvalidDataException>(() => StrictLoader.Load(model, Write(dir, tensors), CPU, Sf3dCheckpoint.LoadOptions()));
        Assert.True(model.get_parameter("decoder.weight")!.Equals(before));
    }

    [Fact]
    public void OnlyTheClipTextTowerAndTheGlobalEstimatorMayBeLeftOver()
    {
        using var dir = new TempDir();
        var tensors = ValidTensors();
        tensors["image_estimator.model.token_embedding.weight"] = zeros(3, 2);
        tensors["image_estimator.model.transformer.resblocks.0.ln_1.bias"] = zeros(2);
        tensors["image_estimator.model.logit_scale"] = zeros(1);
        tensors["global_estimator.heads.sg_amplitudes.0.0.weight"] = zeros(2, 2);
        var result = StrictLoader.Load(new Tiny(), Write(dir, tensors), CPU, Sf3dCheckpoint.LoadOptions());
        Assert.Equal(4, result.SkippedByAllowList);
        Assert.Equal(5, result.TensorsRead);

        // the vision tower is not allowed to be left over: the port has to carry it
        using var dir2 = new TempDir();
        var withVisual = ValidTensors();
        withVisual["image_estimator.model.visual.proj"] = zeros(2, 2);
        var ex = Assert.Throws<InvalidDataException>(() => StrictLoader.Load(new Tiny(), Write(dir2, withVisual), CPU, Sf3dCheckpoint.LoadOptions()));
        Assert.Contains("image_estimator.model.visual.proj", ex.Message);
    }

    [Theory]
    [InlineData("global_estimator.layers.0.weight", true)]
    [InlineData("global_estimator.heads.sg_amplitudes.0.0.weight", true)]
    [InlineData("image_estimator.heads.roughness.0.0.weight", false)]
    public void TheGlobalEstimatorIsToldApart(string key, bool isGlobal) => Assert.Equal(isGlobal, Sf3dCheckpoint.IsGlobalEstimator(key));

    [Theory]
    [InlineData("image_estimator.model.token_embedding.weight", true)]
    [InlineData("image_estimator.model.transformer.resblocks.11.mlp.c_fc.weight", true)]
    [InlineData("image_estimator.model.text_projection", true)]
    [InlineData("image_estimator.model.visual.transformer.resblocks.0.ln_1.weight", false)]
    [InlineData("image_estimator.model.visual.proj", false)]
    [InlineData("image_estimator.heads.roughness.0.0.weight", false)]
    [InlineData("image_tokenizer.model.encoder.layer.0.norm1.weight", false)]
    [InlineData("transformer.resblocks.0.ln_1.weight", false)]
    public void TheTextTowerIsToldApart(string key, bool isText) => Assert.Equal(isText, Sf3dCheckpoint.IsClipTextTower(key));

    [Fact]
    public void OpeningNamesEveryRequiredFileThatIsMissing()
    {
        using var dir = new TempDir();
        Directory.CreateDirectory(Path.Combine(dir.Path, "sf3d"));
        File.WriteAllBytes(Path.Combine(dir.Path, "sf3d", "config.yaml"), []);
        var ex = Assert.Throws<FileNotFoundException>(() => Sf3dCheckpoint.Open(dir.Path));
        Assert.Contains("sf3d/model.safetensors", ex.Message);
        Assert.Contains("sf3d/160_tets.npz", ex.Message);
        Assert.Contains("rembg/u2net.onnx", ex.Message);
        Assert.DoesNotContain("sf3d/config.yaml", ex.Message);
    }

    [Fact]
    public void TheTetrahedraGridIsOneOfTheRequiredFiles() => Assert.Contains(Sf3dCheckpoint.TetsFile, EngineInfo.RequiredFiles);

    // ---- the real checkpoint (3.8 GB: only its header is read) ------------------------------------------------------------------------------

    [RequiresCheckpoint]
    public void TheRealCheckpointHasTheTensorsTheModelsAreMadeOf()
    {
        var w = Checkpoint.Weights;
        var t = w.Index.Tensors;
        Assert.Equal(1276, t.Count);
        Assert.All(t.Values, i => Assert.Equal("F32", i.Dtype));
        Assert.Equal(1_006_027_437L, t.Values.Sum(i => i.Shape.Aggregate(1L, (a, d) => a * d)));

        // every key belongs to a top-level module the port knows
        var tops = t.Keys.Select(k => k.Split('.')[0]).Distinct().Order(StringComparer.Ordinal).ToList();
        Assert.Equal(Sf3dCheckpoint.TopLevelModules, tops);

        Assert.Equal(new long[] { 2, 3 }, t["bbox"].Shape);
        Assert.Equal(new long[] { 768, 25 }, t["camera_embedder.linear.weight"].Shape);
        Assert.Equal(new long[] { 3, 1024, 96, 96 }, t["tokenizer.embeddings"].Shape);
        Assert.Equal(new long[] { 1, 1792, 1024 }, t["backbone.latent_init"].Shape);
        Assert.Equal(new long[] { 64, 120 }, t["decoder.heads.density.0.weight"].Shape);
    }

    [RequiresCheckpoint]
    public void TheDinov2ConfigAgreesWithTheCheckpoint()
    {
        var w = Checkpoint.Weights;
        var c = w.Dinov2;
        var t = w.Index.Tensors;
        const string p = "image_tokenizer.model.";
        Assert.Equal(new long[] { 1, c.NumPositions, c.HiddenSize }, t[p + "embeddings.position_embeddings"].Shape);
        Assert.Equal(new long[] { c.HiddenSize, c.NumChannels, c.PatchSize, c.PatchSize }, t[p + "embeddings.patch_embeddings.projection.weight"].Shape);
        Assert.Equal(new long[] { c.IntermediateSize, c.HiddenSize }, t[p + "encoder.layer.0.mlp.fc1.weight"].Shape);
        var layers = t.Keys.Where(k => k.StartsWith(p + "encoder.layer.", StringComparison.Ordinal)).Select(k => k.Split('.')[4]).Distinct().Count();
        Assert.Equal(c.NumHiddenLayers, layers);
        // the camera modulation of the port: two per layer, in the width the camera embedding has
        Assert.Equal(2 * c.NumHiddenLayers * 2, t.Keys.Count(k => k.Contains("_modulation.")));
        Assert.Equal(new long[] { 2 * c.HiddenSize, 768 }, t[p + "encoder.layer.23.norm2_modulation.linear2.weight"].Shape);
    }

    [RequiresCheckpoint]
    public void TheUnusedKeysAreTheClipTextTowerAndTheGlobalEstimator()
    {
        var t = Checkpoint.Weights.Index.Tensors;
        Assert.Equal(150, t.Keys.Count(Sf3dCheckpoint.IsClipTextTower));
        Assert.Equal(12, t.Keys.Count(Sf3dCheckpoint.IsGlobalEstimator));
        Assert.Equal(162, t.Keys.Count(Sf3dCheckpoint.IsUnused));
        Assert.Equal(152, t.Keys.Count(k => k.StartsWith("image_estimator.model.visual.", StringComparison.Ordinal)));
        Assert.Equal(new long[] { 768, 3, 32, 32 }, t["image_estimator.model.visual.conv1.weight"].Shape);   // ViT-B/32
        Assert.Equal(new long[] { 768, 512 }, t["image_estimator.model.visual.proj"].Shape);
    }

    [RequiresCheckpoint]
    public void TheWeightsFolderOpens()
    {
        var w = Checkpoint.Weights;
        Assert.True(File.Exists(w.TetsPath));
        Assert.True(new FileInfo(w.TetsPath).Length > 1_000_000);
    }
}
