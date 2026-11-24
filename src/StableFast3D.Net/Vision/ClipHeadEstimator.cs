using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Vision;

/// <summary>What one head of the estimator says about a picture: the two parameters of its Beta distribution and its mode (the value used).</summary>
/// <param name="Alpha"><c>concentration1</c>, (B)</param>
/// <param name="Beta"><c>concentration0</c>, (B)</param>
/// <param name="Value">the mode, (B, 1, 1)</param>
public sealed record BetaHead(Tensor Alpha, Tensor Beta, Tensor Value);

/// <param name="Features">CLIP's feature of the picture, (B, 512)</param>
public sealed record ClipEstimate(Tensor Features, IReadOnlyDictionary<string, BetaHead> Heads);

/// <summary>
/// <c>ClipBasedHeadEstimator</c> as the container runs it (<c>distribution: beta</c>, <c>distribution_eval: mode</c>): the picture is resized to 224 x 224 (the tower's image size) and normalised
/// with OpenAI's mean and deviation, CLIP's vision tower gives 512 numbers, and one head per material value (roughness, metallic) turns them into the two
/// parameters of a Beta distribution whose mode is that value. The weights are <c>image_estimator.{model.visual,heads}.*</c> of the checkpoint.
/// </summary>
public sealed class ClipHeadEstimator : nn.Module
{
    private static readonly float[] Mean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] Std = [0.26862954f, 0.26130258f, 0.27577711f];

    public const int HiddenFeatures = 512;
    public const int HiddenLayers = 3;
    /// <summary>Added to the heads' outputs before the softplus (<c>output_bias</c> of both heads in the config).</summary>
    public const double OutputBias = 1.0;

    private readonly string[] _headNames;
    private readonly int _size;

    public readonly ClipModel model;
    public readonly ModuleDict<ModuleList<nn.Module<Tensor, Tensor>>> heads;

    public ClipHeadEstimator(ClipVisionConfig? clip = null, IEnumerable<string>? headNames = null) : base(nameof(ClipHeadEstimator))
    {
        _headNames = (headNames ?? ["roughness", "metallic"]).ToArray();
        clip ??= new ClipVisionConfig();
        _size = clip.ImageSize;
        model = new ClipModel(clip);
        heads = nn.ModuleDict(_headNames.Select(name => (name, nn.ModuleList(Shared(), Branch(), Branch()))).ToArray());
        RegisterComponents();
    }

    // [Linear, ReLU] x HiddenLayers
    private static nn.Module<Tensor, Tensor> Shared()
    {
        var layers = new List<nn.Module<Tensor, Tensor>>();
        for (var i = 0; i < HiddenLayers; i++)
        {
            layers.Add(nn.Linear(HiddenFeatures, HiddenFeatures));
            layers.Add(nn.ReLU());
        }
        return nn.Sequential(layers.ToArray());
    }

    // Linear, ReLU, Linear to one number: the first and the second parameter of the Beta
    private static nn.Module<Tensor, Tensor> Branch() => nn.Sequential(nn.Linear(HiddenFeatures, HiddenFeatures), nn.ReLU(), nn.Linear(HiddenFeatures, 1));

    /// <param name="condImage">(B, 1, H, W, 3), the picture times its mask (what <c>generate_mesh</c> gives the estimator)</param>
    public ClipEstimate Estimate(Tensor condImage)
    {
        var image = condImage.flatten(0, 1).permute(0, 3, 1, 2).contiguous();
        image = F.interpolate(image, size: [_size, _size], mode: InterpolationMode.Bilinear, align_corners: false);
        using var mean = tensor(Mean, [1, 3, 1, 1], device: image.device);
        using var std = tensor(Std, [1, 3, 1, 1], device: image.device);
        var features = model.visual.call((image - mean) / std);

        var result = new Dictionary<string, BetaHead>();
        foreach (var name in _headNames)
        {
            var head = heads[name];
            var shared = head[0].call(features);
            var d1 = head[1].call(shared).squeeze(-1);
            var d2 = head[2].call(shared).squeeze(-1);
            var alpha = F.softplus(d1 + OutputBias);
            var beta = F.softplus(d2 + OutputBias);
            result[name] = new BetaHead(alpha, beta, BetaMode(alpha, beta).reshape(-1, 1, 1));
        }
        return new ClipEstimate(features, result);
    }

    /// <summary>
    /// The mode of a Beta distribution as torch computes it (through its Dirichlet): <c>(a - 1)+ / ((a - 1)+ + (b - 1)+)</c>, and 1 when both
    /// concentrations are below 1 (torch takes the arg-max of two NaNs, which is the first).
    /// </summary>
    public static Tensor BetaMode(Tensor alpha, Tensor beta)
    {
        var a = (alpha - 1).clamp_min(0);
        var b = (beta - 1).clamp_min(0);
        var mode = a / (a + b);
        return where((alpha < 1) & (beta < 1), ones_like(mode), mode);
    }
}
