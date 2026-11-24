using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Field;

/// <summary>One head of the decoder: how deep it is, how wide its output and what is done to it (<c>decoder.heads</c> of <c>sf3d/config.yaml</c>).</summary>
/// <param name="OutBias">Added to the head's output before the activation (-1.0 for the density).</param>
/// <param name="Activation"><c>trunc_exp</c>, <c>sigmoid</c>, <c>normalize_channel_last</c> or null (none).</param>
public sealed record HeadSpec(string Name, int OutChannels, int HiddenLayers, string? Activation = null, double OutBias = 0.0);

public sealed record MaterialMlpConfig
{
    public int InChannels { get; init; } = 120;
    public int Neurons { get; init; } = 64;

    /// <summary>The four heads of Stable Fast 3D's decoder, in the order of the config.</summary>
    public IReadOnlyList<HeadSpec> Heads { get; init; } =
    [
        new("density", 1, 2, "trunc_exp", -1.0),
        new("features", 3, 3, "sigmoid"),
        new("perturb_normal", 3, 3, "normalize_channel_last"),
        new("vertex_offset", 3, 2),
    ];
}

/// <summary>
/// <c>MaterialMLP</c>: the 120 values the triplane gives at a point (3 planes x 40 channels) through one small MLP per property (SiLU between 64-wide layers):
/// the density, the colour (<c>features</c>), the perturbed normal and the offset of the vertex. The weights are <c>decoder.heads.&lt;name&gt;.{0,2,4,...}</c>.
/// </summary>
public sealed class MaterialMlp : nn.Module
{
    // torch.nn.functional.normalize's eps for float32 (sf3d.models.utils.EPS_DTYPE)
    private const double NormalizeEps = 1e-7;

    private readonly MaterialMlpConfig _config;

    public readonly ModuleDict<nn.Module<Tensor, Tensor>> heads;

    public MaterialMlp(MaterialMlpConfig? config = null) : base(nameof(MaterialMlp))
    {
        _config = config ?? new MaterialMlpConfig();
        var entries = new List<(string, nn.Module<Tensor, Tensor>)>();
        foreach (var head in _config.Heads)
        {
            var layers = new List<nn.Module<Tensor, Tensor>>();
            for (var i = 0; i < head.HiddenLayers; i++)
            {
                layers.Add(nn.Linear(i == 0 ? _config.InChannels : _config.Neurons, _config.Neurons));
                layers.Add(nn.SiLU());
            }
            layers.Add(nn.Linear(_config.Neurons, head.OutChannels));
            entries.Add((head.Name, nn.Sequential(layers.ToArray())));
        }
        heads = nn.ModuleDict(entries.ToArray());
        RegisterComponents();
    }

    /// <summary>Every head (or only <paramref name="include"/>, or all but <paramref name="exclude"/>) over (N, 120) values: name to (N, channels).</summary>
    public Dictionary<string, Tensor> Forward(Tensor x, IReadOnlyCollection<string>? include = null, IReadOnlyCollection<string>? exclude = null)
    {
        if (include is not null && exclude is not null) throw new ArgumentException("Cannot specify both include and exclude.");
        var out_ = new Dictionary<string, Tensor>();
        foreach (var head in _config.Heads)
        {
            if (include is not null && !include.Contains(head.Name)) continue;
            if (exclude is not null && exclude.Contains(head.Name)) continue;
            out_[head.Name] = Activate(head.Activation, heads[head.Name].call(x) + head.OutBias);
        }
        return out_;
    }

    private static Tensor Activate(string? name, Tensor x) => name switch
    {
        null or "none" or "linear" or "identity" => x,
        "trunc_exp" => exp(x),                       // truncated only in the backward pass
        "sigmoid" => sigmoid(x),
        "normalize_channel_last" => F.normalize(x, p: 2, dim: -1, eps: NormalizeEps),
        _ => throw new NotSupportedException($"the activation {name} is not one of the decoder's"),
    };
}
