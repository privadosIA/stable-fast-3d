using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D;

/// <summary>
/// A set of modules registered under names, as <c>nn.ModuleDict</c> does for the keys of a checkpoint (<c>heads.density.0.weight</c>), without TorchSharp's <c>ModuleDict</c>, whose
/// <c>to(device)</c> fails with a duplicate key when the model is moved to another device (found moving the decoder and the CLIP estimator to the card).
/// </summary>
public sealed class NamedModules<T> : nn.Module where T : nn.Module
{
    private readonly Dictionary<string, T> _items = new();

    public NamedModules(IEnumerable<(string Name, T Module)> items) : base(nameof(NamedModules<T>))
    {
        foreach (var (name, module) in items)
        {
            _items[name] = module;
            register_module(name, module);
        }
    }

    public T this[string name] => _items[name];
}
