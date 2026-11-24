using System.Globalization;
using System.Text.RegularExpressions;

namespace StableFast3D;

/// <summary>
/// The scalars of <c>sf3d/config.yaml</c> (with the defaults of upstream's <c>SF3D.Config</c> dataclass for what the file does not say) that the port reads:
/// the picture size the model sees, the camera, the radius of the scene and the isosurface. The rest of the file (the modules' sizes) is fixed in the C#
/// modules, which the strict loader checks against the checkpoint.
/// </summary>
public sealed record Sf3dConfig
{
    /// <summary>The side in pixels of the picture the model sees (512).</summary>
    public int CondImageSize { get; init; } = 512;
    public int IsosurfaceResolution { get; init; } = 160;
    public float IsosurfaceThreshold { get; init; } = 10.0f;
    public double Radius { get; init; } = 0.87;
    /// <summary>What the object is composited over, per channel (grey).</summary>
    public float[] BackgroundColor { get; init; } = [0.5f, 0.5f, 0.5f];
    public double DefaultFovyDeg { get; init; } = 40.0;
    public float DefaultDistance { get; init; } = 1.6f;

    private static readonly Regex Scalar = new(@"^(?<key>[a-z_]+):\s*(?<value>-?[0-9.]+)\s*(#.*)?$", RegexOptions.Compiled);

    /// <summary>Reads the top-level numbers of the YAML file this config knows; anything else is left to the C# modules. A key it knows with a value it cannot read is an error.</summary>
    public static Sf3dConfig FromFile(string path)
    {
        var c = new Sf3dConfig();
        foreach (var line in File.ReadLines(path))
        {
            var m = Scalar.Match(line);
            if (!m.Success) continue;
            var v = m.Groups["value"].Value;
            try
            {
                c = m.Groups["key"].Value switch
                {
                    "cond_image_size" => c with { CondImageSize = int.Parse(v, CultureInfo.InvariantCulture) },
                    "isosurface_resolution" => c with { IsosurfaceResolution = int.Parse(v, CultureInfo.InvariantCulture) },
                    "isosurface_threshold" => c with { IsosurfaceThreshold = float.Parse(v, CultureInfo.InvariantCulture) },
                    "radius" => c with { Radius = double.Parse(v, CultureInfo.InvariantCulture) },
                    "default_fovy_deg" => c with { DefaultFovyDeg = double.Parse(v, CultureInfo.InvariantCulture) },
                    "default_distance" => c with { DefaultDistance = float.Parse(v, CultureInfo.InvariantCulture) },
                    _ => c,
                };
            }
            catch (FormatException)
            {
                throw new InvalidDataException($"{path}: cannot read the value of {m.Groups["key"].Value}: {v}");
            }
        }
        return c;
    }
}
