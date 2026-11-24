using StableFast3D.Camera;
using StableFast3D.Mesh;
using StableFast3D.Picture;
using StableFast3D.Texture;
using StableFast3D.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D;

/// <summary>A model made from a picture: the binary glTF and what is said of it.</summary>
/// <param name="Glb">The <c>.glb</c>: Y up, origin at the middle of its base, its colours in two textures and the roughness and metallic of the picture.</param>
/// <param name="Triangles">The triangles of the mesh.</param>
/// <param name="Roughness">the picture's roughness, as written in the material</param>
/// <param name="Metallic">the picture's metallic, as written in the material</param>
public sealed record GeneratedModel(byte[] Glb, int Triangles, float Roughness, float Metallic);

/// <summary>
/// The whole of Stable Fast 3D, picture in and model out: the cut-out, the network, the surface (marching tetrahedra), the UV unwrapping, the bake of the textures and the file.
/// It loads the weights once (<see cref="Load"/>) and makes models one at a time. Deterministic: the same picture and bake size give the same model (the dither of the textures is
/// seeded, and the unwrapping runs on the CPU whatever the device is, so the card does not change the layout).
/// </summary>
public sealed class ModelGenerator : IDisposable
{
    private readonly Sf3dModel _model;
    private readonly Sf3dConfig _config;
    private readonly TetrahedraGrid _grid;
    private readonly U2NetCutout _cutout;
    private readonly TriplaneMesher _mesher;
    private readonly Device _device;

    private ModelGenerator(Sf3dModel model, Sf3dConfig config, TetrahedraGrid grid, U2NetCutout cutout, Device device)
    {
        (_model, _config, _grid, _cutout, _device) = (model, config, grid, cutout, device);
        _mesher = new TriplaneMesher(grid, model.decoder, config);
    }

    /// <summary>Opens the weights folder and loads everything onto the device (<c>cpu</c>, <c>cuda</c> or <c>cuda:N</c>).</summary>
    /// <exception cref="FileNotFoundException">A file of <see cref="EngineInfo.RequiredFiles"/> is not there (the message names them all).</exception>
    public static ModelGenerator Load(string weightsDirectory, string device = "cpu")
    {
        var checkpoint = Sf3dCheckpoint.Open(weightsDirectory);
        var config = Sf3dConfig.FromFile(checkpoint.ConfigPath);
        var dev = new Device(device);
        var model = Sf3dModel.Load(checkpoint, config, dev);
        var grid = new TetrahedraGrid(checkpoint.TetsPath, config.IsosurfaceResolution);
        var cutout = new U2NetCutout(checkpoint.CutoutModelPath);
        return new ModelGenerator(model, config, grid, cutout, dev);
    }

    /// <summary>The resolution of the surface in points a side (the tetrahedra grid's): the same for every detail.</summary>
    public int SurfaceResolution => _config.IsosurfaceResolution;

    /// <summary>Makes the model of a picture (png, jpg or webp) with textures of <paramref name="bakeSize"/> pixels.</summary>
    /// <exception cref="InvalidDataException">The bytes are not a picture.</exception>
    /// <exception cref="ModelRefusedException">The picture gave no surface.</exception>
    /// <exception cref="TooManyTrianglesException">The surface has more triangles than a scene takes.</exception>
    public GeneratedModel Generate(byte[] picture, int bakeSize)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();                       // every tensor of this model is freed when it is made, not whenever the GC gets to it

        var rgba = _cutout.Cutout(PictureDecoder.Decode(picture));
        var (rgb, mask) = ModelInput.Prepare(rgba, _config.CondImageSize, _config.BackgroundColor);
        rgb = rgb.to(_device);
        mask = mask.to(_device);

        var sceneCodes = _model.SceneCodes(rgb, _config);
        var estimate = _model.image_estimator.Estimate((rgb * mask).unsqueeze(0).unsqueeze(0));    // (B, Nv, H, W, 3): the picture times its mask
        var roughness = estimate.Heads["roughness"].Value.item<float>();
        var metallic = estimate.Heads["metallic"].Value.item<float>();

        var surface = _mesher.Extract(sceneCodes);
        var triangles = (int)ModelFinish.Check(surface);           // no surface is refused and too many triangles fail, before the textures are made

        // the unwrapping runs on the CPU, so its layout does not depend on the card
        var unwrapped = UnwrappedMesh.Unwrap(surface.Vertices.cpu(), surface.Triangles.cpu());
        var baked = TextureBakery.Bake(unwrapped, sceneCodes, _model.decoder, bakeSize, _config.Radius, roughness, metallic);
        var glb = GlbWriter.Write(ModelAssembly.Assemble(unwrapped), baked);
        return new GeneratedModel(glb, triangles, roughness, metallic);
    }

    public void Dispose()
    {
        _cutout.Dispose();
        _grid.Dispose();
        _model.Dispose();
    }
}
