using MusicGen.Core.Weights;
using StableFast3D.Camera;
using StableFast3D.Field;
using StableFast3D.Transformer;
using StableFast3D.Triplane;
using StableFast3D.Vision;
using StableFast3D.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D;

/// <summary>
/// The network of Stable Fast 3D, with its modules under the names of the top-level keys of the checkpoint (<c>camera_embedder</c>, <c>image_tokenizer</c>, <c>tokenizer</c>,
/// <c>backbone</c>, <c>post_processor</c>, <c>decoder</c>, <c>image_estimator</c>, <c>bbox</c>), so that the whole checkpoint loads with one strict load: every key is used once, the only
/// ones left over being the CLIP text tower and the global estimator (<see cref="Sf3dCheckpoint.IsUnused"/>).
/// </summary>
public sealed class Sf3dModel : nn.Module
{
    public readonly LinearCameraEmbedder camera_embedder;
    public readonly Dinov2ImageTokenizer image_tokenizer;
    public readonly TriplaneTokenizer tokenizer;
    public readonly TwoStreamInterleaveTransformer backbone;
    public readonly PixelShuffleUpsampleNetwork post_processor;
    public readonly MaterialMlp decoder;
    public readonly ClipHeadEstimator image_estimator;

    public Sf3dModel(Dinov2Config dinov2, double radius) : base(nameof(Sf3dModel))
    {
        camera_embedder = new LinearCameraEmbedder();
        image_tokenizer = new Dinov2ImageTokenizer(dinov2);
        tokenizer = new TriplaneTokenizer();
        backbone = new TwoStreamInterleaveTransformer(new BackboneConfig());
        post_processor = new PixelShuffleUpsampleNetwork();
        decoder = new MaterialMlp();
        image_estimator = new ClipHeadEstimator();
        RegisterComponents();
        register_buffer("bbox", tensor(new float[] { (float)-radius, (float)-radius, (float)-radius, (float)radius, (float)radius, (float)radius }, [2, 3]));
    }

    /// <summary>The cube of the scene as the checkpoint holds it, (2, 3): minimum and maximum corners.</summary>
    public Tensor Bbox => get_buffer("bbox")!;

    /// <summary>Builds the network and loads the whole checkpoint into it, strictly, onto <paramref name="device"/>; the model is left in inference mode.</summary>
    /// <exception cref="InvalidDataException">The checkpoint does not match the network, or its cube is not the radius of the config.</exception>
    public static Sf3dModel Load(Sf3dCheckpoint checkpoint, Sf3dConfig config, Device device)
    {
        var model = new Sf3dModel(checkpoint.Dinov2, config.Radius);
        StrictLoader.Load(model, checkpoint.Index, device, Sf3dCheckpoint.LoadOptions());
        model.eval();
        using var bbox = model.Bbox.cpu();
        var expected = (float)config.Radius;
        if (bbox.data<float>().ToArray().Zip(new[] { -expected, -expected, -expected, expected, expected, expected }, (a, b) => Math.Abs(a - b)).Max() > 1e-6f)
            throw new InvalidDataException($"The checkpoint's cube ({string.Join(", ", bbox.data<float>().ToArray())}) is not the radius {config.Radius} of sf3d/config.yaml");
        return model;
    }

    /// <summary>
    /// <c>SF3D.get_scene_codes</c> for one picture: the camera embedding, the picture's tokens modulated by it, the triplane tokens refined by the backbone against them, and the
    /// upsampled triplanes.
    /// </summary>
    /// <param name="rgb">the picture the model sees, (H, W, 3), 0..1</param>
    /// <returns>(3, 40, 384, 384)</returns>
    public Tensor SceneCodes(Tensor rgb, Sf3dConfig config)
    {
        using var _ = no_grad();
        var device = rgb.device;
        using var c2w = DefaultCamera.CameraToWorld(config.DefaultDistance).to(device);
        var (intrinsic, normed) = DefaultCamera.Intrinsics(config.DefaultFovyDeg, config.CondImageSize, config.CondImageSize);
        using var __ = intrinsic;
        using var normedDevice = normed.to(device);
        normed.Dispose();

        var imageTokens = Frugal.Run(() =>
        {
            var cameraEmbeds = camera_embedder.call(c2w.reshape(1, 1, 4, 4), normedDevice.reshape(1, 1, 3, 3));
            var images = rgb.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3).contiguous();          // (B, Nv, C, H, W)
            var tokensOfPicture = image_tokenizer.call(images, cameraEmbeds);                          // (B, Nv, C, Nt)
            return tokensOfPicture.reshape(1, tokensOfPicture.shape[2], tokensOfPicture.shape[3]).permute(0, 2, 1).contiguous();   // (B, Nv x Nt, C)
        });
        var tokens = Frugal.Run(() => backbone.call(tokenizer.Tokens(1), imageTokens));
        return Frugal.Run(() => post_processor.call(tokenizer.Detokenize(tokens))[0].contiguous());
    }
}
