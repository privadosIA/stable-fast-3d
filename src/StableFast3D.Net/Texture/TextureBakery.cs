using StableFast3D.Field;
using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StableFast3D.Texture;

/// <summary>The two maps of a bake before they are made textures, (size, size, 3) float32, and which pixels a triangle covers (size, size).</summary>
public sealed record BakedMaps(Tensor Albedo, Tensor Normal, Tensor Mask);

/// <summary>What the model leaves with as textures: two 8-bit RGB images, row by row, and the two scalar material values of the picture.</summary>
/// <param name="Size">the side in pixels (512, 1024 or 2048)</param>
/// <param name="AlbedoRgb">the base colour, (size x size x 3) bytes</param>
/// <param name="NormalRgb">the tangent-space normal map, (size x size x 3) bytes</param>
public sealed record BakedTextures(int Size, byte[] AlbedoRgb, byte[] NormalRgb, float Roughness, float Metallic);

/// <summary>
/// The bake of <c>SF3D.generate_mesh</c>: for every pixel of the texture the triangle that covers it (<see cref="TextureBaker"/>), the point of the object it shows, the colour and the
/// perturbed normal the decoder says there, and the normal in the tangent space of the surface; then the filling of the borders and the quantization to 8 bits.
/// </summary>
public static class TextureBakery
{
    /// <summary>The numpy seed of the dither. The container does not seed it (its textures differ in their last bit from one run to the next); the port does, so a picture always gives the same model.</summary>
    public const uint DitherSeed = 0;

    /// <summary>The maps, before padding: the baked albedo and tangent-space normal, and the mask.</summary>
    /// <param name="sceneCodes">the triplane of the scene, (3, C, H, W)</param>
    public static BakedMaps Render(UnwrappedMesh mesh, Tensor sceneCodes, MaterialMlp decoder, int size, double radius)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();                  // the baked points are about a million at 2048: free what the maps do not keep
        var uv = mesh.Uv.cpu().contiguous().data<float>().ToArray();
        var faces = mesh.Triangles.cpu().contiguous().to(ScalarType.Int32).data<int>().ToArray();
        var rast = TextureBaker.Rasterize(uv, faces, size);
        var covered = TextureBaker.Mask(rast);

        var device = sceneCodes.device;
        var maskMap = tensor(covered, [size, size], device: device);
        var rows = maskMap.reshape(-1).nonzero().squeeze(1);                         // the covered pixels, row by row

        Tensor Interpolated(Tensor attribute) =>
            tensor(TextureBaker.Interpolate(attribute.cpu().contiguous().data<float>().ToArray(), faces, rast), [size * size, 3], device: device).index_select(0, rows);

        var gbPos = Interpolated(mesh.Vertices);
        var query = TriplaneQuery.Query(gbPos, sceneCodes, radius);
        var decoded = decoder.Forward(query, exclude: ["density", "vertex_offset"]);

        var gbNrm = F.normalize(Interpolated(mesh.Normals), dim: -1);
        var normal = F.normalize(decoded["perturb_normal"], p: 2, dim: -1, eps: 1e-7);        // utils.normalize: eps of float32

        // the albedo goes where the triangles are
        var albedo = zeros([size * size, 3], device: device);
        albedo.index_copy_(0, rows, decoded["features"]);

        // the normal, from the object's space to the tangent space of the surface at each pixel
        var gbTng = F.normalize(Interpolated(mesh.Tangents), dim: -1);
        var gbBtng = F.normalize(linalg.cross(gbNrm, gbTng, dim: -1), dim: -1);
        var unit = F.normalize(normal, dim: -1);
        var tangentMatrix = stack([gbTng, gbBtng, gbNrm], dim: -1);                          // (M, 3, 3): the tangent, bitangent and normal as columns
        var inTangent = bmm(tangentMatrix.transpose(1, 2), unit.unsqueeze(-1)).squeeze(-1);
        inTangent = (inTangent * 0.5 + 0.5).clamp(0, 1);
        var bump = zeros([size * size, 3], device: device);
        bump.index_copy_(0, rows, inTangent);

        return new BakedMaps(albedo.reshape(size, size, 3).MoveToOuterDisposeScope(), bump.reshape(size, size, 3).MoveToOuterDisposeScope(), maskMap.MoveToOuterDisposeScope());
    }

    /// <summary><c>uv_padding</c>: fills the borders of the covered area by <c>size / 150</c> pixels.</summary>
    public static Tensor Pad(Tensor map, Tensor mask, int size) =>
        TextureFinish.DilateFill(map.permute(2, 0, 1).unsqueeze(0).contiguous(), mask.unsqueeze(0).unsqueeze(0), size / 150).squeeze(0).permute(1, 2, 0).contiguous();

    /// <summary>The padded albedo and normal as 8-bit images, the albedo dithered first and then the normal, from one generator (as in <c>generate_mesh</c>).</summary>
    public static (byte[] Albedo, byte[] Normal) Quantize(Tensor paddedAlbedo, Tensor paddedNormal, NumpyRandom random)
    {
        var albedo = TextureFinish.ToUint8(paddedAlbedo.cpu().contiguous().data<float>().ToArray(), 3, random);

        // the normal is dithered only where it is perfectly flat, (0.5, 0.5, 1)
        var bump = paddedNormal.cpu().contiguous().data<float>().ToArray();
        var flat = new float[bump.Length / 3];
        for (var p = 0; p < flat.Length; p++) flat[p] = bump[p * 3] == 0.5f && bump[p * 3 + 1] == 0.5f && bump[p * 3 + 2] == 1.0f ? 1f : 0f;
        return (albedo, TextureFinish.ToUint8(bump, 3, random, flat));
    }

    /// <summary>The whole bake of a mesh at one size: the textures and the roughness and metallic given for the picture.</summary>
    public static BakedTextures Bake(UnwrappedMesh mesh, Tensor sceneCodes, MaterialMlp decoder, int size, double radius, float roughness, float metallic)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        var maps = Render(mesh, sceneCodes, decoder, size, radius);
        var albedo = Pad(maps.Albedo, maps.Mask, size);
        var normal = Pad(maps.Normal, maps.Mask, size);
        var (albedoRgb, normalRgb) = Quantize(albedo, normal, new NumpyRandom(DitherSeed));
        return new BakedTextures(size, albedoRgb, normalRgb, roughness, metallic);
    }
}
