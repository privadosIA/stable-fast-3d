using System.Text;
using System.Text.Json;
using SkiaSharp;
using StableFast3D.Texture;

namespace StableFast3D.Mesh;

/// <summary>
/// Writes the model as a binary glTF 2.0 with the structure <c>trimesh</c> gives it in the container: one mesh of one triangle primitive (indices, POSITION and TEXCOORD_0), one
/// material with a base colour texture, a normal texture and the roughness and metallic factors, the two pictures as JPEG inside the file, and a node called <c>world</c> that holds
/// the mesh's. Everything is in the file (no external address), which is what <c>blender.ParseModel</c> of Veritmo asks of a model.
/// </summary>
public static class GlbWriter
{
    /// <summary>The JPEG quality of the textures (Pillow's default, which is what the container's textures have).</summary>
    public const int JpegQuality = 75;

    public static byte[] Write(ModelMesh mesh, BakedTextures textures, int jpegQuality = JpegQuality)
    {
        var albedo = Jpeg(textures.AlbedoRgb, textures.Size, jpegQuality);
        var normal = Jpeg(textures.NormalRgb, textures.Size, jpegQuality);

        // the binary chunk, in the order trimesh lays it out: indices, positions, the two pictures, texture coordinates
        var bin = new MemoryStream();
        (int Offset, int Length) Add(ReadOnlySpan<byte> bytes)
        {
            while (bin.Length % 4 != 0) bin.WriteByte(0);
            var offset = (int)bin.Length;
            bin.Write(bytes);
            return (offset, bytes.Length);
        }
        var indexView = Add(System.Runtime.InteropServices.MemoryMarshal.AsBytes<uint>(mesh.Indices));
        var positionView = Add(System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(mesh.Positions));
        var albedoView = Add(albedo);
        var normalView = Add(normal);
        var uvView = Add(System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(mesh.Uv));
        while (bin.Length % 4 != 0) bin.WriteByte(0);

        var json = Json(mesh, textures, [indexView, positionView, albedoView, normalView, uvView], (int)bin.Length);
        return Container(json, bin.ToArray());
    }

    /// <summary>The picture as a JPEG (4:2:0, as Pillow writes it by default).</summary>
    public static byte[] Jpeg(byte[] rgb, int size, int quality)
    {
        using var bitmap = new SKBitmap(new SKImageInfo(size, size, SKColorType.Rgba8888, SKAlphaType.Opaque));
        var rgba = new byte[size * size * 4];
        for (var i = 0; i < size * size; i++)
        {
            rgba[i * 4] = rgb[i * 3]; rgba[i * 4 + 1] = rgb[i * 3 + 1]; rgba[i * 4 + 2] = rgb[i * 3 + 2]; rgba[i * 4 + 3] = 255;
        }
        System.Runtime.InteropServices.Marshal.Copy(rgba, 0, bitmap.GetPixels(), rgba.Length);
        using var pixmap = bitmap.PeekPixels();
        using var data = pixmap.Encode(new SKJpegEncoderOptions(quality, SKJpegEncoderDownsample.Downsample420, SKJpegEncoderAlphaOption.Ignore));
        return data.ToArray();
    }

    private static byte[] Json(ModelMesh mesh, BakedTextures textures, (int Offset, int Length)[] views, int binLength)
    {
        float Min(int stride, int axis) { var m = float.MaxValue; for (var i = axis; i < (stride == 3 ? mesh.Positions.Length : mesh.Uv.Length); i += stride) m = Math.Min(m, stride == 3 ? mesh.Positions[i] : mesh.Uv[i]); return m; }
        float Max(int stride, int axis) { var m = float.MinValue; for (var i = axis; i < (stride == 3 ? mesh.Positions.Length : mesh.Uv.Length); i += stride) m = Math.Max(m, stride == 3 ? mesh.Positions[i] : mesh.Uv[i]); return m; }

        using var stream = new MemoryStream();
        using (var w = new Utf8JsonWriter(stream))
        {
            w.WriteStartObject();
            w.WriteNumber("scene", 0);
            w.WriteStartArray("scenes"); w.WriteStartObject(); w.WriteStartArray("nodes"); w.WriteNumberValue(0); w.WriteEndArray(); w.WriteEndObject(); w.WriteEndArray();
            w.WriteStartObject("asset"); w.WriteString("version", "2.0"); w.WriteString("generator", $"stablefast3d-net {EngineInfo.Version}"); w.WriteEndObject();

            w.WriteStartArray("accessors");
            w.WriteStartObject();                                                                  // 0: the indices
            w.WriteNumber("componentType", 5125); w.WriteString("type", "SCALAR"); w.WriteNumber("bufferView", 0); w.WriteNumber("count", mesh.Indices.Length);
            w.WriteStartArray("max"); w.WriteNumberValue(mesh.Indices.Max()); w.WriteEndArray();
            w.WriteStartArray("min"); w.WriteNumberValue(mesh.Indices.Min()); w.WriteEndArray();
            w.WriteEndObject();
            w.WriteStartObject();                                                                  // 1: the positions
            w.WriteNumber("componentType", 5126); w.WriteString("type", "VEC3"); w.WriteNumber("byteOffset", 0); w.WriteNumber("bufferView", 1); w.WriteNumber("count", mesh.VertexCount);
            w.WriteStartArray("max"); for (var a = 0; a < 3; a++) w.WriteNumberValue((double)Max(3, a)); w.WriteEndArray();
            w.WriteStartArray("min"); for (var a = 0; a < 3; a++) w.WriteNumberValue((double)Min(3, a)); w.WriteEndArray();
            w.WriteEndObject();
            w.WriteStartObject();                                                                  // 2: the texture coordinates
            w.WriteNumber("componentType", 5126); w.WriteString("type", "VEC2"); w.WriteNumber("byteOffset", 0); w.WriteNumber("bufferView", 4); w.WriteNumber("count", mesh.VertexCount);
            w.WriteStartArray("max"); for (var a = 0; a < 2; a++) w.WriteNumberValue((double)Max(2, a)); w.WriteEndArray();
            w.WriteStartArray("min"); for (var a = 0; a < 2; a++) w.WriteNumberValue((double)Min(2, a)); w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("meshes");
            w.WriteStartObject(); w.WriteString("name", "geometry_0");
            w.WriteStartObject("extras"); w.WriteBoolean("processed", true); w.WriteEndObject();
            w.WriteStartArray("primitives"); w.WriteStartObject();
            w.WriteStartObject("attributes"); w.WriteNumber("POSITION", 1); w.WriteNumber("TEXCOORD_0", 2); w.WriteEndObject();
            w.WriteNumber("indices", 0); w.WriteNumber("mode", 4); w.WriteNumber("material", 0);
            w.WriteEndObject(); w.WriteEndArray();
            w.WriteEndObject(); w.WriteEndArray();

            w.WriteStartArray("images");
            foreach (var view in new[] { 2, 3 }) { w.WriteStartObject(); w.WriteNumber("bufferView", view); w.WriteString("mimeType", "image/jpeg"); w.WriteEndObject(); }
            w.WriteEndArray();
            w.WriteStartArray("textures");
            foreach (var source in new[] { 0, 1 }) { w.WriteStartObject(); w.WriteNumber("source", source); w.WriteEndObject(); }
            w.WriteEndArray();

            w.WriteStartArray("materials"); w.WriteStartObject();
            w.WriteStartObject("pbrMetallicRoughness");
            w.WriteStartObject("baseColorTexture"); w.WriteNumber("index", 0); w.WriteEndObject();
            w.WriteNumber("roughnessFactor", (double)textures.Roughness);
            w.WriteNumber("metallicFactor", (double)textures.Metallic);
            w.WriteEndObject();
            w.WriteBoolean("doubleSided", false);
            w.WriteStartObject("normalTexture"); w.WriteNumber("index", 1); w.WriteEndObject();
            w.WriteEndObject(); w.WriteEndArray();

            w.WriteStartArray("nodes");
            w.WriteStartObject(); w.WriteString("name", "world"); w.WriteStartArray("children"); w.WriteNumberValue(1); w.WriteEndArray(); w.WriteEndObject();
            w.WriteStartObject(); w.WriteString("name", "geometry_0"); w.WriteNumber("mesh", 0); w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("buffers"); w.WriteStartObject(); w.WriteNumber("byteLength", binLength); w.WriteEndObject(); w.WriteEndArray();
            w.WriteStartArray("bufferViews");
            foreach (var (offset, length) in views)
            {
                w.WriteStartObject(); w.WriteNumber("buffer", 0); w.WriteNumber("byteOffset", offset); w.WriteNumber("byteLength", length); w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteEndObject();
        }
        return stream.ToArray();
    }

    /// <summary>The binary container: the 12-byte header, the JSON chunk (padded with spaces) and the binary chunk (padded with zeros), each chunk a multiple of 4 bytes.</summary>
    private static byte[] Container(byte[] json, byte[] bin)
    {
        var jsonPadded = (json.Length + 3) / 4 * 4;
        var binPadded = (bin.Length + 3) / 4 * 4;
        var total = 12 + 8 + jsonPadded + 8 + binPadded;
        using var o = new MemoryStream(total);
        using var w = new BinaryWriter(o);
        w.Write(0x46546C67u); w.Write(2u); w.Write((uint)total);
        w.Write((uint)jsonPadded); w.Write(0x4E4F534Au); w.Write(json);
        for (var i = json.Length; i < jsonPadded; i++) w.Write((byte)' ');
        w.Write((uint)binPadded); w.Write(0x004E4942u); w.Write(bin);
        for (var i = bin.Length; i < binPadded; i++) w.Write((byte)0);
        return o.ToArray();
    }
}
