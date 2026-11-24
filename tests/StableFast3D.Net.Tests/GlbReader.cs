using System.Text;
using System.Text.Json;

namespace StableFast3D.Net.Tests;

/// <summary>A reader of binary glTF for the tests: the JSON and the bytes of the images a material's base colour and normal textures point at.</summary>
public sealed class GlbReader
{
    public JsonDocument Json { get; }
    private readonly byte[] _bin;

    public GlbReader(byte[] glb)
    {
        if (glb.Length < 20 || Encoding.ASCII.GetString(glb, 0, 4) != "glTF") throw new InvalidDataException("not a glb");
        var version = BitConverter.ToUInt32(glb, 4);
        if (version != 2) throw new InvalidDataException($"glTF version {version}");
        var total = (int)BitConverter.ToUInt32(glb, 8);
        var offset = 12;
        JsonDocument? json = null;
        byte[] bin = [];
        while (offset + 8 <= total)
        {
            var length = (int)BitConverter.ToUInt32(glb, offset);
            var type = BitConverter.ToUInt32(glb, offset + 4);
            var data = glb.AsSpan(offset + 8, length);
            if (type == 0x4E4F534A) json = JsonDocument.Parse(Encoding.UTF8.GetString(data).TrimEnd('\0', ' '));
            else if (type == 0x004E4942) bin = data.ToArray();
            offset += 8 + length;
        }
        Json = json ?? throw new InvalidDataException("the glb has no JSON chunk");
        _bin = bin;
    }

    public byte[] BufferView(int index)
    {
        var view = Json.RootElement.GetProperty("bufferViews")[index];
        var start = view.TryGetProperty("byteOffset", out var o) ? o.GetInt32() : 0;
        return _bin.AsSpan(start, view.GetProperty("byteLength").GetInt32()).ToArray();
    }

    /// <summary>The encoded bytes (png or jpeg) of the image of a texture slot of the first material, or null when the material has none.</summary>
    public (string Mime, byte[] Bytes)? Image(string slot)
    {
        var material = Json.RootElement.GetProperty("materials")[0];
        JsonElement texture;
        if (slot == "baseColor")
        {
            if (!material.TryGetProperty("pbrMetallicRoughness", out var pbr) || !pbr.TryGetProperty("baseColorTexture", out texture)) return null;
        }
        else if (!material.TryGetProperty(slot, out texture)) return null;
        var source = Json.RootElement.GetProperty("textures")[texture.GetProperty("index").GetInt32()].GetProperty("source").GetInt32();
        var image = Json.RootElement.GetProperty("images")[source];
        return (image.GetProperty("mimeType").GetString()!, BufferView(image.GetProperty("bufferView").GetInt32()));
    }

    public (double Roughness, double Metallic) MaterialFactors()
    {
        var pbr = Json.RootElement.GetProperty("materials")[0].GetProperty("pbrMetallicRoughness");
        return (pbr.GetProperty("roughnessFactor").GetDouble(), pbr.GetProperty("metallicFactor").GetDouble());
    }
}
