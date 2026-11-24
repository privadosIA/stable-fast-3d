using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Mesh;

/// <summary>One array of a <c>.npy</c> file: its element type (numpy's <c>descr</c>), its shape and its bytes, little-endian, C order.</summary>
public sealed record NpyArray(string Descr, long[] Shape, byte[] Data)
{
    public long Length => Shape.Aggregate(1L, (a, d) => a * d);

    /// <summary>The array as a tensor of the type it is stored in (float64 stays float64, int64 stays int64).</summary>
    public Tensor ToTensor() => Descr switch
    {
        "<f8" => tensor(MemoryCast<double>(), Shape),
        "<f4" => tensor(MemoryCast<float>(), Shape),
        "<i8" => tensor(MemoryCast<long>(), Shape),
        "<i4" => tensor(MemoryCast<int>(), Shape),
        _ => throw new NotSupportedException($"the array type {Descr} is not read (float32, float64, int32 and int64 are)"),
    };

    private T[] MemoryCast<T>() where T : unmanaged
    {
        var o = new T[Length];
        Buffer.BlockCopy(Data, 0, o, 0, Data.Length);
        return o;
    }
}

/// <summary>Reads numpy's <c>.npz</c> (a zip of <c>.npy</c> files), which is how the tetrahedra grid of Stable Fast 3D is stored. No Python, no numpy.</summary>
public static class NpzReader
{
    private static readonly byte[] Magic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];
    private static readonly Regex Header = new(@"'descr':\s*'(?<d>[^']+)'.*?'fortran_order':\s*(?<f>True|False).*?'shape':\s*\((?<s>[^)]*)\)", RegexOptions.Compiled | RegexOptions.Singleline);

    /// <summary>Every array of the file by name (without the <c>.npy</c>).</summary>
    public static Dictionary<string, NpyArray> Read(string path)
    {
        using var zip = ZipFile.OpenRead(path);
        var arrays = new Dictionary<string, NpyArray>();
        foreach (var entry in zip.Entries.Where(e => e.FullName.EndsWith(".npy", StringComparison.Ordinal)))
        {
            using var s = entry.Open();
            using var ms = new MemoryStream((int)entry.Length);
            s.CopyTo(ms);
            arrays[entry.FullName[..^4]] = ParseNpy(ms.ToArray(), entry.FullName);
        }
        return arrays;
    }

    /// <exception cref="InvalidDataException">Not a <c>.npy</c>, a version it does not read, an array in Fortran order or one whose size does not match its header.</exception>
    public static NpyArray ParseNpy(byte[] bytes, string name = "array")
    {
        if (bytes.Length < 10 || !bytes.AsSpan(0, 6).SequenceEqual(Magic)) throw new InvalidDataException($"{name} is not a .npy file");
        int major = bytes[6];
        int headerLength, headerStart;
        if (major == 1) { headerLength = BitConverter.ToUInt16(bytes, 8); headerStart = 10; }
        else if (major is 2 or 3) { headerLength = (int)BitConverter.ToUInt32(bytes, 8); headerStart = 12; }
        else throw new InvalidDataException($"{name}: .npy version {major} is not read");

        var m = Header.Match(Encoding.Latin1.GetString(bytes, headerStart, headerLength));
        if (!m.Success) throw new InvalidDataException($"{name}: cannot read the .npy header");
        if (m.Groups["f"].Value == "True") throw new InvalidDataException($"{name}: arrays in Fortran order are not read");
        var shape = m.Groups["s"].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(long.Parse).ToArray();
        var descr = m.Groups["d"].Value;
        var itemSize = descr.Length >= 3 && int.TryParse(descr[2..], out var n) ? n : throw new InvalidDataException($"{name}: cannot read the type {descr}");
        var count = shape.Aggregate(1L, (a, d) => a * d);
        var data = bytes.AsSpan(headerStart + headerLength).ToArray();
        if (data.Length != count * itemSize) throw new InvalidDataException($"{name}: the header says {count} elements of {itemSize} bytes, the file holds {data.Length} bytes");
        return new NpyArray(descr, shape, data);
    }
}
