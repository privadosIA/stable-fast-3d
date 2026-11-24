using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Mesh;

/// <summary>A triangle mesh: vertex positions (Nv, 3) float32 and triangles (Nf, 3) int64.</summary>
public sealed record TriMesh(Tensor Vertices, Tensor Triangles)
{
    public long VertexCount => Vertices.shape[0];
    public long TriangleCount => Triangles.shape[0];
}

/// <summary>The grid of tetrahedra the surface is extracted on (<c>load/tets/160_tets.npz</c> of the upstream repository, kept as <c>sf3d/160_tets.npz</c>).</summary>
public sealed class TetrahedraGrid : IDisposable
{
    /// <summary>The grid's vertices in [0, 1]^3, (Nv, 3) float32 (upstream stores float64 and casts).</summary>
    public Tensor Vertices { get; }

    /// <summary>The four vertices of each tetrahedron, (Nt, 4) int64.</summary>
    public Tensor Indices { get; }

    /// <summary>The side of the grid in points (160), which sets the size of a vertex's allowed offset.</summary>
    public int Resolution { get; }

    public TetrahedraGrid(string path, int resolution = 160)
    {
        var arrays = NpzReader.Read(path);
        if (!arrays.TryGetValue("vertices", out var v) || !arrays.TryGetValue("indices", out var i))
            throw new InvalidDataException($"{path} must hold 'vertices' and 'indices' (it holds {string.Join(", ", arrays.Keys)})");
        if (v.Shape.Length != 2 || v.Shape[1] != 3) throw new InvalidDataException($"{path}: vertices are (N, 3), got ({string.Join(", ", v.Shape)})");
        if (i.Shape.Length != 2 || i.Shape[1] != 4) throw new InvalidDataException($"{path}: indices are (N, 4), got ({string.Join(", ", i.Shape)})");
        using var raw = v.ToTensor();
        Vertices = raw.to(ScalarType.Float32);
        Indices = i.ToTensor().to(ScalarType.Int64);
        Resolution = resolution;
        if (Indices.max().item<long>() >= Vertices.shape[0]) throw new InvalidDataException($"{path}: a tetrahedron names a vertex that is not there");
    }

    public void Dispose()
    {
        Vertices.Dispose();
        Indices.Dispose();
    }
}

/// <summary>
/// <c>MarchingTetrahedraHelper</c>: the surface where a value on the grid's vertices changes sign. The vertices of the mesh come out in the order of upstream (the
/// sorted edges that cross the surface) and its triangles in the order of upstream too (the tetrahedra that give one triangle, then the ones that give two), so
/// the same density gives the same mesh index by index.
/// </summary>
public sealed class MarchingTetrahedra
{
    // the grid is in [0, 1]^3
    public const double RangeMin = 0.0, RangeMax = 1.0;

    // for each of the 16 sign patterns of a tetrahedron (bit i set when vertex i is inside), the edges (of its 6: 01 02 03 12 13 23) that make its triangles
    private static readonly long[] TriangleTable =
    [
        -1, -1, -1, -1, -1, -1,
        1, 0, 2, -1, -1, -1,
        4, 0, 3, -1, -1, -1,
        1, 4, 2, 1, 3, 4,
        3, 1, 5, -1, -1, -1,
        2, 3, 0, 2, 5, 3,
        1, 4, 0, 1, 5, 4,
        4, 2, 5, -1, -1, -1,
        4, 5, 2, -1, -1, -1,
        4, 1, 0, 4, 5, 1,
        3, 2, 0, 3, 5, 2,
        1, 3, 5, -1, -1, -1,
        4, 1, 2, 4, 3, 1,
        3, 0, 4, -1, -1, -1,
        2, 0, 1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1,
    ];

    private static readonly long[] NumTriangles = [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0];
    private static readonly long[] BaseTetEdges = [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3];

    private readonly TetrahedraGrid _grid;

    public MarchingTetrahedra(TetrahedraGrid grid) => _grid = grid;

    /// <summary>The grid's vertices moved by a learned offset: at most half a tetrahedron (<c>(max - min) / resolution</c>) in each direction (tanh keeps them there).</summary>
    public Tensor Deform(Tensor offsets) => _grid.Vertices.to(offsets.device) + (RangeMax - RangeMin) / _grid.Resolution * tanh(offsets);

    /// <param name="positions">(Nv, 3): where each grid vertex is</param>
    /// <param name="sdf">(Nv, 1): the value at each vertex; a vertex is inside where it is above 0</param>
    public TriMesh Extract(Tensor positions, Tensor sdf)
    {
        var (vertices, triangles) = Frugal.Run(() => { var m = ExtractCore(positions, sdf); return (m.Vertices, m.Triangles); });
        return new TriMesh(vertices, triangles);
    }

    private TriMesh ExtractCore(Tensor positions, Tensor sdf)
    {
        using var _ = no_grad();
        var device = positions.device;
        var tets = _grid.Indices.to(device);
        var occ = sdf.gt(0).reshape(-1);                                                     // (Nv)
        var occTets = occ.index_select(0, tets.reshape(-1)).reshape(-1, 4);                  // (Nt, 4)
        var occSum = occTets.to(ScalarType.Int64).sum(-1);
        var valid = occSum.gt(0) & occSum.lt(4);
        var validIdx = valid.nonzero().squeeze(1);
        if (validIdx.numel() == 0) return new TriMesh(zeros(0, 3, device: device), zeros(0, 3, dtype: ScalarType.Int64, device: device));

        // every edge of the tetrahedra the surface crosses, as a sorted pair, once
        var edgeIdx = tensor(BaseTetEdges, device: device);
        var validTets = tets.index_select(0, validIdx);
        var allEdges = validTets.index_select(1, edgeIdx).reshape(-1, 2);
        var lo = minimum(allEdges[TensorIndex.Colon, 0], allEdges[TensorIndex.Colon, 1]);
        var hi = maximum(allEdges[TensorIndex.Colon, 0], allEdges[TensorIndex.Colon, 1]);
        // sorted by (lo, hi), as numpy/torch sort the rows: one key per edge keeps that order
        var key = lo * positions.shape[0] + hi;
        var (uniqueKeys, inverse, _) = unique(key, sorted: true, return_inverse: true);
        var uniqueLo = uniqueKeys.floor_divide(positions.shape[0]);
        var uniqueHi = uniqueKeys - uniqueLo * positions.shape[0];

        // the edges whose ends are on different sides are where a vertex of the mesh goes
        var cross = occ.index_select(0, uniqueLo).to(ScalarType.Int64) + occ.index_select(0, uniqueHi).to(ScalarType.Int64) == 1;
        var mapping = full(new long[] { uniqueKeys.shape[0] }, -1L, dtype: ScalarType.Int64, device: device);
        var crossIdx = cross.nonzero().squeeze(1);
        mapping.index_copy_(0, crossIdx, arange(crossIdx.shape[0], dtype: ScalarType.Int64, device: device));
        var idxMap = mapping.index_select(0, inverse!).reshape(-1, 6);                        // (valid tets, 6): the mesh vertex on each edge, or -1

        var interpA = uniqueLo.index_select(0, crossIdx);
        var interpB = uniqueHi.index_select(0, crossIdx);
        var pa = positions.index_select(0, interpA);
        var pb = positions.index_select(0, interpB);
        var sa = sdf.reshape(-1).index_select(0, interpA).unsqueeze(-1);
        var sb = -sdf.reshape(-1).index_select(0, interpB).unsqueeze(-1);                    // upstream negates the second value, then flips
        var denominator = sa + sb;
        var wa = sb / denominator;
        var wb = sa / denominator;
        var verts = pa * wa + pb * wb;

        // the triangles: which of the 6 edges of a tetrahedron make them depends on which of its vertices are inside
        var vId = tensor(new long[] { 1, 2, 4, 8 }, device: device);
        var tetIndex = (occTets.index_select(0, validIdx).to(ScalarType.Int64) * vId.unsqueeze(0)).sum(-1);
        var numTriangles = tensor(NumTriangles, device: device).index_select(0, tetIndex);
        var table = tensor(TriangleTable, [16, 6], device: device);

        Tensor Faces(long count, int columns)
        {
            var sel = numTriangles.eq(count).nonzero().squeeze(1);
            var cols = table.index_select(0, tetIndex.index_select(0, sel))[TensorIndex.Colon, TensorIndex.Slice(0, columns)];
            return gather(idxMap.index_select(0, sel), 1, cols).reshape(-1, 3);
        }

        return new TriMesh(verts, cat([Faces(1, 3), Faces(2, 6)], 0));
    }
}
