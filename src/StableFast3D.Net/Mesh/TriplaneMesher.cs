using StableFast3D.Field;
using TorchSharp;
using static TorchSharp.torch;

namespace StableFast3D.Mesh;

/// <summary>
/// <c>SF3D.triplane_to_meshes</c>: the surface of the object the scene codes describe. The decoder is asked for the density and the offset of every vertex of the
/// tetrahedra grid (placed in the scene's cube), the offsets move the vertices a little, and marching tetrahedra finds where the density crosses the threshold.
/// </summary>
public sealed class TriplaneMesher
{
    private readonly TetrahedraGrid _grid;
    private readonly MarchingTetrahedra _marching;
    private readonly MaterialMlp _decoder;
    private readonly Sf3dConfig _config;
    private readonly Tensor _bbox;

    public TriplaneMesher(TetrahedraGrid grid, MaterialMlp decoder, Sf3dConfig config)
    {
        _grid = grid;
        _marching = new MarchingTetrahedra(grid);
        _decoder = decoder;
        _config = config;
        _bbox = tensor(new float[] { (float)-config.Radius, (float)-config.Radius, (float)-config.Radius, (float)config.Radius, (float)config.Radius, (float)config.Radius }, [2, 3]);
    }

    /// <summary><c>scale_tensor(x, (0, 1), bbox)</c>: the same operations in the same order as upstream, the box's float32 corners as tensors.</summary>
    public Tensor ToScene(Tensor unit)
    {
        var scaled = (unit - MarchingTetrahedra.RangeMin) / (MarchingTetrahedra.RangeMax - MarchingTetrahedra.RangeMin);
        var bbox = _bbox.to(unit.device);
        return scaled * (bbox[1] - bbox[0]) + bbox[0];
    }

    /// <summary>The density and the vertex offset at every vertex of the grid, with the triplane of a scene (3, C, H, W): (Nv, 1) and (Nv, 3).</summary>
    public (Tensor Density, Tensor Offset) DecodeGrid(Tensor sceneCodes)
    {
        using var _ = no_grad();
        var values = TriplaneQuery.Query(ToScene(_grid.Vertices.to(sceneCodes.device)), sceneCodes, _config.Radius);
        var decoded = _decoder.Forward(values, include: ["vertex_offset", "density"]);
        return (decoded["density"], decoded["vertex_offset"]);
    }

    /// <summary>The mesh of a scene (3, C, H, W), in the scene's cube (<c>[-radius, radius]^3</c>); empty when nothing is above the threshold.</summary>
    public TriMesh Extract(Tensor sceneCodes)
    {
        var (density, offset) = DecodeGrid(sceneCodes);
        return Extract(density, offset);
    }

    /// <summary>The mesh for a density (Nv, 1) and an offset (Nv, 3) already decoded on the grid (the stage after the decoder).</summary>
    public TriMesh Extract(Tensor density, Tensor offset)
    {
        using var _ = no_grad();
        var sdf = (density - _config.IsosurfaceThreshold).reshape(-1, 1);
        var positions = _marching.Deform(offset.reshape(-1, 3).to(_grid.Vertices.device));
        var mesh = _marching.Extract(positions, sdf);
        return mesh with { Vertices = ToScene(mesh.Vertices) };
    }
}
