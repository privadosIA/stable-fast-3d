namespace StableFast3D.Service;

/// <summary>A model the engine made: the binary glTF and what is said of it in the job.</summary>
/// <param name="Glb">The <c>.glb</c>: upright (Y up), its origin at the middle of its base, with its colours, within <see cref="EngineInfo.MaxTriangles"/>.</param>
/// <param name="DetailUsed">The detail it was made at: always the one asked for, because the mesh of Stable Fast 3D does not depend on it (the detail is the size of the textures).</param>
/// <param name="Resolution">The resolution of the surface in points a side (the grid of the marching tetrahedra, 160): the same for every detail.</param>
/// <param name="Triangles">The triangles of the mesh.</param>
internal sealed record GenerationResult(byte[] Glb, string DetailUsed, int Resolution, int Triangles);

/// <summary>The engine of the service: one picture in, one model out.</summary>
internal interface IModelEngine : IDisposable
{
    /// <summary>What a model made by this engine says made it.</summary>
    string Version { get; }

    /// <summary>Makes the model of a picture (a png, jpg or webp). Throws <see cref="RefusedException"/> for what it will not make, with a sentence for the caller.</summary>
    GenerationResult Generate(byte[] picture, string detail, CancellationToken cancel);
}

/// <summary>The engine refuses the request (not an error of the service): the message is what the caller is told.</summary>
internal sealed class RefusedException(string message) : Exception(message);

/// <summary>Chooses the engine from the settings.</summary>
internal static class EngineFactory
{
    public static IModelEngine Create(ServiceSettings settings) => new StableFast3DEngine(settings);
}
