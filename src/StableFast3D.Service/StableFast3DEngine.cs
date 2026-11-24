using StableFast3D.Mesh;

namespace StableFast3D.Service;

/// <summary>
/// The engine of the service: Stable Fast 3D, loaded when the first model is asked for (so the service starts at once and <c>/health</c> can say what is missing before any weights are read)
/// and kept in memory. The jobs reach it one at a time (<see cref="FifoGate"/>). What it will not make is a <see cref="RefusedException"/> with a sentence for the caller.
/// </summary>
internal sealed class StableFast3DEngine(ServiceSettings settings) : IModelEngine
{
    private readonly object _lock = new();
    private ModelGenerator? _generator;

    public string Version => EngineInfo.Version;

    public GenerationResult Generate(byte[] picture, string detail, CancellationToken cancel)
    {
        var size = EngineInfo.DetailSizes[detail];
        var generator = Load();
        try
        {
            var made = generator.Generate(picture, size);
            return new GenerationResult(made.Glb, detail, generator.SurfaceResolution, made.Triangles);
        }
        catch (InvalidDataException e) { throw new RefusedException(e.Message); }                // not a picture the decoder can read
        catch (ModelRefusedException e) { throw new RefusedException(e.Message); }               // no surface
        catch (TooManyTrianglesException e) { throw new RefusedException(e.Message); }           // the mesh does not depend on the detail: there is nothing to make again
    }

    private ModelGenerator Load()
    {
        lock (_lock)
        {
            if (_generator is not null) return _generator;
            try { return _generator = ModelGenerator.Load(settings.CheckpointDirectory, settings.Device); }
            catch (Exception e) when (e is FileNotFoundException or InvalidDataException or EndOfStreamException or IOException)
            {
                throw new RefusedException($"the weights cannot be loaded: {e.Message}");
            }
        }
    }

    public void Dispose()
    {
        lock (_lock) { _generator?.Dispose(); _generator = null; }
    }
}
