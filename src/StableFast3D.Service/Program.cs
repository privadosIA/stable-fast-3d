using StableFast3D.Service;

ServiceSettings settings;
try { settings = ServiceSettings.FromEnvironment(); }
catch (SettingsException e)
{
    await Console.Error.WriteLineAsync($"error: {e.Message}");
    return 2;
}

using var engine = EngineFactory.Create(settings);
var gate = new FifoGate();
var jobs = new JobStore(engine, gate);
var api = new ServiceApp(settings, jobs, new HealthProbe(settings, gate, engine.Version), gate);
var app = ServiceApp.Build(api, settings.ListenUrl);
await app.StartAsync();
Console.WriteLine($"stablefast3d-service {settings.Version} listening on {settings.Host}:{settings.Port} (weights: {settings.CheckpointDirectory})");
await app.WaitForShutdownAsync();
return 0;
