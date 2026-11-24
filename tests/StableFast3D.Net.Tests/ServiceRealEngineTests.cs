using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using StableFast3D.Service;

namespace StableFast3D.Net.Tests;

/// <summary>The service with the real engine and the real weights: a picture is posted and a model comes back, through the whole HTTP contract.</summary>
[Collection(Heavy.Name)]
public class ServiceRealEngineTests : IAsyncLifetime
{
    private WebApplication _app = null!;
    private HttpClient _http = null!;

    public async Task InitializeAsync()
    {
        if (Checkpoint.Dir is null) return;
        var settings = new ServiceSettings(Checkpoint.Dir, "", 0, "test", "cpu");
        var engine = EngineFactory.Create(settings);
        var gate = new FifoGate();
        var api = new ServiceApp(settings, new JobStore(engine, gate), new HealthProbe(settings, gate, engine.Version, TimeSpan.Zero), gate);
        _app = ServiceApp.Build(api, "http://127.0.0.1:0");
        await _app.StartAsync();
        var address = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        _http = new HttpClient { BaseAddress = new Uri(address), Timeout = TimeSpan.FromMinutes(5) };
    }

    public async Task DisposeAsync()
    {
        if (Checkpoint.Dir is null) return;
        _http.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
    }

    private static async Task<JsonElement> Json(HttpResponseMessage r) => JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement;

    private async Task<JsonElement> Wait(string id)
    {
        for (var i = 0; i < 600; i++)
        {
            var job = await Json(await _http.GetAsync($"/jobs/{id}"));
            if (job.GetProperty("state").GetString() is "done" or "failed") return job;
            await Task.Delay(500);
        }
        throw new TimeoutException("the job did not finish");
    }

    [RequiresCheckpointAndPictureFixtures]
    public async Task APictureBecomesAModelThroughTheWholeContract()
    {
        var health = await Json(await _http.GetAsync("/health"));
        Assert.True(health.GetProperty("ready").GetBoolean());

        var r = await _http.PostAsync("/generate?engine=sf3d&detail=low", new ByteArrayContent(File.ReadAllBytes(Fixtures.Dir.PathOf("pictures/axe.png"))));
        Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
        var id = (await Json(r)).GetProperty("id").GetString()!;
        var job = await Wait(id);

        Assert.Equal("done", job.GetProperty("state").GetString());
        Assert.Equal("", job.GetProperty("error").GetString());
        Assert.Equal(("sf3d", EngineInfo.Version, "low", "low"), (job.GetProperty("engine").GetString(), job.GetProperty("engine_version").GetString(), job.GetProperty("detail").GetString(), job.GetProperty("detail_used").GetString()));
        Assert.Equal(160, job.GetProperty("resolution").GetInt32());
        Assert.Equal(5988, job.GetProperty("triangles").GetInt32());
        Assert.Equal(1, job.GetProperty("progress").GetProperty("done").GetInt32());

        var model = await _http.GetAsync($"/jobs/{id}/model");
        Assert.Equal(HttpStatusCode.OK, model.StatusCode);
        Assert.Equal("model/gltf-binary", model.Content.Headers.ContentType?.MediaType);
        var glb = new GlbReader(await model.Content.ReadAsByteArrayAsync());
        Assert.Equal(5988, glb.Accessor(0).Count / 3);

        Assert.Equal(HttpStatusCode.OK, (await _http.DeleteAsync($"/jobs/{id}")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await _http.GetAsync($"/jobs/{id}/model")).StatusCode);
    }

    [RequiresCheckpoint]
    public async Task APictureTheDecoderCannotReadFailsTheJobWithASentence()
    {
        // it looks like a png to the service (the signature) but is not one
        byte[] fake = [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3, 4, 5, 6, 7, 8];
        var r = await _http.PostAsync("/generate?detail=low", new ByteArrayContent(fake));
        Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
        var job = await Wait((await Json(r)).GetProperty("id").GetString()!);
        Assert.Equal("failed", job.GetProperty("state").GetString());
        Assert.False(string.IsNullOrWhiteSpace(job.GetProperty("error").GetString()));
        Assert.Equal(0, job.GetProperty("triangles").GetInt32());
    }
}
