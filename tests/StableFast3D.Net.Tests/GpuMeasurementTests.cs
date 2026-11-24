using System.Diagnostics;
using System.Globalization;
using TorchSharp;

namespace StableFast3D.Net.Tests;

/// <summary>[Fact] for the measurements on the card: it needs the weights, SF3D_DEVICE=cuda (or cuda:N), a CUDA build of the libraries and the fixtures.</summary>
public sealed class RequiresCudaCheckpointAndPicturesAttribute : FactAttribute
{
    public RequiresCudaCheckpointAndPicturesAttribute() =>
        Skip = Checkpoint.Env.NoCheckpointReason
            ?? (!Checkpoint.Env.DeviceName.StartsWith("cuda") ? "SF3D_DEVICE is not cuda: the measurements on the card are opt-in"
            : !torch.cuda.is_available() ? "no CUDA device or CUDA build (build with -p:TorchBackend=cuda-linux)" : null)
            ?? Fixtures.Missing(Fixtures.Pictures.Select(p => $"pictures/{p}.png").ToArray());
}

/// <summary>What a model costs on the card: seconds and video memory per picture and detail, against the 8 GB of the container (task 8.2).</summary>
[Collection(Heavy.Name)]
public class GpuMeasurementTests
{
    /// <summary>The video memory this process has in use, in MiB, as <c>nvidia-smi</c> sees it (the CUDA context, the weights and the allocator's cache): an upper bound of what a model needs.</summary>
    private static long ProcessVideoMemoryMb()
    {
        try
        {
            using var p = Process.Start(new ProcessStartInfo("nvidia-smi", "--query-compute-apps=pid,used_memory --format=csv,noheader,nounits")
            { RedirectStandardOutput = true, UseShellExecute = false });
            var text = p!.StandardOutput.ReadToEnd();
            p.WaitForExit(3000);
            foreach (var line in text.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                var parts = line.Split(',', StringSplitOptions.TrimEntries);
                if (parts.Length == 2 && int.TryParse(parts[0], out var pid) && pid == Environment.ProcessId && long.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out var mb)) return mb;
            }
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException) { }
        return 0;
    }

    [RequiresCudaCheckpointAndPictures]
    public void TheModelsAreMadeInSecondsOnTheCardAndFitItsEightGigabytes()
    {
        var device = Checkpoint.Env.DeviceName;
        var sw = Stopwatch.StartNew();
        using var generator = ModelGenerator.Load(Checkpoint.Dir!, device);
        var loadSeconds = sw.Elapsed.TotalSeconds;
        torch.cuda.synchronize();
        var afterLoad = ProcessVideoMemoryMb();
        Console.WriteLine($"MEASURE device {device}: weights loaded in {loadSeconds:F1} s, {afterLoad} MiB of video memory in use after loading");

        generator.Generate(File.ReadAllBytes(Fixtures.Dir.PathOf("pictures/axe.png")), 512);   // warm-up: kernels, allocator, the first use of every op

        var peak = 0L;
        var failures = new List<string>();
        foreach (var size in new[] { 512, 1024, 2048 })
            foreach (var name in Fixtures.Pictures)
            {
                var picture = File.ReadAllBytes(Fixtures.Dir.PathOf($"pictures/{name}.png"));
                long local = 0;
                using var stop = new CancellationTokenSource();
                var poll = Task.Run(() => { while (!stop.IsCancellationRequested) { local = Math.Max(local, ProcessVideoMemoryMb()); Thread.Sleep(100); } });
                sw.Restart();
                var made = generator.Generate(picture, size);
                torch.cuda.synchronize();
                var seconds = sw.Elapsed.TotalSeconds;
                stop.Cancel();
                poll.Wait();
                local = Math.Max(local, ProcessVideoMemoryMb());
                peak = Math.Max(peak, local);
                Console.WriteLine($"MEASURE {name} at {size}: {seconds:F2} s, peak video memory {local} MiB, {made.Triangles} triangles, {made.Glb.Length / 1024} KiB");
            }

        Console.WriteLine($"MEASURE overall peak video memory {peak} MiB, peak resident memory {Process.GetCurrentProcess().PeakWorkingSet64 >> 20} MiB");
        if (peak > 8192) failures.Add($"the peak video memory, {peak} MiB, is above the 8 GB of the container");
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
