using StableFast3D.Service;

namespace StableFast3D.Net.Tests;

public class ServiceSettingsTests
{
    private static ServiceSettings From(params (string Name, string Value)[] env) =>
        ServiceSettings.From(name => env.Where(e => e.Name == name).Select(e => e.Value).FirstOrDefault());

    [Fact]
    public void WithNothingSetItListensOnEveryInterfaceOfThePortOfTheContainer()
    {
        var s = From();
        Assert.Equal(("0.0.0.0", 8791, "cpu", "/ckpt", "1"), (s.Host, s.Port, s.Device, s.CheckpointDirectory, s.Version));
        Assert.Equal("http://0.0.0.0:8791", s.ListenUrl);
    }

    [Theory]
    [InlineData("127.0.0.1", "http://127.0.0.1:9000")]
    [InlineData("localhost", "http://localhost:9000")]
    [InlineData("::1", "http://[::1]:9000")]
    public void TheHostCanBeLoopbackOnly(string host, string url)
    {
        var s = From((ServiceSettings.HostVariable, host), (ServiceSettings.PortVariable, "9000"));
        Assert.Equal(host, s.Host);
        Assert.Equal(url, s.ListenUrl);
    }

    [Theory]
    [InlineData("example.com")]
    [InlineData("0.0.0.0:80")]
    [InlineData("300.1.1.1")]
    public void AHostThatIsNotAnAddressIsRefusedNamingTheVariable(string host)
    {
        var ex = Assert.Throws<SettingsException>(() => From((ServiceSettings.HostVariable, host)));
        Assert.Contains("SF3D_HOST", ex.Message);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("70000")]
    [InlineData("eighty")]
    public void APortThatIsNotOneIsRefusedNamingTheVariable(string port) =>
        Assert.Contains("SF3D_PORT", Assert.Throws<SettingsException>(() => From((ServiceSettings.PortVariable, port))).Message);

    [Fact]
    public void ADeviceThatIsNotOneIsRefusedNamingTheVariable() =>
        Assert.Contains("SF3D_DEVICE", Assert.Throws<SettingsException>(() => From((ServiceSettings.DeviceVariable, "tpu"))).Message);
}
