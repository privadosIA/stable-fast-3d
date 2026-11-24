namespace StableFast3D.Texture;

/// <summary>
/// numpy's legacy <c>RandomState</c> (the Mersenne Twister MT19937 seeded as <c>np.random.seed(seed)</c> does, and <c>random_sample</c> for doubles): the generator the
/// reference's <c>float32_to_uint8_np</c> draws its dither from. Same numbers as numpy for the same seed.
/// </summary>
public sealed class NumpyRandom
{
    private const int N = 624, M = 397;
    private const uint MatrixA = 0x9908b0df, UpperMask = 0x80000000, LowerMask = 0x7fffffff;

    private readonly uint[] _mt = new uint[N];
    private int _index = N;

    public NumpyRandom(uint seed)
    {
        // numpy's mt19937_seed: the 1812433253 recurrence of the reference implementation
        var s = seed;
        for (var i = 0; i < N; i++)
        {
            _mt[i] = s;
            s = unchecked(1812433253u * (s ^ (s >> 30)) + (uint)i + 1);
        }
    }

    public uint NextUInt32()
    {
        if (_index >= N) Twist();
        var y = _mt[_index++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }

    /// <summary><c>random_sample</c>: 53 random bits as a double in [0, 1).</summary>
    public double NextDouble()
    {
        var a = NextUInt32() >> 5;
        var b = NextUInt32() >> 6;
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }

    private void Twist()
    {
        for (var k = 0; k < N - M; k++)
        {
            var y = (_mt[k] & UpperMask) | (_mt[k + 1] & LowerMask);
            _mt[k] = _mt[k + M] ^ (y >> 1) ^ ((y & 1) != 0 ? MatrixA : 0);
        }
        for (var k = N - M; k < N - 1; k++)
        {
            var y = (_mt[k] & UpperMask) | (_mt[k + 1] & LowerMask);
            _mt[k] = _mt[k + (M - N)] ^ (y >> 1) ^ ((y & 1) != 0 ? MatrixA : 0);
        }
        var last = (_mt[N - 1] & UpperMask) | (_mt[0] & LowerMask);
        _mt[N - 1] = _mt[M - 1] ^ (last >> 1) ^ ((last & 1) != 0 ? MatrixA : 0);
        _index = 0;
    }
}
