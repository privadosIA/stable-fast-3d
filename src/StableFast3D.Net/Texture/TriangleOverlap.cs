namespace StableFast3D.Texture;

// A literal port of uv_unwrapper/csrc/intersect.cpp and the helpers of common.h it uses: the test that two triangles of the UV layout overlap, and the one that a triangle
// touches a bounding box. float32 throughout, the operations in the order upstream writes them (no fused multiply-add, as upstream is built for generic x86-64); the
// only double-precision parts are the ones upstream has in double (std::pow in a distance, the 4 x 4 inverse). The tests and their thresholds are upstream's, including
// the ones that are not exact: the unwrapper's result depends on them.

internal struct V2(float x, float y)
{
    public float X = x, Y = y;
    public readonly bool SameAs(in V2 o) => X == o.X && Y == o.Y;
}

internal struct V3(float x, float y, float z)
{
    public float X = x, Y = y, Z = z;

    public static V3 operator -(in V3 a, in V3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    public static V3 operator +(in V3 a, in V3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static V3 operator *(in V3 a, float s) => new(a.X * s, a.Y * s, a.Z * s);

    public static float Dot(in V3 a, in V3 b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;
    public static V3 Cross(in V3 a, in V3 b) => new(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
    public static float Magnitude(in V3 v) => MathF.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);

    public static V3 Normalize(in V3 v)
    {
        var len = MathF.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);
        return new V3(v.X / len, v.Y / len, v.Z / len);
    }

    // std::sqrt(std::pow(dx, 2) + std::pow(dy, 2) + std::pow(dz, 2)): pow and sqrt run in double on the float differences
    public static float Distance(in V3 a, in V3 b)
    {
        double dx = a.X - b.X, dy = a.Y - b.Y, dz = a.Z - b.Z;
        return (float)Math.Sqrt(Math.Pow(dx, 2) + Math.Pow(dy, 2) + Math.Pow(dz, 2));
    }
}

/// <summary>A 4 x 4 float matrix as upstream's <c>Matrix4</c>: row-major, the determinant as an unrolled sum, the inverse in double.</summary>
internal struct Matrix4
{
    public float[] M;          // 16, row-major

    public static Matrix4 Zero() => new() { M = new float[16] };

    // upstream's default constructor: zeros with m[3][3] = 1
    public static Matrix4 Default()
    {
        var m = Zero();
        m.M[15] = 1.0f;
        return m;
    }

    public static Matrix4 Of(params float[] values) => new() { M = (float[])values.Clone() };

    public readonly float Determinant()
    {
        var m = M;
        float m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3], m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7],
              m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11], m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];
        return m03 * m12 * m21 * m30 -
               m02 * m13 * m21 * m30 -
               m03 * m11 * m22 * m30 +
               m01 * m13 * m22 * m30 +
               m02 * m11 * m23 * m30 -
               m01 * m12 * m23 * m30 -
               m03 * m12 * m20 * m31 +
               m02 * m13 * m20 * m31 +
               m03 * m10 * m22 * m31 -
               m00 * m13 * m22 * m31 -
               m02 * m10 * m23 * m31 +
               m00 * m12 * m23 * m31 +
               m03 * m11 * m20 * m32 -
               m01 * m13 * m20 * m32 -
               m03 * m10 * m21 * m32 +
               m00 * m13 * m21 * m32 +
               m01 * m10 * m23 * m32 -
               m00 * m11 * m23 * m32 -
               m02 * m11 * m20 * m33 +
               m01 * m12 * m20 * m33 +
               m02 * m10 * m21 * m33 -
               m00 * m12 * m21 * m33 -
               m01 * m10 * m22 * m33 +
               m00 * m11 * m22 * m33;
    }

    public static Matrix4 operator *(in Matrix4 a, in Matrix4 b)
    {
        var r = Default();
        for (var row = 0; row < 4; row++)
            for (var col = 0; col < 4; col++)
                r.M[row * 4 + col] = a.M[row * 4] * b.M[col] + a.M[row * 4 + 1] * b.M[4 + col] + a.M[row * 4 + 2] * b.M[8 + col] + a.M[row * 4 + 3] * b.M[12 + col];
        return r;
    }

    /// <summary>Inverts in place, in double; leaves the matrix as it is and returns false when the determinant is below 1e-6.</summary>
    public bool Invert()
    {
        var m = new double[16];
        for (var i = 0; i < 16; i++) m[i] = M[i];
        var inv = new double[16];
        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
        var det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
        if (Math.Abs(det) < 1e-6) return false;
        det = 1.0 / det;
        for (var i = 0; i < 16; i++) M[i] = (float)(inv[i] * det);
        return true;
    }

    public readonly void Apply(ref V3 v)
    {
        var m = M;
        var newX = v.X * m[0] + v.Y * m[1] + v.Z * m[2] + m[3];
        var newY = v.X * m[4] + v.Y * m[5] + v.Z * m[6] + m[7];
        var newZ = v.X * m[8] + v.Y * m[9] + v.Z * m[10] + m[11];
        var w = v.X * m[12] + v.Y * m[13] + v.Z * m[14] + m[15];
        if (MathF.Abs(w) > TriangleOverlap.Epsilon)
        {
            newX /= w; newY /= w; newZ /= w;
        }
        v.X = newX; v.Y = newY; v.Z = newZ;
    }
}

internal static class TriangleOverlap
{
    public const float Epsilon = 1e-7f;

    private struct Tri
    {
        public V3 A, B, C;

        public Tri(V3 a, V3 b, V3 c) { A = a; B = b; C = c; }
        public Tri(V2 a, V2 b, V2 c) { A = new V3(a.X, a.Y, 0); B = new V3(b.X, b.Y, 0); C = new V3(c.X, c.Y, 0); }

        public readonly V3 Normal()
        {
            var u = B - A;
            var v = C - A;
            return V3.Normalize(V3.Cross(u, v));
        }
    }

    /// <summary>Whether the triangle touches the box: upstream's corner-code test (strict comparisons, edges against the box sides, then the box's corners against the triangle).</summary>
    public static bool TriangleAabb(V2 aabbMin, V2 aabbMax, V2 v0, V2 v1, V2 v2)
    {
        float l = aabbMin.X, r = aabbMax.X, t = aabbMin.Y, b = aabbMax.Y;
        int Code(V2 v) => (v.X > l ? 1 : 0) | (v.Y > t ? 2 : 0) | (v.X > r ? 4 : 0) | (v.Y > b ? 8 : 0);

        var b0 = Code(v0);
        if (b0 == 3) return true;
        var b1 = Code(v1);
        if (b1 == 3) return true;
        var b2 = Code(v2);
        if (b2 == 3) return true;

        bool Edge(int code, V2 p, V2 q)
        {
            if (code == 0) return false;
            if (q.X != p.X)
            {
                var m = (q.Y - p.Y) / (q.X - p.X);
                var c = p.Y - (m * p.X);
                float s;
                if ((code & 1) != 0) { s = m * l + c; if (s >= t && s <= b) return true; }
                if ((code & 2) != 0) { s = (t - c) / m; if (s >= l && s <= r) return true; }
                if ((code & 4) != 0) { s = m * r + c; if (s >= t && s <= b) return true; }
                if ((code & 8) != 0) { s = (b - c) / m; if (s >= l && s <= r) return true; }
            }
            else
            {
                if (l == p.X || r == p.X) return true;
                if (p.X > l && p.X < r) return true;
            }
            return false;
        }

        if (Edge(b0 ^ b1, v0, v1)) return true;
        if (Edge(b1 ^ b2, v1, v2)) return true;
        if (Edge(b0 ^ b2, v0, v2)) return true;

        var tbbL = MathF.Min(v0.X, MathF.Min(v1.X, v2.X));
        var tbbT = MathF.Min(v0.Y, MathF.Min(v1.Y, v2.Y));
        var tbbR = MathF.Max(v0.X, MathF.Max(v1.X, v2.X));
        var tbbB = MathF.Max(v0.Y, MathF.Max(v1.Y, v2.Y));

        if (tbbL <= l && tbbR >= r && tbbT <= t && tbbB >= b)
        {
            var v0x = v2.X - v0.X;
            var v0y = v2.Y - v0.Y;
            var v1x = v1.X - v0.X;
            var v1y = v1.Y - v0.Y;

            var dot00 = v0x * v0x + v0y * v0y;
            var dot01 = v0x * v1x + v0y * v1y;
            var dot11 = v1x * v1x + v1y * v1y;
            var invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);

            bool Corner(float cx, float cy)
            {
                var v2x = cx - v0.X;
                var v2y = cy - v0.Y;
                var dot02 = v0x * v2x + v0y * v2y;
                var dot12 = v1x * v2x + v1y * v2y;
                var u = (dot11 * dot02 - dot01 * dot12) * invDenom;
                var v = (dot00 * dot12 - dot01 * dot02) * invDenom;
                return u >= 0 && v >= 0 && (u + v) <= 1;
            }

            if (Corner(l, t)) return true;     // top-left
            if (Corner(l, b)) return true;     // bottom-left
            if (Corner(r, b)) return true;     // bottom-right
            if (Corner(r, t)) return true;     // top-right
        }
        return false;
    }

    private static bool IsDegenerate(in Tri tri)
    {
        var u = tri.A - tri.B;
        var v = tri.A - tri.C;
        var cr = V3.Cross(u, v);
        return MathF.Abs(cr.X) < Epsilon && MathF.Abs(cr.Y) < Epsilon && MathF.Abs(cr.Z) < Epsilon;
    }

    private static int Orient3D(in V3 a, in V3 b, in V3 c, in V3 d)
    {
        var m = Matrix4.Of(a.X, a.Y, a.Z, 1, b.X, b.Y, b.Z, 1, c.X, c.Y, c.Z, 1, d.X, d.Y, d.Z, 1);
        var det = m.Determinant();
        if (det < -Epsilon) return -1;
        if (det > Epsilon) return 1;
        return 0;
    }

    private static int Orient2D(in V3 a, in V3 b, in V3 c)
    {
        var det = (a.X * (b.Y - c.Y) + b.X * (c.Y - a.Y) + c.X * (a.Y - b.Y));
        if (det < -Epsilon) return -1;
        if (det > Epsilon) return 1;
        return 0;
    }

    private static void PermuteLeft(ref Tri tri) { var tmp = tri.A; tri.A = tri.B; tri.B = tri.C; tri.C = tmp; }
    private static void PermuteRight(ref Tri tri) { var tmp = tri.C; tri.C = tri.B; tri.B = tri.A; tri.A = tmp; }

    private static void MakeCounterClockwise(ref Tri tri)
    {
        if (Orient2D(tri.A, tri.B, tri.C) < 0) (tri.B, tri.C) = (tri.C, tri.B);
    }

    private static V3 IntersectPlane(in V3 a, in V3 b, in V3 p, in V3 n)
    {
        var u = b - a;
        var v = a - p;
        var dot1 = V3.Dot(n, u);
        var dot2 = V3.Dot(n, v);
        u = u * (-dot2 / dot1);
        return a + u;
    }

    private static void ComputeLineIntersection(in Tri t1, in Tri t2, List<V3> target)
    {
        var n1 = t1.Normal();
        var n2 = t2.Normal();
        var o1 = Orient3D(t1.A, t1.C, t2.B, t2.A);
        var o2 = Orient3D(t1.A, t1.B, t2.C, t2.A);
        V3 i1, i2;
        if (o1 > 0)
        {
            if (o2 > 0) { i1 = IntersectPlane(t1.A, t1.C, t2.A, n2); i2 = IntersectPlane(t2.A, t2.C, t1.A, n1); }
            else { i1 = IntersectPlane(t1.A, t1.C, t2.A, n2); i2 = IntersectPlane(t1.A, t1.B, t2.A, n2); }
        }
        else
        {
            if (o2 > 0) { i1 = IntersectPlane(t2.A, t2.B, t1.A, n1); i2 = IntersectPlane(t2.A, t2.C, t1.A, n1); }
            else { i1 = IntersectPlane(t2.A, t2.B, t1.A, n1); i2 = IntersectPlane(t1.A, t1.B, t2.A, n2); }
        }
        target.Add(i1);
        if (V3.Distance(i1, i2) >= Epsilon) target.Add(i2);
    }

    private static void MakeVertexAlone(ref Tri tri, int oa, int ob, int oc)
    {
        if (oa == ob) PermuteRight(ref tri);
        else if (oa == oc) PermuteLeft(ref tri);
        else if (ob != oc)
        {
            if (ob > 0) PermuteLeft(ref tri);
            else if (oc > 0) PermuteRight(ref tri);
        }
    }

    private static void MakeVertexPositive(ref Tri tri, in Tri other)
    {
        var o = Orient3D(other.A, other.B, other.C, tri.A);
        if (o < 0) (tri.B, tri.C) = (tri.C, tri.B);
    }

    private static bool CrossIntersect(ref Tri t1, ref Tri t2, int o1a, int o1b, int o1c, List<V3>? target)
    {
        var o2a = Orient3D(t1.A, t1.B, t1.C, t2.A);
        var o2b = Orient3D(t1.A, t1.B, t1.C, t2.B);
        var o2c = Orient3D(t1.A, t1.B, t1.C, t2.C);
        if (o2a == o2b && o2a == o2c) return false;

        MakeVertexAlone(ref t1, o1a, o1b, o1c);
        MakeVertexAlone(ref t2, o2a, o2b, o2c);
        MakeVertexPositive(ref t2, t1);
        MakeVertexPositive(ref t1, t2);

        var o1 = Orient3D(t1.A, t1.B, t2.A, t2.B);
        var o2 = Orient3D(t1.A, t1.C, t2.C, t2.A);
        if (o1 <= 0 && o2 <= 0)
        {
            if (target is not null) ComputeLineIntersection(t1, t2, target);
            return true;
        }
        return false;
    }

    private static V3 LinesIntersect2d(in V3 a1, in V3 b1, in V3 a2, in V3 b2)
    {
        var dx1 = a1.X - b1.X;
        var dx2 = a2.X - b2.X;
        var dy1 = a1.Y - b1.Y;
        var dy2 = a2.Y - b2.Y;
        var d = dx1 * dy2 - dx2 * dy1;
        var n1 = a1.X * b1.Y - a1.Y * b1.X;
        var n2 = a2.X * b2.Y - a2.Y * b2.X;
        return new V3((n1 * dx2 - n2 * dx1) / d, (n1 * dy2 - n2 * dy1) / d, 0);
    }

    private static void ClipTriangle(in Tri t1, in Tri t2, List<V3> target)
    {
        V3[] clip = [t1.A, t1.B, t1.C];
        var output = new List<V3> { t2.A, t2.B, t2.C };
        var orients = new int[9];
        for (var i = 0; i < 3; i++)
        {
            var iPrev = (i + 2) % 3;
            var input = new List<V3>(output);
            output.Clear();
            for (var j = 0; j < input.Count; j++) orients[j] = Orient2D(clip[iPrev], clip[i], input[j]);
            for (var j = 0; j < input.Count; j++)
            {
                var jPrev = (j - 1 + input.Count) % input.Count;
                if (orients[j] >= 0)
                {
                    if (orients[jPrev] < 0) output.Add(LinesIntersect2d(clip[iPrev], clip[i], input[jPrev], input[j]));
                    output.Add(input[j]);
                }
                else if (orients[jPrev] >= 0)
                {
                    output.Add(LinesIntersect2d(clip[iPrev], clip[i], input[jPrev], input[j]));
                }
            }
        }
        foreach (var point in output)
        {
            var j = 0;
            var sameFound = false;
            while (!sameFound && j < target.Count)
            {
                sameFound = V3.Distance(point, target[j]) <= 1e-6;
                j++;
            }
            if (!sameFound) target.Add(point);
        }
    }

    private static bool TypeR1(in Tri t1, in Tri t2)
    {
        var (p1, q1, r1, p2, r2) = (t1.A, t1.B, t1.C, t2.A, t2.C);
        if (Orient2D(r2, p2, q1) >= 0)
        {
            if (Orient2D(r2, p1, q1) >= 0)
            {
                if (Orient2D(p1, p2, q1) >= 0) return true;
                if (Orient2D(p1, p2, r1) >= 0 && Orient2D(q1, r1, p2) >= 0) return true;
            }
        }
        else
        {
            if (Orient2D(r2, p2, r1) >= 0 && Orient2D(q1, r1, r2) >= 0 && Orient2D(p1, p2, r1) >= 0) return true;
        }
        return false;
    }

    private static bool TypeR2(in Tri t1, in Tri t2)
    {
        var (p1, q1, r1, p2, q2, r2) = (t1.A, t1.B, t1.C, t2.A, t2.B, t2.C);
        if (Orient2D(r2, p2, q1) >= 0)
        {
            if (Orient2D(q2, r2, q1) >= 0)
            {
                if (Orient2D(p1, p2, q1) >= 0)
                {
                    if (Orient2D(p1, q2, q1) <= 0) return true;
                }
                else
                {
                    if (Orient2D(p1, p2, r1) >= 0 && Orient2D(r2, p2, r1) <= 0) return true;
                }
            }
            else
            {
                if (Orient2D(p1, q2, q1) <= 0 && Orient2D(q2, r2, r1) >= 0 && Orient2D(q1, r1, q2) >= 0) return true;
            }
        }
        else
        {
            if (Orient2D(r2, p2, r1) >= 0)
            {
                if (Orient2D(q1, r1, r2) >= 0)
                {
                    if (Orient2D(r1, p1, p2) >= 0) return true;
                }
                else
                {
                    if (Orient2D(q1, r1, q2) >= 0 && Orient2D(q2, r2, r1) >= 0) return true;
                }
            }
        }
        return false;
    }

    private static bool CoplanarIntersect(ref Tri t1, ref Tri t2, List<V3>? target)
    {
        var normal = t1.Normal();
        normal = V3.Normalize(normal);
        var u = V3.Normalize(t1.A - t1.B);
        var v = V3.Cross(normal, u);

        // move the basis to t1.a
        u = u + t1.A;
        v = v + t1.A;
        normal = normal + t1.A;

        var matrix = Matrix4.Of(t1.A.X, u.X, v.X, normal.X, t1.A.Y, u.Y, v.Y, normal.Y, t1.A.Z, u.Z, v.Z, normal.Z, 1, 1, 1, 1);
        var affine = Matrix4.Of(0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1);
        matrix.Invert();                       // upstream ignores whether it could
        matrix = affine * matrix;

        matrix.Apply(ref t1.A); matrix.Apply(ref t1.B); matrix.Apply(ref t1.C);
        matrix.Apply(ref t2.A); matrix.Apply(ref t2.B); matrix.Apply(ref t2.C);

        MakeCounterClockwise(ref t1);
        MakeCounterClockwise(ref t2);

        var p1 = t1.A;
        var (p2, q2, r2) = (t2.A, t2.B, t2.C);
        var oP2Q2 = Orient2D(p2, q2, p1);
        var oQ2R2 = Orient2D(q2, r2, p1);
        var oR2P2 = Orient2D(r2, p2, p1);

        bool intersecting;
        if (oP2Q2 >= 0)
        {
            if (oQ2R2 >= 0)
            {
                if (oR2P2 >= 0) intersecting = true;                                      // + + +
                else intersecting = TypeR1(t1, t2);                                       // + + -
            }
            else
            {
                if (oR2P2 >= 0) { PermuteRight(ref t2); intersecting = TypeR1(t1, t2); } // + - +
                else intersecting = TypeR2(t1, t2);                                       // + - -
            }
        }
        else
        {
            if (oQ2R2 >= 0)
            {
                if (oR2P2 >= 0) { PermuteLeft(ref t2); intersecting = TypeR1(t1, t2); }  // - + +
                else { PermuteLeft(ref t2); intersecting = TypeR2(t1, t2); }            // - + -
            }
            else
            {
                if (oR2P2 >= 0) { PermuteRight(ref t2); intersecting = TypeR2(t1, t2); } // - - +
                else return false;                                                        // - - -: upstream says the triangles should not be flat
            }
        }

        if (intersecting && target is not null)
        {
            ClipTriangle(t1, t2, target);
            matrix.Invert();
            for (var i = 0; i < target.Count; i++)
            {
                var p = target[i];
                matrix.Apply(ref p);
                target[i] = p;
            }
        }
        return intersecting;
    }

    private static float PolygonArea(List<V3> polygon)
    {
        if (polygon.Count < 3) return 0.0f;
        var normal = new V3(0, 0, 0);
        for (var i = 0; i < polygon.Count; i++)
        {
            var p1 = polygon[i];
            var p2 = polygon[(i + 1) % polygon.Count];
            normal = normal + V3.Cross(p1, p2);
        }
        return V3.Magnitude(normal) / 2.0f;
    }

    /// <summary>Whether two triangles of the layout overlap over an area (touching along an edge or at a point, or an overlap under 1e-10, does not count).</summary>
    public static bool TriangleTriangle(V2 p1, V2 q1, V2 r1, V2 p2, V2 q2, V2 r2)
    {
        var t1 = new Tri(p1, q1, r1);
        var t2 = new Tri(p2, q2, r2);
        if (IsDegenerate(t1) || IsDegenerate(t2)) return false;

        var o1a = Orient3D(t2.A, t2.B, t2.C, t1.A);
        var o1b = Orient3D(t2.A, t2.B, t2.C, t1.B);
        var o1c = Orient3D(t2.A, t2.B, t2.C, t1.C);

        var intersections = new List<V3>();
        bool intersects;
        if (o1a == o1b && o1a == o1c) intersects = o1a == 0 && CoplanarIntersect(ref t1, ref t2, intersections);
        else intersects = CrossIntersect(ref t1, ref t2, o1a, o1b, o1c, intersections);

        if (intersects)
        {
            var area = PolygonArea(intersections);
            if (area < 1e-10f || !float.IsFinite(area)) return false;
        }
        return intersects;
    }
}
