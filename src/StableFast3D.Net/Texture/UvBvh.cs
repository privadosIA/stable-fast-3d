namespace StableFast3D.Texture;

/// <summary>
/// The BVH of <c>texture_baker</c> over the triangles of a UV layout, built the way upstream builds it so that a pixel on the shared edge of two triangles
/// finds the same one first (the traversal order decides there): a breadth-first build with a surface-area heuristic over 8 bins. Upstream runs the SSE variant
/// of the heuristic on x86, whose right-hand areas take the box of the bin one to the left of the one its count comes from (a quirk, kept here on purpose), and the
/// node bounds are recomputed by the plain loop right after the SSE ones. float32 throughout, no fused multiply-add (upstream is built for generic x86-64).
/// </summary>
public sealed class UvBvh
{
    private const int Bins = 8;
    private const float FltMax = float.MaxValue;
    private const float FltMin = 1.17549435e-38f;          // std::numeric_limits<float>::min(): upstream's "empty box" maximum (the smallest positive normal)
    private const int MaxStack = 64;

    private struct Aabb
    {
        public float MinX, MinY, MaxX, MaxY;

        public static Aabb Empty => new() { MinX = FltMax, MinY = FltMax, MaxX = FltMin, MaxY = FltMin };

        public void Grow(float x, float y)
        {
            MinX = Math.Min(MinX, x); MinY = Math.Min(MinY, y);
            MaxX = Math.Max(MaxX, x); MaxY = Math.Max(MaxY, y);
        }

        public void Grow(in Aabb b)
        {
            if (b.MinX == FltMax) return;
            Grow(b.MinX, b.MinY);
            Grow(b.MaxX, b.MaxY);
        }

        public readonly bool Contains(float x, float y) => x >= MinX && x <= MaxX && y >= MinY && y <= MaxY;

        public readonly float Area => (MaxX - MinX) * (MaxY - MinY);
    }

    private struct Node
    {
        public Aabb Box;
        public int Start, End, Left, Right;
        public readonly bool IsLeaf => Left == -1 && Right == -1;
    }

    private struct Tri
    {
        public float V0x, V0y, V1x, V1y, V2x, V2y, Cx, Cy;
        public int Index;
    }

    private readonly Tri[] _triangles;
    private readonly int[] _order;
    private readonly List<Node> _nodes;

    /// <param name="uv">(vertices x 2) floats</param>
    /// <param name="indices">(triangles x 3) vertex indices</param>
    public UvBvh(float[] uv, int[] indices)
    {
        var count = indices.Length / 3;
        _triangles = new Tri[count];
        for (var i = 0; i < count; i++)
        {
            int a = indices[i * 3], b = indices[i * 3 + 1], c = indices[i * 3 + 2];
            float v0x = uv[a * 2], v0y = uv[a * 2 + 1], v1x = uv[b * 2], v1y = uv[b * 2 + 1], v2x = uv[c * 2], v2y = uv[c * 2 + 1];
            _triangles[i] = new Tri
            {
                V0x = v0x, V0y = v0y, V1x = v1x, V1y = v1y, V2x = v2x, V2y = v2y, Index = i,
                Cx = (v0x + v1x + v2x) * 0.3333f, Cy = (v0y + v1y + v2y) * 0.3333f,   // upstream's 0.3333, not 1/3
            };
        }
        _order = Enumerable.Range(0, count).ToArray();
        _nodes = new List<Node>(count * 2 + 1);
        Build();
    }

    public int NodeCount => _nodes.Count;

    private static float Axis(float x, float y, int axis) => axis == 0 ? x : y;

    private void Build()
    {
        if (_triangles.Length == 0) { _nodes.Add(new Node { Left = -1, Right = -1 }); return; }
        _nodes.Add(default);
        var queue = new Queue<(int Node, int Start, int End)>();
        queue.Enqueue((0, 0, _triangles.Length));
        while (queue.Count > 0)
        {
            var (nodeIdx, start, end) = queue.Dequeue();
            var node = _nodes[nodeIdx];
            node.Start = start;
            node.End = end;

            // the bounds of the node and of the centroids, by the plain loop
            node.Box = Aabb.Empty;
            var centroids = Aabb.Empty;
            for (var i = start; i < end; i++)
            {
                ref readonly var t = ref _triangles[_order[i]];
                node.Box.Grow(t.V0x, t.V0y); node.Box.Grow(t.V1x, t.V1y); node.Box.Grow(t.V2x, t.V2y);
                centroids.Grow(t.Cx, t.Cy);
            }

            int bestAxis = 0, bestPos = 0;
            var splitCost = FindBestSplitPlane(start, end, centroids, ref bestAxis, ref bestPos);
            var noSplitCost = (end - start) * node.Box.Area;
            if (splitCost >= noSplitCost) { node.Left = node.Right = -1; _nodes[nodeIdx] = node; continue; }

            var cmin = bestAxis == 0 ? centroids.MinX : centroids.MinY;
            var cmax = bestAxis == 0 ? centroids.MaxX : centroids.MaxY;
            var scale = Bins / (cmax - cmin);
            int i2 = start, j = end - 1;
            while (i2 <= j)
            {
                ref readonly var t = ref _triangles[_order[i2]];
                var bin = Math.Min(Bins - 1, (int)((Axis(t.Cx, t.Cy, bestAxis) - cmin) * scale));
                if (bin < bestPos) i2++;
                else { (_order[i2], _order[j]) = (_order[j], _order[i2]); j--; }
            }
            var leftCount = i2 - start;
            if (leftCount == 0 || leftCount == end - start) { node.Left = node.Right = -1; _nodes[nodeIdx] = node; continue; }

            var mid = i2;
            node.Left = _nodes.Count;
            _nodes.Add(default);
            queue.Enqueue((node.Left, start, mid));
            node.Right = _nodes.Count;
            _nodes.Add(default);
            queue.Enqueue((node.Right, mid, end));
            _nodes[nodeIdx] = node;
        }
    }

    /// <summary>The cost of the best split of a node (float.MaxValue when its centroids have no extent), through upstream's SSE variant (see the class summary).</summary>
    private float FindBestSplitPlane(int start, int end, in Aabb centroids, ref int bestAxis, ref int bestPos)
    {
        var bestCost = float.MaxValue;
        for (var axis = 0; axis < 2; axis++)
        {
            var boundsMin = axis == 0 ? centroids.MinX : centroids.MinY;
            var boundsMax = axis == 0 ? centroids.MaxX : centroids.MaxY;
            if (boundsMin == boundsMax) continue;
            var scale = Bins / (boundsMax - boundsMin);

            // per bin: the count and the box (lane 3 is x, lane 2 is y; 1e30 / -1e30 while the bin is empty)
            Span<int> count = stackalloc int[Bins];
            Span<float> minX = stackalloc float[Bins], minY = stackalloc float[Bins], maxX = stackalloc float[Bins], maxY = stackalloc float[Bins];
            for (var b = 0; b < Bins; b++) { minX[b] = minY[b] = 1e30f; maxX[b] = maxY[b] = -1e30f; }
            for (var i = start; i < end; i++)
            {
                ref readonly var t = ref _triangles[_order[i]];
                var bin = Math.Min(Bins - 1, (int)((Axis(t.Cx, t.Cy, axis) - boundsMin) * scale));
                count[bin]++;
                minX[bin] = Math.Min(Math.Min(Math.Min(minX[bin], t.V0x), t.V1x), t.V2x);
                maxX[bin] = Math.Max(Math.Max(Math.Max(maxX[bin], t.V0x), t.V1x), t.V2x);
                minY[bin] = Math.Min(Math.Min(Math.Min(minY[bin], t.V0y), t.V1y), t.V2y);
                maxY[bin] = Math.Max(Math.Max(Math.Max(maxY[bin], t.V0y), t.V1y), t.V2y);
            }

            Span<float> leftCountArea = stackalloc float[Bins - 1], rightCountArea = stackalloc float[Bins - 1];
            int leftSum = 0, rightSum = 0;
            float leftMinX = 1e30f, leftMinY = 1e30f, leftMaxX = -1e30f, leftMaxY = -1e30f;
            float rightMinX = 1e30f, rightMinY = 1e30f, rightMaxX = -1e30f, rightMaxY = -1e30f;
            for (var i = 0; i < Bins - 1; i++)
            {
                leftSum += count[i];
                rightSum += count[Bins - 1 - i];                       // the count of bin 7 - i ...
                leftMinX = Math.Min(leftMinX, minX[i]); leftMinY = Math.Min(leftMinY, minY[i]);
                leftMaxX = Math.Max(leftMaxX, maxX[i]); leftMaxY = Math.Max(leftMaxY, maxY[i]);
                rightMinX = Math.Min(rightMinX, minX[Bins - 2 - i]); rightMinY = Math.Min(rightMinY, minY[Bins - 2 - i]);   // ... with the box of bin 6 - i
                rightMaxX = Math.Max(rightMaxX, maxX[Bins - 2 - i]); rightMaxY = Math.Max(rightMaxY, maxY[Bins - 2 - i]);
                leftCountArea[i] = leftSum * ((leftMaxY - leftMinY) * (leftMaxX - leftMinX));
                rightCountArea[Bins - 2 - i] = rightSum * ((rightMaxY - rightMinY) * (rightMaxX - rightMinX));
            }

            for (var i = 0; i < Bins - 1; i++)
            {
                var planeCost = leftCountArea[i] + rightCountArea[i];
                if (planeCost < bestCost) { bestAxis = axis; bestPos = i + 1; bestCost = planeCost; }
            }
        }
        return bestCost;
    }

    /// <summary>The first triangle (in traversal order) that holds the point, with its barycentric coordinates; false when none does.</summary>
    public bool Intersect(float px, float py, out float u, out float v, out float w, out int index)
    {
        u = v = w = 0;
        index = -1;
        Span<int> stack = stackalloc int[MaxStack];
        var size = 0;
        stack[size++] = 0;
        while (size > 0)
        {
            var node = _nodes[stack[--size]];
            if (node.IsLeaf)
            {
                for (var i = node.Start; i < node.End; i++)
                {
                    ref readonly var t = ref _triangles[_order[i]];
                    if (Barycentric(px, py, t, out u, out v, out w)) { index = t.Index; return true; }
                }
            }
            else
            {
                if (_nodes[node.Right].Box.Contains(px, py))
                {
                    if (size >= MaxStack) throw new InvalidOperationException("Node stack overflow");
                    stack[size++] = node.Right;
                }
                if (_nodes[node.Left].Box.Contains(px, py))
                {
                    if (size >= MaxStack) throw new InvalidOperationException("Node stack overflow");
                    stack[size++] = node.Left;
                }
            }
        }
        return false;
    }

    private static bool Barycentric(float px, float py, in Tri t, out float u, out float v, out float w)
    {
        float v1v2x = t.V1x - t.V0x, v1v2y = t.V1y - t.V0y;
        float v1v3x = t.V2x - t.V0x, v1v3y = t.V2y - t.V0y;
        float xyv1x = px - t.V0x, xyv1y = py - t.V0y;
        var d00 = v1v2x * v1v2x + v1v2y * v1v2y;
        var d01 = v1v2x * v1v3x + v1v2y * v1v3y;
        var d11 = v1v3x * v1v3x + v1v3y * v1v3y;
        var d20 = xyv1x * v1v2x + xyv1y * v1v2y;
        var d21 = xyv1x * v1v3x + xyv1y * v1v3y;
        var denom = d00 * d11 - d01 * d01;
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0f - v - w;
        return v >= 0.0f && w >= 0.0f && v + w <= 1.0f;
    }
}
