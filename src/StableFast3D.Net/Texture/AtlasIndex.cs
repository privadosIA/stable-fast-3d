using System.Collections.Concurrent;

namespace StableFast3D.Texture;

/// <summary>
/// <c>UVUnwrapper.assign_faces_uv_to_atlas_index</c> (uv_unwrapper/csrc/unwrapper.cpp and bvh.cpp): the slot of the atlas of every triangle. A triangle starts in the slot of
/// the cube face it was projected on (0 to 5); one that overlaps another of its slot in the UV layout and lies behind it (along the face's axis) moves 6 slots on, and the
/// same test is made once more among the moved ones (the last slot is 12). The overlap tests (<see cref="TriangleOverlap"/>) and the BVH that prunes them are upstream's, since
/// the box test is not exact and what it prunes depends on the shape of the tree.
/// </summary>
public static class AtlasIndex
{
    private const int Bins = 8;
    private const float FltMax = float.MaxValue;
    private const float FltMin = 1.17549435e-38f;       // std::numeric_limits<float>::min()

    private struct Aabb
    {
        public V2 Min, Max;

        public static Aabb Empty => new() { Min = new V2(FltMax, FltMax), Max = new V2(FltMin, FltMin) };

        public void Grow(V2 p)
        {
            Min.X = Math.Min(Min.X, p.X); Min.Y = Math.Min(Min.Y, p.Y);
            Max.X = Math.Max(Max.X, p.X); Max.Y = Math.Max(Max.Y, p.Y);
        }

        public readonly float Area => (Max.X - Min.X) * (Max.Y - Min.Y);
    }

    private struct UvTri
    {
        public V2 V0, V1, V2_, Centroid;

        public readonly bool SameAs(in UvTri o) => V0.SameAs(o.V0) && V1.SameAs(o.V1) && V2_.SameAs(o.V2_);
    }

    private struct Node
    {
        public Aabb Box;
        public int Start, End, Left, Right;
        public readonly bool IsLeaf => Left == -1 && Right == -1;
    }

    /// <summary>The BVH of the triangles of one slot: <c>Intersect</c> gives every triangle (by its index in the whole mesh) that overlaps a given one.</summary>
    private sealed class Bvh
    {
        private readonly UvTri[] _tri;
        private readonly int[] _actual;
        private readonly int[] _order;
        private readonly Node[] _nodes;

        public bool Empty { get; }

        public Bvh() { _tri = []; _actual = []; _order = []; _nodes = []; Empty = true; }

        public Bvh(UvTri[] triangles, int[] actualIndex)
        {
            _tri = triangles;
            _actual = actualIndex;
            var n = triangles.Length;
            _order = Enumerable.Range(0, n).ToArray();
            _nodes = new Node[n * 2 + 64];
            var nodesUsed = 2;
            _nodes[0].Start = 0; _nodes[0].End = n;
            var centroids = Aabb.Empty;
            UpdateNodeBounds(0, ref centroids);
            Subdivide(0, ref nodesUsed, centroids);
        }

        private void UpdateNodeBounds(int nodeIdx, ref Aabb centroids)
        {
            ref var node = ref _nodes[nodeIdx];
            node.Box = Aabb.Empty;
            centroids = Aabb.Empty;
            for (var i = node.Start; i < node.End; i++)
            {
                ref readonly var t = ref _tri[_order[i]];
                node.Box.Grow(t.V0); node.Box.Grow(t.V1); node.Box.Grow(t.V2_);
                centroids.Grow(t.Centroid);
            }
        }

        private static float Axis(V2 v, int axis) => axis == 0 ? v.X : v.Y;

        private void Subdivide(int rootIdx, ref int nodePtr, Aabb rootCentroids)
        {
            var queue = new Queue<(int Node, Aabb Centroids)>();
            queue.Enqueue((rootIdx, rootCentroids));
            Aabb scratch = default;
            while (queue.Count > 0)
            {
                var (nodeIdx, centroids) = queue.Dequeue();
                int axis = 0, splitPos = 0;
                var cost = FindBestSplitPlane(nodeIdx, ref axis, ref splitPos, centroids);
                ref var node = ref _nodes[nodeIdx];
                if (cost >= (node.End - node.Start) * node.Box.Area) { node.Left = node.Right = -1; continue; }

                int i = node.Start, j = node.End - 1;
                var cmin = Axis(centroids.Min, axis);
                var cmax = Axis(centroids.Max, axis);
                var scale = Bins / (cmax - cmin);
                while (i <= j)
                {
                    var bin = Math.Min(Bins - 1, (int)((Axis(_tri[_order[i]].Centroid, axis) - cmin) * scale));
                    if (bin < splitPos) i++;
                    else { (_order[i], _order[j]) = (_order[j], _order[i]); j--; }
                }
                var leftCount = i - node.Start;
                if (leftCount == 0 || leftCount == node.End - node.Start) { node.Left = node.Right = -1; continue; }

                var mid = i;
                var left = nodePtr++;
                var right = nodePtr++;
                _nodes[left].Start = node.Start; _nodes[left].End = mid;
                _nodes[right].Start = mid; _nodes[right].End = node.End;
                node.Left = left;
                node.Right = right;

                UpdateNodeBounds(left, ref scratch);
                queue.Enqueue((left, scratch));
                UpdateNodeBounds(right, ref scratch);
                queue.Enqueue((right, scratch));
            }
        }

        // upstream's SSE variant: the right-hand boxes take bin 6 - i while their counts take bin 7 - i (kept), the empty box is (FLT_MAX, FLT_MIN)
        private float FindBestSplitPlane(int nodeIdx, ref int bestAxis, ref int bestPos, in Aabb centroids)
        {
            var node = _nodes[nodeIdx];
            var bestCost = FltMax;
            for (var axis = 0; axis < 2; axis++)
            {
                var boundsMin = Axis(centroids.Min, axis);
                var boundsMax = Axis(centroids.Max, axis);
                if (boundsMin == boundsMax || boundsMax - boundsMin < 1e-8f) continue;
                var scale = Bins / (boundsMax - boundsMin);

                Span<int> count = stackalloc int[Bins];
                Span<float> minX = stackalloc float[Bins], minY = stackalloc float[Bins], maxX = stackalloc float[Bins], maxY = stackalloc float[Bins];
                for (var b = 0; b < Bins; b++) { minX[b] = minY[b] = FltMax; maxX[b] = maxY[b] = FltMin; }
                for (var i = node.Start; i < node.End; i++)
                {
                    ref readonly var t = ref _tri[_order[i]];
                    var bin = Math.Min(Bins - 1, (int)((Axis(t.Centroid, axis) - boundsMin) * scale));
                    count[bin]++;
                    minX[bin] = Math.Min(Math.Min(Math.Min(minX[bin], t.V0.X), t.V1.X), t.V2_.X);
                    maxX[bin] = Math.Max(Math.Max(Math.Max(maxX[bin], t.V0.X), t.V1.X), t.V2_.X);
                    minY[bin] = Math.Min(Math.Min(Math.Min(minY[bin], t.V0.Y), t.V1.Y), t.V2_.Y);
                    maxY[bin] = Math.Max(Math.Max(Math.Max(maxY[bin], t.V0.Y), t.V1.Y), t.V2_.Y);
                }

                Span<float> leftCountArea = stackalloc float[Bins - 1], rightCountArea = stackalloc float[Bins - 1];
                int leftSum = 0, rightSum = 0;
                float leftMinX = FltMax, leftMinY = FltMax, leftMaxX = FltMin, leftMaxY = FltMin;
                float rightMinX = FltMax, rightMinY = FltMax, rightMaxX = FltMin, rightMaxY = FltMin;
                for (var i = 0; i < Bins - 1; i++)
                {
                    leftSum += count[i];
                    rightSum += count[Bins - 1 - i];
                    leftMinX = Math.Min(leftMinX, minX[i]); leftMinY = Math.Min(leftMinY, minY[i]);
                    leftMaxX = Math.Max(leftMaxX, maxX[i]); leftMaxY = Math.Max(leftMaxY, maxY[i]);
                    rightMinX = Math.Min(rightMinX, minX[Bins - 2 - i]); rightMinY = Math.Min(rightMinY, minY[Bins - 2 - i]);
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

        public List<int> Intersect(in UvTri query)
        {
            var found = new List<int>();
            Span<int> stack = stackalloc int[64];
            var size = 0;
            stack[size++] = 0;
            while (size > 0)
            {
                var node = _nodes[stack[--size]];
                if (node.IsLeaf)
                {
                    for (var i = node.Start; i < node.End; i++)
                    {
                        ref readonly var tri = ref _tri[_order[i]];
                        if (tri.SameAs(query)) continue;
                        if (TriangleOverlap.TriangleTriangle(query.V0, query.V1, query.V2_, tri.V0, tri.V1, tri.V2_)) found.Add(_actual[_order[i]]);
                    }
                }
                else
                {
                    if (TriangleOverlap.TriangleAabb(_nodes[node.Right].Box.Min, _nodes[node.Right].Box.Max, query.V0, query.V1, query.V2_))
                    {
                        if (size >= 64) throw new InvalidOperationException("Node stack overflow");
                        stack[size++] = node.Right;
                    }
                    if (TriangleOverlap.TriangleAabb(_nodes[node.Left].Box.Min, _nodes[node.Left].Box.Max, query.V0, query.V1, query.V2_))
                    {
                        if (size >= 64) throw new InvalidOperationException("Node stack overflow");
                        stack[size++] = node.Left;
                    }
                }
            }
            return found;
        }
    }

    private static Bvh[] CreateBvhs(UvTri[] triangles, SortedSet<int>[] perFace, int start, int end)
    {
        var bvhs = new Bvh[end - start];
        Parallel.For(start, end, i =>
        {
            var members = perFace[i].ToArray();                        // ascending, as std::set iterates
            bvhs[i - start] = members.Length == 0 ? new Bvh() : new Bvh(members.Select(m => triangles[m]).ToArray(), members);
        });
        return bvhs;
    }

    private static void IntersectionCheck(Bvh[] bvhs, UvTri[] triangles, V3[] centroids3d, long[] assign, int offset, SortedSet<int>[] perFace)
    {
        var pairs = new ConcurrentBag<(int, int)>();
        Parallel.For(0, assign.Length, i =>
        {
            if (assign[i] < offset) return;
            var bvh = bvhs[assign[i] - offset];
            if (bvh.Empty) return;
            foreach (var other in bvh.Intersect(triangles[i]))
                if (other != i) pairs.Add(i < other ? (i, other) : (other, i));
        });

        // of every pair, the one that is behind (along the axis of its slot) is the second; only the set of the seconds matters
        var seconds = new SortedSet<int>();
        foreach (var (a, b) in pairs)
        {
            int first = a, second = b;
            var slot = (int)assign[first];
            var norm = slot % 6;
            var axis = norm < 2 ? 0 : norm < 4 ? 1 : 2;
            var useMax = slot % 2 == 1;
            float PosOf(int tri) => axis == 0 ? centroids3d[tri].X : axis == 1 ? centroids3d[tri].Y : centroids3d[tri].Z;
            float posA = PosOf(first), posB = PosOf(second);
            if (useMax) { if (posA < posB) (first, second) = (second, first); }
            else { if (posA > posB) (first, second) = (second, first); }
            seconds.Add(second);
        }

        foreach (var idx in seconds)
        {
            var from = (int)assign[idx];
            var to = Math.Clamp(from + 6, 0, 12);
            assign[idx] = to;
            perFace[from].Remove(idx);
            perFace[to].Add(idx);
        }
    }

    /// <param name="positions">(vertices x 3) floats: the aligned positions</param>
    /// <param name="triangles">(faces x 3) vertex indices</param>
    /// <param name="faceUv">(faces x 3 x 2) floats: the UV of each corner</param>
    /// <param name="faceIndex">the cube face (0 to 5) each triangle was projected on</param>
    /// <returns>the atlas slot (0 to 12) of each triangle</returns>
    public static long[] Assign(float[] positions, long[] triangles, float[] faceUv, long[] faceIndex)
    {
        var faces = faceIndex.Length;
        var assign = (long[])faceIndex.Clone();
        var uvTris = new UvTri[faces];
        var centroids3d = new V3[faces];
        var perFace = Enumerable.Range(0, 13).Select(_ => new SortedSet<int>()).ToArray();
        for (var i = 0; i < faces; i++)
        {
            V2 Uv(int k) => new(faceUv[(i * 3 + k) * 2], faceUv[(i * 3 + k) * 2 + 1]);
            V3 P(int k) { var v = (int)triangles[i * 3 + k]; return new V3(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]); }
            var (a, b, c) = (Uv(0), Uv(1), Uv(2));
            uvTris[i] = new UvTri { V0 = a, V1 = b, V2_ = c, Centroid = new V2((a.X + b.X + c.X) / 3, (a.Y + b.Y + c.Y) / 3) };
            var (p0, p1, p2) = (P(0), P(1), P(2));
            centroids3d[i] = new V3((p0.X + p1.X + p2.X) / 3, (p0.Y + p1.Y + p2.Y) / 3, (p0.Z + p1.Z + p2.Z) / 3);
            perFace[faceIndex[i]].Add(i);
        }

        var bvhs = CreateBvhs(uvTris, perFace, 0, 6);
        IntersectionCheck(bvhs, uvTris, centroids3d, assign, 0, perFace);
        var moved = CreateBvhs(uvTris, perFace, 6, 12);
        IntersectionCheck(moved, uvTris, centroids3d, assign, 6, perFace);
        return assign;
    }
}
