import numpy as np


def _bisect_segment(p0, p1, max_seg_length):
    """Bisect a single segment recursively until all sub-segments have length <= max_seg_length."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    seg_len = np.sqrt(np.sum((p1 - p0) ** 2))

    if seg_len <= max_seg_length:
        return [p0]

    mid = (p0 + p1) * 0.5
    left_pts = _bisect_segment(p0, mid, max_seg_length)
    right_pts = _bisect_segment(mid, p1, max_seg_length)

    return left_pts + right_pts


def subdivide_contour(pts, max_seg_length=None, n_subdiv=None, bisect_level=None):
    """Subdivide contour segments into smaller segments to increase triangulation density.

    Supports three strategies (highest priority first):
    1. max_seg_length:  maximum segment length -- uses bisection recursively
                        until all sub-segments have length <= max_seg_length.
    2. bisect_level:    bisection level -- each segment is bisected 'level' times,
                        yielding 2^level sub-segments.
    3. n_subdiv:        each segment is split into n_subdiv equal segments.

    Parameters:
        pts:              contour points (numpy array), last point equals first point (closed)
        max_seg_length:   max segment length (enables bisection, recommended)
        bisect_level:     number of bisection levels (0 = no subdivision)
        n_subdiv:         number of equal segments per segment

    Returns: subdivided contour points (numpy array)
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts

    if max_seg_length is not None and max_seg_length > 0:
        new_pts = []
        for i in range(n - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            seg_pts = _bisect_segment(p0, p1, max_seg_length)
            new_pts.extend(seg_pts)

        new_pts.append(new_pts[0])
        return np.array(new_pts, dtype=np.float32)

    if bisect_level is not None and bisect_level > 0:
        level = int(bisect_level)
        n_segments = 1 << level
        new_pts = []
        for i in range(n - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            for k in range(n_segments):
                t = k / n_segments
                new_pts.append(p0 * (1 - t) + p1 * t)

        new_pts.append(new_pts[0])
        return np.array(new_pts, dtype=np.float32)

    if n_subdiv is not None and n_subdiv > 1:
        n_splits = int(n_subdiv)
        new_pts = []
        for i in range(n - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            for k in range(n_splits):
                t = k / n_splits
                new_pts.append(p0 * (1 - t) + p1 * t)

        new_pts.append(new_pts[0])
        return np.array(new_pts, dtype=np.float32)

    return pts.astype(np.float32)
