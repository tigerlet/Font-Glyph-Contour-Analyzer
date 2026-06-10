import numpy as np


def contour_sign(pts):
    """Polygon winding direction (using shoelace formula).
    TrueType spec: outer contour CW -> sign < 0; inner contour CCW -> sign > 0.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def is_outer_contour(pts):
    """TrueType spec: outer contour is clockwise (sign < 0)."""
    return contour_sign(pts) < 0


def fix_contour(pts):
    """Contour repair: remove adjacent duplicate points, ensure closure. Keep point order!"""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return pts
    
    diffs = np.diff(pts, axis=0)
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(diffs != 0, axis=1)
    pts = pts[keep]
    
    if len(pts) >= 3 and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def point_in_poly(pt, poly):
    """Ray-casting test for point inside polygon (independent of shapely)."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    for i in range(n - 1):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[i + 1][0]), float(poly[i + 1][1])
        if ((yi > y) != (yj > y)):
            if abs(yj - yi) < 1e-12:
                continue
            x_intersect = (y - yi) * (xj - xi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
    return inside
