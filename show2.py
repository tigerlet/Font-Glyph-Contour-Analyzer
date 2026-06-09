import os
import sys
import numpy as np
import freetype
from shapely.geometry import Polygon
from shapely.ops import triangulate
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Unified two-level module data structure
# ============================================================
class GlyphModule:
    """Glyph module: outer contour + its direct inner contours (holes), with hierarchy information."""
    def __init__(self):
        self.outer_contour = None
        self.inner_contours = []
        self.module_type = 1
        self.parent_idx = -1
        self.children_indices = []
        self.depth = 0

    def __repr__(self):
        inner_count = len(self.inner_contours)
        n_outer = len(self.outer_contour) if self.outer_contour is not None else 0
        return (f"GlyphModule(type={self.module_type}, outer_pts={n_outer}, inners={inner_count}, "
                f"parent={self.parent_idx}, children={self.children_indices}, depth={self.depth}")


# ============================================================
# Core 1: contour direction calculation + preprocessing fix
# ============================================================
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
    # Remove adjacent duplicate points (preserve order)
    diffs = np.diff(pts, axis=0)
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(diffs != 0, axis=1)
    pts = pts[keep]
    # Close the contour
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


def classify_contours(contours_list, outer_indices=None):
    """Classify outer/inner contours in batch, and match inner contours to their direct parent holes.

    Key improvement: judge ownership by using all vertices of inner contours (rather than a single center point),
    and match each inner contour to the smallest outer contour that directly contains it.
    """
    contour_info = []

    for i, cnt in enumerate(contours_list):
        cnt = fix_contour(np.array(cnt, dtype=np.float32))
        if len(cnt) < 4:
            continue
        sign = contour_sign(cnt)
        area = abs(sign) * 0.5

        contour_info.append({"index": i, "contour": cnt, "area": area, "sign": sign})

    if outer_indices is not None:
        outer_info = [info for info in contour_info if info["index"] in outer_indices]
        inner_info = [info for info in contour_info if info["index"] not in outer_indices]
    else:
        # TrueType spec: outer contours CW (sign < 0), inner contours CCW (sign > 0)
        outer_info = [info for info in contour_info if info["sign"] < 0]
        inner_info = [info for info in contour_info if info["sign"] > 0]

    # Outer contours sorted by area from large to small, useful for finding "smallest container"
    outer_info.sort(key=lambda x: x["area"], reverse=True)
    all_outer = [info["contour"] for info in outer_info]

    inner_dict = {}
    for o_cnt in all_outer:
        inner_dict[id(o_cnt)] = []

    # For each inner contour, find the "smallest outer contour that directly contains it"
    for i_info in inner_info:
        i_cnt = i_info["contour"]
        i_area = i_info["area"]
        vertices = i_cnt[:-1] if len(i_cnt) > 1 else i_cnt
        n_verts = len(vertices)

        # Step 1: find all outer contours that "contain this inner contour"
        # Criterion: all (or vast majority of) vertices of the inner contour are inside the outer
        # Extra check: inner contour area < outer contour area (otherwise it cannot be inside)
        containing = []
        for o_info in outer_info:
            o_cnt = o_info["contour"]
            o_area = o_info["area"]

            # Inner contour area >= outer contour area -> cannot be inside it
            if i_area >= o_area * 0.95:
                continue

            # Check ratio of inner contour vertices that fall inside the outer
            n_inside = 0
            for v in vertices:
                if point_in_poly(v, o_cnt):
                    n_inside += 1
            ratio = n_inside / n_verts if n_verts > 0 else 0

            # Require all vertices inside the outer (or at least 90%, tolerant of some edge cases)
            if ratio >= 0.9:
                containing.append((o_cnt, o_area))

        # Step 2: among qualifying outer contours, pick the one with the smallest area
        # (smallest area = innermost = the direct parent contour)
        if containing:
            containing.sort(key=lambda x: x[1])
            best_outer = containing[0][0]
            inner_dict[id(best_outer)].append(i_cnt)
        else:
            # Fallback: find the outer contour with the highest containment ratio
            best_ratio = 0
            best_outer = None
            for o_info in outer_info:
                o_cnt = o_info["contour"]
                o_area = o_info["area"]
                if i_area >= o_area * 0.95:
                    continue
                n_inside = sum(1 for v in vertices if point_in_poly(v, o_cnt))
                ratio = n_inside / n_verts if n_verts > 0 else 0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_outer = o_cnt
            if best_outer is not None and best_ratio > 0.3:
                inner_dict[id(best_outer)].append(i_cnt)

    return all_outer, inner_dict


# ============================================================
# Core 2: build standardized two-level modules
# ============================================================
def build_modules(outer_list, inner_dict):
    modules = []
    for o_cnt in outer_list:
        mod = GlyphModule()
        mod.outer_contour = o_cnt
        mod.inner_contours = inner_dict.get(id(o_cnt), [])
        mod.module_type = 1 if len(mod.inner_contours) >= 1 else 2
        modules.append(mod)
    return modules


# ============================================================
# Core 2b: build module hierarchy (tree of outer contour membership)
# ============================================================
def _module_contains(parent_mod, child_mod, tol=1e-3):
    """Check whether the outer contour of parent contains the outer contour of child.
    Judged by the ratio of vertices of child's outer contour that fall inside parent's outer contour.
    """
    if parent_mod is None or child_mod is None:
        return False
    p_cnt = np.asarray(parent_mod.outer_contour, dtype=np.float64)
    c_cnt = np.asarray(child_mod.outer_contour, dtype=np.float64)
    if len(p_cnt) < 3 or len(c_cnt) < 3:
        return False
    # Vertices of parent outer contour
    vertices = c_cnt
    n_verts = len(vertices)
    if n_verts == 0:
        return False
    # Ratio of vertices of child outer contour that fall inside parent outer contour
    n_inside = sum(1 for v in vertices if point_in_poly(v, p_cnt))
    return n_inside / n_verts > 0.9


def build_module_hierarchy(modules):
    """Build a tree of module membership based on containment relationships among outer contours.
    Key logic:
    - Modules with larger area may contain modules with smaller area
    - Each module finds the "smallest outer contour that directly contains it" as its parent
    - Modules that cannot be contained by any -> depth=0 (root nodes)

    Sets parent_idx, children_indices, depth on each module object.
    Returns: modules (modified in place)
    """
    n = len(modules)
    if n == 0:
        return modules

    # Compute the area of each module's outer contour for comparison
    areas = []
    for m in modules:
        arr = np.asarray(m.outer_contour, dtype=np.float64)
        areas.append(abs(contour_sign(arr)) * 0.5)

    # For each module, find the parent module with the smallest area that directly contains it
    for i in range(n):
        best_parent = -1
        best_area = float('inf')
        for j in range(n):
            if i == j:
                continue
            # Area check: parent module area must be significantly larger than child
            if areas[j] <= areas[i] * 1.1:
                continue
            # Geometry check: does j's outer contour contain i's outer contour
            if _module_contains(modules[j], modules[i]):
                if areas[j] < best_area:
                    best_area = areas[j]
                    best_parent = j
        modules[i].parent_idx = best_parent
        if best_parent >= 0:
            modules[best_parent].children_indices.append(i)

    # Compute depth for each module (starting from root nodes depth=0)
    def _compute_depth(idx, d):
        modules[idx].depth = d
        for child_idx in modules[idx].children_indices:
            _compute_depth(child_idx, d + 1)

    root_indices = [i for i in range(n) if modules[i].parent_idx < 0]
    for ri in root_indices:
        _compute_depth(ri, 0)

    return modules


def module_hierarchy_tree(modules):
    """Return a text representation of the module hierarchy (for display/debugging)."""
    lines = []
    def _render(idx, depth):
        m = modules[idx]
        indent = "  " * depth
        tname = "With hole" if m.module_type == 1 else "Solid"
        lines.append(f"{indent}└─ Module {idx + 1} ({tname}, holes={len(m.inner_contours)})")
        for ci in m.children_indices:
            _render(ci, depth + 1)

    root_indices = [i for i in range(len(modules)) if modules[i].parent_idx < 0]
    for ri in root_indices:
        _render(ri, 0)
    return "\n".join(lines)


# ============================================================
# Core 3a: contour subdivision (bisection / max segment length, to increase triangulation density)
# ============================================================
def _bisect_segment(p0, p1, max_seg_length):
    """Bisect a single segment recursively until all sub-segments have length <= max_seg_length.
    Returns: a list of all subdivision points starting from p0, excluding p1.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    seg_len = np.sqrt(np.sum((p1 - p0) ** 2))

    if seg_len <= max_seg_length:
        return [p0]

    # Recursive bisection: first process left half [p0, mid], then right half [mid, p1]
    mid = (p0 + p1) * 0.5
    left_pts = _bisect_segment(p0, mid, max_seg_length)
    right_pts = _bisect_segment(mid, p1, max_seg_length)

    # Left half returns [p0, ..., points_before_mid], right half returns [mid, ...]
    # The first element of right_pts is mid, it is not a duplicate - must be preserved
    return left_pts + right_pts


def subdivide_contour(pts, max_seg_length=None, n_subdiv=None, bisect_level=None):
    """Subdivide contour segments into smaller segments to increase triangulation density.

    Supports three strategies (highest priority first):
    1. max_seg_length:  maximum segment length -- uses **bisection** recursively
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

    # —— Strategy 1: bisection by max segment length ——
    if max_seg_length is not None and max_seg_length > 0:
        new_pts = []
        for i in range(n - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            seg_pts = _bisect_segment(p0, p1, max_seg_length)
            # _bisect_segment returns points excluding p1 (segment end), so seg_pts of all segments
            # can be concatenated directly without duplication
            new_pts.extend(seg_pts)

        # Close the contour
        new_pts.append(new_pts[0])
        return np.array(new_pts, dtype=np.float32)

    # —— Strategy 2: bisection by level ——
    if bisect_level is not None and bisect_level > 0:
        level = int(bisect_level)
        n_segments = 1 << level  # 2^level sub-segments
        new_pts = []
        for i in range(n - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            for k in range(n_segments):
                t = k / n_segments
                new_pts.append(p0 * (1 - t) + p1 * t)

        # Close the contour
        new_pts.append(new_pts[0])
        return np.array(new_pts, dtype=np.float32)

    # —— Strategy 3: split each segment into n_subdiv equal segments ——
    if n_subdiv is not None and n_subdiv > 1:
        n_splits = int(n_subdiv)
        new_pts = []
        for i in range(n - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            for k in range(n_splits):
                t = k / n_splits
                new_pts.append(p0 * (1 - t) + p1 * t)

        # Close the contour
        new_pts.append(new_pts[0])
        return np.array(new_pts, dtype=np.float32)

    # Default: no subdivision
    return pts.astype(np.float32)


# ============================================================
# Core 3b: Shapely differential triangulation (with contour subdivision + precise clipping)
# Referenced from tt.py: after triangulation, each triangle is precisely clipped via intersection/difference
# ============================================================
def triangulate_module(mod, max_seg_length=None, bisect_level=None, n_subdiv=None):
    outer = mod.outer_contour
    inners = mod.inner_contours

    try:
        # Subdivide contours to increase triangulation density
        need_subdiv = (
            (max_seg_length is not None and max_seg_length > 0)
            or (bisect_level is not None and bisect_level > 0)
            or (n_subdiv is not None and n_subdiv > 1)
        )
        if need_subdiv:
            outer = subdivide_contour(outer, max_seg_length=max_seg_length,
                                      bisect_level=bisect_level, n_subdiv=n_subdiv)
            if len(inners) > 0:
                inners = [subdivide_contour(ic, max_seg_length=max_seg_length,
                                            bisect_level=bisect_level, n_subdiv=n_subdiv)
                          for ic in inners]

        # Build outer contour polygon & inner contour list
        outer_poly = Polygon(outer.tolist())
        inner_polys = []
        if len(inners) > 0:
            for ic in inners:
                if len(ic) >= 4:
                    inner_polys.append(Polygon(ic.tolist()))

        # Build polygon with holes (for triangulate)
        if inner_polys:
            poly = Polygon(outer.tolist(),
                           holes=[list(hole.exterior.coords) for hole in inner_polys])
        else:
            poly = Polygon(outer.tolist())

        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return {"vertices": np.array([]), "triangles": np.array([])}

        # Unified vertex index table (referenced from tt.py's vmap)
        vert_map = {}
        vertices = []

        def _add_vertex(x, y):
            """Add or look up a vertex, return its index"""
            key = (round(float(x), 4), round(float(y), 4))
            if key not in vert_map:
                vert_map[key] = len(vertices)
                vertices.append([x, y])
            return vert_map[key]

        def _collect_triangles(geom):
            """Collect triangulation results from geometry into global tris"""
            results = []
            for sub_t in triangulate(geom):
                sub_pts = np.array(list(sub_t.exterior.coords)[:-1])
                if len(sub_pts) == 3:
                    idx = [_add_vertex(pt[0], pt[1]) for pt in sub_pts]
                    results.append(idx)
            return results

        all_valid_tris = []

        # Initial triangulation of polygon with holes
        tri_candidates = triangulate(poly)

        for tri in tri_candidates:
            tri_pts = np.array(list(tri.exterior.coords)[:-1])
            if len(tri_pts) != 3:
                continue

            tri_poly = Polygon(tri_pts)

            # === Referenced from tt.py: precise clipping of candidate triangles ===
            # 1) Intersect with outer contour, keep only parts inside the outer contour
            clipped = tri_poly.intersection(outer_poly)

            # 2) Subtract each inner contour (hole)
            for inner_p in inner_polys:
                try:
                    clipped = clipped.difference(inner_p)
                except Exception:
                    pass

            if clipped.is_empty:
                continue

            # 3) Re-triangulate the clipped result
            # Polygon: directly triangulate
            if hasattr(clipped, 'exterior'):
                coords = np.array(list(clipped.exterior.coords)[:-1])
                if len(coords) >= 3:
                    all_valid_tris.extend(_collect_triangles(Polygon(coords)))
            # MultiPolygon: iterate each sub-polygon
            elif hasattr(clipped, 'geoms'):
                for geom in clipped.geoms:
                    if hasattr(geom, 'exterior'):
                        coords = np.array(list(geom.exterior.coords)[:-1])
                        if len(coords) >= 3:
                            all_valid_tris.extend(_collect_triangles(Polygon(coords)))

        # Filter out zero-area triangles
        verts_np = np.array(vertices, dtype=np.float32) if vertices else np.array([], dtype=np.float32)
        final_tris = []
        for t in all_valid_tris:
            if len(t) != 3 or len(verts_np) == 0:
                continue
            pts = verts_np[t]
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
            if area > 1e-4:
                final_tris.append(t)

        return {
            "vertices": verts_np,
            "triangles": np.array(final_tris, dtype=np.int32) if final_tris else np.array([], dtype=np.int32),
        }
    except Exception as e:
        print(f"  Triangulation error: {e}")
        return {"vertices": np.array([]), "triangles": np.array([])}


# ============================================================
# Core 4: multi-module result merging & global optimization
# ============================================================
def merge_modules_results(module_results):
    all_verts = []
    all_tris = []
    offset = 0

    for res in module_results:
        verts = res["vertices"]
        tris = res["triangles"]
        if len(verts) == 0 or len(tris) == 0:
            continue
        all_verts.append(verts)
        all_tris.append(tris + offset)
        offset += len(verts)

    if not all_verts:
        return {"vertices": np.array([]), "triangles": np.array([])}

    final_verts = np.vstack(all_verts)
    final_tris = np.vstack(all_tris)

    unique_verts, idx_map = np.unique(final_verts, axis=0, return_inverse=True)
    unified_tris = idx_map[final_tris]

    return {
        "vertices": unique_verts.astype(np.float32),
        "triangles": unified_tris.astype(np.int32),
    }


# ============================================================
# Core 5: FreeType glyph contour parsing (corrected version)
# Correctly handles quadratic Bezier curves (including implicit points of consecutive off-curve points)
# ============================================================
def _normalize_ttf_points(points, tags):
    """Normalize a contour point sequence according to TrueType spec: insert implicit on-curve points between consecutive off-curve points.
    Returns: normalized (x,y) sequence, tags sequence (ensures head/tail are both on-curve with no non-consecutive off-curve).
    """
    if len(points) == 0:
        return [], []

    pts = [(float(x), float(y)) for (x, y) in points]
    tags = list(tags)
    n = len(pts)

    # If the first/last points are off-curve, insert implicit on-curve between them
    def on_curve(t):
        return bool(t & 1)

    # Make a copy for processing in circular sense: copy the start point to the end to form a circle
    # Find the first on-curve point, start processing from there
    first_on = 0
    for i in range(n):
        if on_curve(tags[i]):
            first_on = i
            break
    # Rearrange: start from first_on
    pts = pts[first_on:] + pts[:first_on]
    tags = tags[first_on:] + tags[:first_on]
    # Close: append starting point at end
    pts.append(pts[0])
    tags.append(tags[0])

    # Handle consecutive off-curve points: insert a midpoint is inserted as implicit on-curve between two consecutive off-curve
    new_pts = [pts[0]]
    new_tags = [tags[0]]
    for i in range(1, len(pts)):
        if not on_curve(tags[i - 1]) and not on_curve(tags[i]):
            mid_x = (pts[i - 1][0] + pts[i][0]) * 0.5
            mid_y = (pts[i - 1][1] + pts[i][1]) * 0.5
            new_pts.append((mid_x, mid_y))
            new_tags.append(1)
        new_pts.append(pts[i])
        new_tags.append(tags[i])

    return new_pts, new_tags


def _sample_quadratic(p0, p1, p2, n_samples=8):
    """Sample quadratic Bezier curve B(t) = (1-t)^2 p0 + 2(1-t)t p1 + t^2 p2.
    Returns: list of sample points, excluding p0, including p2.
    """
    samples = []
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    for k in range(1, n_samples + 1):
        t = k / (n_samples + 1)
        mt = 1 - t
        x = mt * mt * x0 + 2 * mt * t * x1 + t * t * x2
        y = mt * mt * y0 + 2 * mt * t * y1 + t * t * y2
        samples.append((x, y))
    samples.append((x2, y2))
    return samples


def char_to_contours_ft(font_path, char, resolution=200, scale=1 / 64, bezier_samples=8):
    """Read glyph contours from FreeType.
    Correctly parses quadratic Bezier curves (TrueType spec), supporting multiple contours and holes.
    """
    face = freetype.Face(font_path)
    face.set_char_size(resolution * 64)

    char_code = ord(char)
    face.load_char(char_code, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP | freetype.FT_LOAD_NO_HINTING)
    outline = face.glyph.outline

    contours = []
    start = 0
    for end in outline.contours:
        raw_pts = []
        raw_tags = []
        for i in range(start, end + 1):
            x, y = outline.points[i]
            raw_pts.append((x, y))
            raw_tags.append(outline.tags[i])

        # Normalize: handle consecutive off-curve points
        pts_norm, tags_norm = _normalize_ttf_points(raw_pts, raw_tags)
        if len(pts_norm) < 3:
            start = end + 1
            continue

        # Quadratic Bezier sampling following [on-curve, off-curve, on-curve] pattern
        out_pts = []
        i = 0
        n_norm = len(pts_norm)
        while i < n_norm - 1:
            cur = pts_norm[i]
            nxt = pts_norm[i + 1]
            if tags_norm[i + 1] & 1:
                # Line segment: next point is on-curve
                out_pts.append([cur[0] * scale, cur[1] * scale])
                i += 1
            else:
                # Quadratic Bezier: cur (on) -> nxt (off, control) -> next_next (on)
                if i + 2 < n_norm:
                    next_next = pts_norm[i + 2]
                else:
                    next_next = pts_norm[0]
                out_pts.append([cur[0] * scale, cur[1] * scale])
                samples = _sample_quadratic(cur, nxt, next_next, n_samples=bezier_samples)
                for (sx, sy) in samples:
                    out_pts.append([sx * scale, sy * scale])
                i += 2

        # Close the contour
        if out_pts and (out_pts[0][0] != out_pts[-1][0] or out_pts[0][1] != out_pts[-1][1]):
            out_pts.append(out_pts[0])
        if len(out_pts) >= 3:
            contours.append(out_pts)

        start = end + 1

    return contours


# ============================================================
# Unified public entry point
# ============================================================
def glyph_triangulate_all(glyph_contours, outer_indices=None, max_seg_length=None, bisect_level=None, n_subdiv=None):
    outer_list, inner_dict = classify_contours(glyph_contours, outer_indices)
    modules = build_modules(outer_list, inner_dict)
    tri_res_list = [triangulate_module(m, max_seg_length=max_seg_length, bisect_level=bisect_level, n_subdiv=n_subdiv) for m in modules]
    final_mesh = merge_modules_results(tri_res_list)
    return final_mesh, modules


# ============================================================
# Graphical UI
# ============================================================
class GlyphAnalyzerApp:
    DEFAULT_FONT_CANDIDATES = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simkai.ttf",
    ]

    def __init__(self, root):
        self.root = root

        # Chinese font (must be initialized before _build_ui, used during UI initialization)
        self.font_path = self._auto_font_path()
        try:
            self.font_prop = FontProperties(fname=self.font_path)
        except Exception:
            self.font_prop = FontProperties(family="SimHei")

        # Configure matplotlib global Chinese font to avoid missing glyph warnings for Chinese labels
        try:
            import matplotlib
            matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        self._build_ui()

        # Initially display a character
        self.char_var.set("国")
        self.root.after(300, self.analyze)

    # --------------------------------------------------------
    # UI construction
    # --------------------------------------------------------
    def _auto_font_path(self):
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simhei.ttf")
        if os.path.exists(local):
            return local
        for p in self.DEFAULT_FONT_CANDIDATES:
            if os.path.exists(p):
                return p
        return self.DEFAULT_FONT_CANDIDATES[0]

    def _build_ui(self):
        self.root.title("Glyph Contour Analyzer")
        self.root.geometry("1400x860")
        self.root.minsize(1100, 700)

        # ============ Top Control Bar ============
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(side="top", fill="x")

        ttk.Label(top, text="Character:").pack(side="left")
        self.char_var = tk.StringVar(value="A")
        char_entry = ttk.Entry(top, textvariable=self.char_var, width=5, font=("Arial", 14))
        char_entry.pack(side="left", padx=(4, 12))
        char_entry.bind("<Return>", lambda _e: self.analyze())

        ttk.Label(top, text="Font file:").pack(side="left")
        self.font_var = tk.StringVar(value=self.font_path)
        ttk.Entry(top, textvariable=self.font_var, width=48).pack(side="left", padx=(4, 6))
        ttk.Button(top, text="...", width=3, command=self._browse_font).pack(side="left", padx=(0, 12))

        ttk.Label(top, text="Resolution:").pack(side="left")
        self.res_var = tk.IntVar(value=200)
        ttk.Spinbox(top, from_=50, to=800, textvariable=self.res_var, width=6).pack(side="left", padx=(4, 12))

        ttk.Label(top, text="Bisect level:").pack(side="left")
        self.bisect_var = tk.IntVar(value=0)
        ttk.Spinbox(top, from_=0, to=6, textvariable=self.bisect_var, width=4).pack(side="left", padx=(4, 6))
        ttk.Label(top, text="(0=off, 1=2segs, 2=4segs, 3=8segs...)").pack(side="left", padx=(0, 12))

        self.show_tri_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Triangulation overlay", variable=self.show_tri_var,
                        command=self._redraw_contour).pack(side="left", padx=(0, 10))
        self.show_label_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Contour labels", variable=self.show_label_var,
                        command=self._redraw_contour).pack(side="left", padx=(0, 10))

        ttk.Button(top, text="-> Analyze", command=self.analyze).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Save Image", command=self.export_image).pack(side="left", padx=(6, 0))

        # ============ Main Area: Two Columns (grid layout, 2:1) ============
        main_wrap = ttk.Frame(self.root)
        main_wrap.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 8))
        main_wrap.columnconfigure(0, weight=2)  # left 2/3
        main_wrap.columnconfigure(1, weight=1)  # right 1/3
        main_wrap.rowconfigure(0, weight=1)

        # -------- Left Panel: Glyph Contour Plot --------
        left_frame = tk.LabelFrame(main_wrap, text="  Glyph Contours  ", padx=6, pady=6,
                                   font=("Arial", 10))
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)

        self.contour_fig = plt.Figure(figsize=(8, 7), dpi=100, facecolor="white")
        self.contour_canvas = FigureCanvasTkAgg(self.contour_fig, master=left_frame)
        canvas_widget = self.contour_canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Toolbar below canvas (separate row)
        tb_frame = ttk.Frame(left_frame)
        tb_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        toolbar1 = NavigationToolbar2Tk(self.contour_canvas, tb_frame)
        toolbar1.update()

        # Draw initial placeholder to ensure the figure is rendered
        self.contour_ax = self.contour_fig.add_subplot(111)
        self.contour_ax.set_aspect("equal")
        self.contour_ax.text(0.5, 0.5, "Enter a character, then click -> Analyze",
                             ha="center", va="center", fontsize=14, color="gray",
                             transform=self.contour_ax.transAxes)
        self.contour_ax.set_xlim(0, 1); self.contour_ax.set_ylim(0, 1)
        self.contour_ax.axis("off")
        self.contour_fig.tight_layout()
        self.contour_canvas.draw()

        # -------- Right Panel: Analysis Results --------
        right_frame = tk.LabelFrame(main_wrap, text="  Analysis Results  ", padx=6, pady=6,
                                    font=("Arial", 10))
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_frame.rowconfigure(2, weight=1)
        right_frame.rowconfigure(4, weight=1)
        right_frame.columnconfigure(0, weight=1)

        # Summary info
        self.summary_label = tk.Label(right_frame, text="(No analysis yet)",
                                      justify="left", anchor="w", font=("Arial", 9),
                                      relief="solid", bd=1, padx=8, pady=6)
        self.summary_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        # Tabs: Contour table + Module table + Module hierarchy + Triangulation plot
        self.nb = ttk.Notebook(right_frame)
        self.nb.grid(row=2, column=0, sticky="nsew", pady=(2, 6))

        tab_c = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab_c, text="Contours")
        tab_m = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab_m, text="Modules")
        tab_h = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab_h, text="Hierarchy")
        tab_t = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab_t, text="Triangulation")

        # Contour table
        tab_c.columnconfigure(0, weight=1)
        tab_c.rowconfigure(0, weight=1)
        cols_c = ("cidx", "ctype", "cpts", "carea", "cdir")
        self.tree_contour = ttk.Treeview(tab_c, columns=cols_c, show="headings", height=5)
        hdrs_c = ["ID", "Type", "Points", "Area", "Direction"]
        widths_c = [50, 60, 55, 80, 100]
        for c, h, w in zip(cols_c, hdrs_c, widths_c):
            self.tree_contour.heading(c, text=h)
            self.tree_contour.column(c, width=w, anchor="center")
        self.tree_contour.grid(row=0, column=0, sticky="nsew")
        ysb_c = ttk.Scrollbar(tab_c, orient="vertical", command=self.tree_contour.yview)
        self.tree_contour.configure(yscroll=ysb_c.set)
        ysb_c.grid(row=0, column=1, sticky="ns")
        self.tree_contour.bind("<<TreeviewSelect>>", self._on_contour_select)

        # Module table
        tab_m.columnconfigure(0, weight=1)
        tab_m.rowconfigure(0, weight=1)
        cols_m = ("midx", "mtype", "mouter", "minner", "mparent", "mdepth", "marea")
        self.tree_module = ttk.Treeview(tab_m, columns=cols_m, show="headings", height=5)
        hdrs_m = ["ID", "Type", "Outer Pts", "Holes", "Parent", "Depth", "Area"]
        widths_m = [50, 70, 65, 55, 60, 50, 70]
        for c, h, w in zip(cols_m, hdrs_m, widths_m):
            self.tree_module.heading(c, text=h)
            self.tree_module.column(c, width=w, anchor="center")
        self.tree_module.grid(row=0, column=0, sticky="nsew")
        ysb_m = ttk.Scrollbar(tab_m, orient="vertical", command=self.tree_module.yview)
        self.tree_module.configure(yscroll=ysb_m.set)
        ysb_m.grid(row=0, column=1, sticky="ns")
        self.tree_module.bind("<<TreeviewSelect>>", self._on_module_select)

        # Module hierarchy tree
        tab_h.columnconfigure(0, weight=1)
        tab_h.rowconfigure(0, weight=1)
        cols_h = ("hidx", "htype", "hchildren", "hchildren_ids", "hinner")
        self.tree_hierarchy = ttk.Treeview(tab_h, columns=cols_h, show="tree headings", height=8)
        hdrs_h = ["ID", "Type", "Children", "Child IDs", "Holes"]
        widths_h = [60, 80, 70, 100, 60]
        for c, h, w in zip(cols_h, hdrs_h, widths_h):
            self.tree_hierarchy.heading(c, text=h)
            self.tree_hierarchy.column(c, width=w, anchor="center")
        self.tree_hierarchy.grid(row=0, column=0, sticky="nsew")
        ysb_h = ttk.Scrollbar(tab_h, orient="vertical", command=self.tree_hierarchy.yview)
        self.tree_hierarchy.configure(yscroll=ysb_h.set)
        ysb_h.grid(row=0, column=1, sticky="ns")
        self.tree_hierarchy.bind("<<TreeviewSelect>>", self._on_hierarchy_select)

        # Triangulation mini plot
        tab_t.columnconfigure(0, weight=1)
        tab_t.rowconfigure(0, weight=1)
        self.result_fig = plt.Figure(figsize=(4, 3), dpi=100, facecolor="white")
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, master=tab_t)
        self.result_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.result_ax = self.result_fig.add_subplot(111)
        self.result_ax.set_aspect("equal"); self.result_ax.axis("off")
        self.result_ax.text(0.5, 0.5, "(No triangulation yet)", ha="center", va="center",
                            fontsize=10, color="gray", transform=self.result_ax.transAxes)
        self.result_fig.tight_layout()
        self.result_canvas.draw()

        # Detail info text
        detail_frame = tk.LabelFrame(right_frame, text="  Details  ", padx=4, pady=4)
        detail_frame.grid(row=4, column=0, sticky="nsew", pady=(4, 0))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        self.detail_text = scrolledtext.ScrolledText(
            detail_frame, height=6, font=("Consolas", 9), wrap="word",
            relief="flat", padx=6, pady=4
        )
        self.detail_text.grid(row=0, column=0, sticky="nsew")
        self.detail_text.insert("1.0", "Click an entry in the Contours or Modules tables above to highlight it in the left panel.")

        # Status bar
        self.status_var = tk.StringVar(value="Ready. Enter a character and click -> Analyze.")
        status = tk.Label(self.root, textvariable=self.status_var, anchor="w",
                          relief="sunken", bd=1, padx=8, pady=2, font=("Arial", 9))
        status.pack(side="bottom", fill="x")

        # ============ Runtime state ============
        self.current_char = None
        self.raw_contours_np = []
        self.contour_info = []
        self.modules = []
        self.mesh = {"vertices": np.array([]), "triangles": np.array([])}
        self.selected_module = -1
        self.selected_contour = -1
        self._contour_to_module = {}

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------
    def _browse_font(self):
        path = filedialog.askopenfilename(
            title="Choose font file",
            filetypes=[("Font files", "*.ttf *.otf *.ttc"), ("All files", "*.*")],
        )
        if path:
            self.font_var.set(path)
            self.font_path = path

    def analyze(self):
        char_text = self.char_var.get().strip()
        if not char_text:
            messagebox.showwarning("Warning", "Please enter at least one character")
            return
        char = char_text[0]
        self.current_char = char

        font_path = self.font_var.get().strip() or self.font_path
        if not os.path.exists(font_path):
            messagebox.showerror("Error", f"Font file not found: {font_path}")
            return

        resolution = int(self.res_var.get() or 200)

        self.status_var.set(f"Analyzing character '{char}'...")
        self.root.update_idletasks()

        try:
            contours = char_to_contours_ft(font_path, char, resolution=resolution)
        except Exception as e:
            messagebox.showerror("Font loading failed", str(e))
            self.status_var.set("Font loading failed")
            return

        if not contours:
            messagebox.showinfo("Info", f"No contour data found for character '{char}'")
            self.status_var.set("Done (no contours)")
            return

        # Compute detailed info for each contour, and normalize direction
        cinfo = []
        normalized = []
        outer_palette = ["#E53935", "#FB8C00", "#F4511E", "#D81B60", "#8E24AA"]
        inner_palette = ["#1E88E5", "#00ACC1", "#43A047", "#3949AB", "#00897B"]
        oi, ii = 0, 0
        for i, cnt in enumerate(contours):
            arr = np.array(cnt, dtype=np.float32)
            s = contour_sign(arr)
            ctype = "outer" if s < 0 else "inner"
            # Normalize outer contours to CW (sign<0), inner contours to CCW (sign>0)
            if ctype == "outer" and s > 0:
                arr = arr[::-1]
                s = -s
            elif ctype == "inner" and s < 0:
                arr = arr[::-1]
                s = -s
            area = abs(s) * 0.5
            color = outer_palette[oi % len(outer_palette)] if ctype == "outer" else inner_palette[ii % len(inner_palette)]
            if ctype == "outer":
                oi += 1
            else:
                ii += 1
            cinfo.append({"idx": i, "type": ctype, "points": len(arr), "area": area, "sign": s, "color": color})
            normalized.append(arr)
        self.raw_contours_np = normalized
        self.contour_info = cinfo

        # Build modules and triangulate (with contour subdivision / bisection)
        try:
            outer_list, inner_dict = classify_contours(contours)
            self.modules = build_modules(outer_list, inner_dict)
            build_module_hierarchy(self.modules)
            bisect_level = int(self.bisect_var.get() or 0)
            tri_res_list = [triangulate_module(m, bisect_level=bisect_level) for m in self.modules]
            self.mesh = merge_modules_results(tri_res_list)
        except Exception as e:
            messagebox.showerror("Analysis failed", str(e))
            self.status_var.set("Analysis failed")
            return

        # Build contour->module mapping for highlighting
        self._contour_to_module = {}
        for mi, mod in enumerate(self.modules):
            outer_np = np.asarray(mod.outer_contour, dtype=np.float32)
            for ci, info in enumerate(cinfo):
                cnt = normalized[ci]
                if cnt.shape == outer_np.shape and np.allclose(cnt, outer_np, atol=1e-3):
                    self._contour_to_module[ci] = mi
                    break

        self.selected_module = -1
        self.selected_contour = -1
        self._redraw_contour()
        self._update_result_panel()
        self._redraw_result_mini()
        self.status_var.set(
            f"Done: '{char}' {len(cinfo)} contours, {len(self.modules)} modules, "
            f"{len(self.mesh['vertices'])} vertices, {len(self.mesh['triangles'])} triangles"
        )

    # --------------------------------------------------------
    # Interaction: selection
    # --------------------------------------------------------
    def _on_contour_select(self, _event):
        sel = self.tree_contour.selection()
        if not sel:
            return
        idx = int(self.tree_contour.item(sel[0], "values")[0])
        self.selected_contour = idx
        # Sync to the corresponding module
        self.selected_module = self._contour_to_module.get(idx, -1)
        self._redraw_contour()
        self._update_detail_for_contour(idx)

    def _on_module_select(self, _event):
        sel = self.tree_module.selection()
        if not sel:
            return
        idx = int(self.tree_module.item(sel[0], "values")[0]) - 1
        self.selected_module = idx
        self._redraw_contour()
        self._update_detail_for_module(idx)

    def _on_hierarchy_select(self, _event):
        sel = self.tree_hierarchy.selection()
        if not sel:
            return
        # Extract index from the tree node (first column)
        values = self.tree_hierarchy.item(sel[0], "values")
        if not values:
            return
        idx = int(values[0]) - 1
        self.selected_module = idx
        self._redraw_contour()
        self._update_detail_for_module(idx)

    # --------------------------------------------------------
    # Drawing: left panel - glyph contours (outer/inner contours with different line styles + direction arrows + labels)
    # --------------------------------------------------------
    def _redraw_contour(self):
        ax = self.contour_ax
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")

        if not self.raw_contours_np:
            title = "Glyph contours" if not self.current_char else f"Glyph: '{self.current_char}'"
            ax.set_title(title, fontsize=14)
            self.contour_canvas.draw()
            return

        all_pts = np.vstack(self.raw_contours_np)

        # Triangulation background (if checked)
        if self.show_tri_var.get() and len(self.mesh["vertices"]) > 0:
            verts = self.mesh["vertices"]
            tris = self.mesh["triangles"]
            patches = []
            for tri in tris:
                pts = verts[tri]
                patches.append(MplPolygon(pts, closed=True))
            pc = PatchCollection(patches, facecolor="#E8F5E9", edgecolor="#90CAF9",
                                 linewidth=0.25, alpha=0.7)
            ax.add_collection(pc)

        # Highlight marking
        hl_outer = None
        hl_inners = []
        if 0 <= self.selected_module < len(self.modules):
            m = self.modules[self.selected_module]
            hl_outer = np.asarray(m.outer_contour, dtype=np.float32)
            hl_inners = [np.asarray(ic, dtype=np.float32) for ic in m.inner_contours]
        elif self.selected_contour >= 0 and self.selected_contour < len(self.raw_contours_np):
            cnt = self.raw_contours_np[self.selected_contour]
            if self.contour_info[self.selected_contour]["type"] == "outer":
                hl_outer = cnt
            else:
                hl_inners = [cnt]

        # Draw inner contours first, then outer contours (outer on top)
        legend_lines = []
        legend_labels = []
        x_min, y_min = np.min(all_pts, axis=0)
        x_max, y_max = np.max(all_pts, axis=0)
        x_range = (x_max - x_min) if x_max > x_min else 1.0
        y_range = (y_max - y_min) if y_max > y_min else 1.0
        char_size = max(x_range, y_range)

        # Inner contours: dashed + cool colors, smaller arrows
        for i, info in enumerate(self.contour_info):
            if info["type"] != "inner":
                continue
            cnt = self.raw_contours_np[i]
            line, = ax.plot(cnt[:, 0], cnt[:, 1], color=info["color"], lw=2.2, ls="--")
            legend_lines.append(line)
            legend_labels.append(f"Contour {i} (inner)")
            self._draw_arrow_and_label(ax, cnt, info, i, char_size, is_inner=True)

        # Outer contours: solid + warm colors, slightly larger arrows
        for i, info in enumerate(self.contour_info):
            if info["type"] != "outer":
                continue
            cnt = self.raw_contours_np[i]
            line, = ax.plot(cnt[:, 0], cnt[:, 1], color=info["color"], lw=2.8, ls="-")
            legend_lines.append(line)
            legend_labels.append(f"Contour {i} (outer)")
            self._draw_arrow_and_label(ax, cnt, info, i, char_size, is_inner=False)

        # Highlight stroke
        if hl_outer is not None:
            ax.plot(hl_outer[:, 0], hl_outer[:, 1], color="#FFEB3B", lw=5, alpha=0.9, zorder=10)
        for ic in hl_inners:
            ax.plot(ic[:, 0], ic[:, 1], color="#FFEB3B", lw=4, alpha=0.9, zorder=10)

        margin = max((x_max - x_min), (y_max - y_min)) * 0.10
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

        # Title and legend
        title = f"Glyph: '{self.current_char}' (TrueType)" if self.current_char else "Glyph Contours"
        ax.set_title(title, fontsize=13, pad=8)
        if legend_lines:
            ax.legend(legend_lines, legend_labels, loc="upper left",
                      bbox_to_anchor=(1.0, 1.0),
                      fontsize=8, framealpha=0.9, ncol=1)

        self.contour_fig.tight_layout()
        self.contour_canvas.draw()

    def _draw_arrow_and_label(self, ax, cnt, info, idx, char_size, is_inner=False):
        """Draw a direction arrow on a longer segment of the contour, and a text label at the first point"""
        n = len(cnt)
        if n < 2:
            return

        # Find the longest segment in the contour (avoid drawing arrows on very short segments, which would be invisible)
        seg_lens = []
        for i in range(n - 1):
            dx = cnt[i + 1, 0] - cnt[i, 0]
            dy = cnt[i + 1, 1] - cnt[i, 1]
            seg_lens.append((dx * dx + dy * dy) ** 0.5)
        if not seg_lens:
            return

        # Find the index of the longest segment
        max_len = max(seg_lens)
        best_idx = seg_lens.index(max_len)
        min_required = char_size * 0.04

        # If the longest segment is still too short, try the median-length segment
        if max_len < min_required:
            sorted_lens = sorted(seg_lens)
            median_len = sorted_lens[len(sorted_lens) // 2]
            if median_len >= min_required:
                best_idx = seg_lens.index(median_len)

        sp = cnt[best_idx]
        ep = cnt[best_idx + 1]
        dx = ep[0] - sp[0]
        dy = ep[1] - sp[1]
        seg_len = (dx * dx + dy * dy) ** 0.5
        if seg_len < 1e-8:
            return

        # Arrow size based on overall character size
        scale = 0.015 if is_inner else 0.025
        hw = char_size * scale
        hl = char_size * scale * 1.6

        # Arrow is drawn inside the segment: start at the middle of the segment, arrow length is part of segment
        # To avoid the arrow going beyond the segment, the arrow starts at 1/4 position pointing toward 3/4 position
        arrow_start_x = sp[0] + dx * 0.25
        arrow_start_y = sp[1] + dy * 0.25
        arrow_dx = dx * 0.5
        arrow_dy = dy * 0.5

        # Ensure arrow does not exceed segment bounds
        arrow_len = (arrow_dx * arrow_dx + arrow_dy * arrow_dy) ** 0.5
        if arrow_len < hl:
            # If the segment is too short, use a shorter arrow
            arrow_dx = dx * 0.3
            arrow_dy = dy * 0.3

        ax.arrow(arrow_start_x, arrow_start_y, arrow_dx, arrow_dy,
                 head_width=hw, head_length=min(hl, arrow_len * 0.6),
                 fc=info["color"], ec=info["color"],
                 length_includes_head=True, alpha=0.95, zorder=8)

        # Text label (placed next to the first point)
        if getattr(self, "show_label_var", None) and self.show_label_var.get():
            label_text = f"#{idx}"
            # If it's an outer contour, add module depth info
            if not is_inner:
                mod_idx = getattr(self, "_contour_to_module", {}).get(idx, -1)
                if mod_idx >= 0 and 0 <= mod_idx < len(getattr(self, "modules", [])):
                    mod = self.modules[mod_idx]
                    depth_indicator = "·" * mod.depth if mod.depth > 0 else ""
                    label_text = f"M{mod_idx + 1}{depth_indicator}"
            ax.annotate(label_text, xy=(cnt[0][0], cnt[0][1]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color=info["color"],
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.2",
                                                     fc="white", ec=info["color"], alpha=0.9))

    # --------------------------------------------------------
    # Drawing: right panel - triangulation plot
    # --------------------------------------------------------
    def _redraw_result_mini(self):
        ax = self.result_ax
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")

        verts = self.mesh["vertices"]
        tris = self.mesh["triangles"]
        if len(verts) == 0 or len(tris) == 0:
            ax.set_title("(No triangulation)", fontsize=10)
            self.result_fig.tight_layout()
            self.result_canvas.draw()
            return

        patches = []
        for tri in tris:
            pts = verts[tri]
            patches.append(MplPolygon(pts, closed=True))
        pc = PatchCollection(patches, facecolor="#C8E6C9", edgecolor="#888888", linewidth=0.25, alpha=0.85)
        ax.add_collection(pc)
        for cnt in self.raw_contours_np:
            ax.plot(cnt[:, 0], cnt[:, 1], "#E53935", lw=1.5)

        x_min, y_min = np.min(verts, axis=0)
        x_max, y_max = np.max(verts, axis=0)
        margin = max((x_max - x_min), (y_max - y_min)) * 0.08
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_title(f"Triangulation ({len(tris)} triangles)", fontsize=10)

        self.result_fig.tight_layout()
        self.result_canvas.draw()

    # --------------------------------------------------------
    # Update right panel
    # --------------------------------------------------------
    def _update_result_panel(self):
        n_outer = sum(1 for c in self.contour_info if c["type"] == "outer")
        n_inner = sum(1 for c in self.contour_info if c["type"] == "inner")
        total_area = sum(c["area"] for c in self.contour_info)
        total_pts = sum(c["points"] for c in self.contour_info)

        summary = (
            f"Char: '{self.current_char}'   "
            f"Contours: {len(self.contour_info)} ({n_outer} outer / {n_inner} inner)\n"
            f"Points: {total_pts}   Total area ~ {total_area:.1f}\n"
            f"Modules: {len(self.modules)}   Triangulation: {len(self.mesh['triangles'])} tri / {len(self.mesh['vertices'])} verts"
        )
        self.summary_label.config(text=summary)

        # Contour table
        for item in self.tree_contour.get_children():
            self.tree_contour.delete(item)
        for c in self.contour_info:
            cdir = "CW" if c["type"] == "outer" else "CCW"
            self.tree_contour.insert(
                "", "end",
                values=(c["idx"], "outer" if c["type"] == "outer" else "inner",
                        c["points"], f"{c['area']:.2f}", cdir),
            )

        # Module table
        for item in self.tree_module.get_children():
            self.tree_module.delete(item)
        for i, mod in enumerate(self.modules):
            tname = "With hole (Type 1)" if mod.module_type == 1 else "Solid (Type 2)"
            outer = np.asarray(mod.outer_contour, dtype=np.float32)
            area = abs(contour_sign(outer)) * 0.5
            parent_label = f"#{mod.parent_idx + 1}" if mod.parent_idx >= 0 else "root"
            self.tree_module.insert(
                "", "end",
                values=(i + 1, tname, len(mod.outer_contour), len(mod.inner_contours),
                        parent_label, mod.depth, f"{area:.2f}"),
            )

        # Module hierarchy tree
        for item in self.tree_hierarchy.get_children():
            self.tree_hierarchy.delete(item)

        def _insert_children(parent_idx, parent_tid):
            m = self.modules[parent_idx]
            for child_idx in m.children_indices:
                cm = self.modules[child_idx]
                tname_c = "With hole" if cm.module_type == 1 else "Solid"
                if cm.children_indices:
                    children_ids_str = ",".join(str(ci + 1) for ci in cm.children_indices)
                else:
                    children_ids_str = "-1"
                tid = self.tree_hierarchy.insert(
                    parent_tid, "end",
                    values=(child_idx + 1, tname_c, len(cm.children_indices),
                            children_ids_str, len(cm.inner_contours)),
                )
                _insert_children(child_idx, tid)

        root_indices = [i for i in range(len(self.modules)) if self.modules[i].parent_idx < 0]
        for ri in root_indices:
            rm = self.modules[ri]
            tname_r = "With hole" if rm.module_type == 1 else "Solid"
            if rm.children_indices:
                children_ids_str = ",".join(str(ci + 1) for ci in rm.children_indices)
            else:
                children_ids_str = "-1"
            tid = self.tree_hierarchy.insert(
                "", "end",
                values=(ri + 1, tname_r, len(rm.children_indices),
                        children_ids_str, len(rm.inner_contours)),
            )
            _insert_children(ri, tid)

        # Default details
        self.detail_text.delete("1.0", "end")
        lines = [f"[Character '{self.current_char}' analysis summary]",
                 f"- Outer contours: {n_outer} (clockwise, warm solid lines)",
                 f"- Inner contours: {n_inner} (counter-clockwise, cool dashed lines)",
                 f"- Modules: {len(self.modules)}",
                 f"- Triangles: {len(self.mesh['triangles'])}, Vertices: {len(self.mesh['vertices'])}",
                 "",
                 "Click an entry in the Contours or Modules table above to highlight in the left panel."]
        self.detail_text.insert("1.0", "\n".join(lines))

    def _update_detail_for_module(self, idx):
        if idx < 0 or idx >= len(self.modules):
            return
        mod = self.modules[idx]
        tname = "Polygon with hole (Type 1)" if mod.module_type == 1 else "Simple solid polygon (Type 2)"
        outer = np.asarray(mod.outer_contour, dtype=np.float32)
        lines = [f"[Module {idx + 1} details]", f"Type: {tname}"]
        if len(outer) >= 3:
            area = abs(contour_sign(outer)) * 0.5
            lines.append(f"Outer contour: {len(outer)} points, area ~ {area:.2f}")
            lines.append(f"Bounding box: x∈[{outer[:,0].min():.1f}, {outer[:,0].max():.1f}], "
                         f"y∈[{outer[:,1].min():.1f}, {outer[:,1].max():.1f}]")
        lines.append(f"Inner contours: {len(mod.inner_contours)}")
        for j, inner in enumerate(mod.inner_contours):
            arr = np.asarray(inner, dtype=np.float32)
            area = abs(contour_sign(arr)) * 0.5
            lines.append(f"  · Inner {j + 1}: {len(inner)} points, area ~ {area:.2f}")
        # Hierarchy info
        parent_label = f"Module #{mod.parent_idx + 1}" if mod.parent_idx >= 0 else "(root module)"
        lines.append(f"\n[Hierarchy]")
        lines.append(f"  Parent: {parent_label}")
        lines.append(f"  Depth: {mod.depth}")
        lines.append(f"  Children: {len(mod.children_indices)}")
        if mod.children_indices:
            lines.append(f"  Child IDs: " + ", ".join(f"#{ci + 1}" for ci in mod.children_indices))
        try:
            res = triangulate_module(mod)
            lines.append(f"\nThis module triangulation: {len(res['triangles'])} tri, {len(res['vertices'])} vertices")
        except Exception as e:
            lines.append(f"\nThis module triangulation failed: {e}")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", "\n".join(lines))

    def _update_detail_for_contour(self, idx):
        if idx < 0 or idx >= len(self.contour_info):
            return
        c = self.contour_info[idx]
        cnt = self.raw_contours_np[idx]
        mod_idx = self._contour_to_module.get(idx, -1)
        lines = [f"[Contour {idx} details]",
                 f"Type: {'outer' if c['type'] == 'outer' else 'inner'}",
                 f"Points: {c['points']}",
                 f"Area: {c['area']:.2f}",
                 f"Direction: {'CW' if c['type'] == 'outer' else 'CCW'}",
                 f"Winding sign: {c['sign']:+.1f}",
                 f"Module: {'Module ' + str(mod_idx + 1) if mod_idx >= 0 else 'unassigned'}"]
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", "\n".join(lines))

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------
    def export_image(self):
        if self.current_char is None:
            messagebox.showinfo("Info", "Please run analysis first")
            return
        path = filedialog.asksaveasfilename(
            title="Save image",
            defaultextension=".png",
            initialfile=f"{self.current_char}_glyph_analysis.png",
            filetypes=[("PNG image", "*.png"), ("SVG vector", "*.svg")],
        )
        if not path:
            return
        try:
            self.contour_fig.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Success", f"Saved to: {path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


# ============================================================
# Entry point
# ============================================================
def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if sys.platform.startswith("win"):
            style.theme_use("vista")
    except Exception:
        pass
    GlyphAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
