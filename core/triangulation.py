import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate
from .contour_subdivision import subdivide_contour


def triangulate_module(mod, max_seg_length=None, bisect_level=None, n_subdiv=None):
    """Triangulate a single glyph module using Shapely with contour subdivision."""
    outer = mod.outer_contour
    inners = mod.inner_contours

    try:
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

        outer_poly = Polygon(outer.tolist())
        inner_polys = []
        if len(inners) > 0:
            for ic in inners:
                if len(ic) >= 4:
                    inner_polys.append(Polygon(ic.tolist()))

        if inner_polys:
            poly = Polygon(outer.tolist(),
                           holes=[list(hole.exterior.coords) for hole in inner_polys])
        else:
            poly = Polygon(outer.tolist())

        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return {"vertices": np.array([]), "triangles": np.array([])}

        vert_map = {}
        vertices = []

        def _add_vertex(x, y):
            key = (round(float(x), 4), round(float(y), 4))
            if key not in vert_map:
                vert_map[key] = len(vertices)
                vertices.append([x, y])
            return vert_map[key]

        def _collect_triangles(geom):
            results = []
            for sub_t in triangulate(geom):
                sub_pts = np.array(list(sub_t.exterior.coords)[:-1])
                if len(sub_pts) == 3:
                    idx = [_add_vertex(pt[0], pt[1]) for pt in sub_pts]
                    results.append(idx)
            return results

        all_valid_tris = []

        tri_candidates = triangulate(poly)

        for tri in tri_candidates:
            tri_pts = np.array(list(tri.exterior.coords)[:-1])
            if len(tri_pts) != 3:
                continue

            tri_poly = Polygon(tri_pts)

            clipped = tri_poly.intersection(outer_poly)

            for inner_p in inner_polys:
                try:
                    clipped = clipped.difference(inner_p)
                except Exception:
                    pass

            if clipped.is_empty:
                continue

            if hasattr(clipped, 'exterior'):
                coords = np.array(list(clipped.exterior.coords)[:-1])
                if len(coords) >= 3:
                    all_valid_tris.extend(_collect_triangles(Polygon(coords)))
            elif hasattr(clipped, 'geoms'):
                for geom in clipped.geoms:
                    if hasattr(geom, 'exterior'):
                        coords = np.array(list(geom.exterior.coords)[:-1])
                        if len(coords) >= 3:
                            all_valid_tris.extend(_collect_triangles(Polygon(coords)))

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


def merge_modules_results(module_results):
    """Merge multiple module triangulation results into a single unified mesh."""
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
