import numpy as np
from .contour_geometry import contour_sign, fix_contour, point_in_poly
from .glyph_module import GlyphModule


def classify_contours(contours_list, outer_indices=None):
    """Classify outer/inner contours in batch, and match inner contours to their direct parent holes."""
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
        outer_info = [info for info in contour_info if info["sign"] < 0]
        inner_info = [info for info in contour_info if info["sign"] > 0]

    outer_info.sort(key=lambda x: x["area"], reverse=True)
    all_outer = [info["contour"] for info in outer_info]

    inner_dict = {}
    for o_cnt in all_outer:
        inner_dict[id(o_cnt)] = []

    for i_info in inner_info:
        i_cnt = i_info["contour"]
        i_area = i_info["area"]
        vertices = i_cnt[:-1] if len(i_cnt) > 1 else i_cnt
        n_verts = len(vertices)

        containing = []
        for o_info in outer_info:
            o_cnt = o_info["contour"]
            o_area = o_info["area"]

            if i_area >= o_area * 0.95:
                continue

            n_inside = 0
            for v in vertices:
                if point_in_poly(v, o_cnt):
                    n_inside += 1
            ratio = n_inside / n_verts if n_verts > 0 else 0

            if ratio >= 0.9:
                containing.append((o_cnt, o_area))

        if containing:
            containing.sort(key=lambda x: x[1])
            best_outer = containing[0][0]
            inner_dict[id(best_outer)].append(i_cnt)
        else:
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


def build_modules(outer_list, inner_dict):
    """Build standardized two-level modules from outer contours and their inner contours."""
    modules = []
    for o_cnt in outer_list:
        mod = GlyphModule()
        mod.outer_contour = o_cnt
        mod.inner_contours = inner_dict.get(id(o_cnt), [])
        mod.module_type = 1 if len(mod.inner_contours) >= 1 else 2
        modules.append(mod)
    return modules


def _module_contains(parent_mod, child_mod, tol=1e-3):
    """Check whether the outer contour of parent contains the outer contour of child."""
    if parent_mod is None or child_mod is None:
        return False
    p_cnt = np.asarray(parent_mod.outer_contour, dtype=np.float64)
    c_cnt = np.asarray(child_mod.outer_contour, dtype=np.float64)
    if len(p_cnt) < 3 or len(c_cnt) < 3:
        return False
    
    vertices = c_cnt
    n_verts = len(vertices)
    if n_verts == 0:
        return False
    
    n_inside = sum(1 for v in vertices if point_in_poly(v, p_cnt))
    return n_inside / n_verts > 0.9


def build_module_hierarchy(modules):
    """Build a tree of module membership based on containment relationships."""
    n = len(modules)
    if n == 0:
        return modules

    areas = []
    for m in modules:
        arr = np.asarray(m.outer_contour, dtype=np.float64)
        areas.append(abs(contour_sign(arr)) * 0.5)

    for i in range(n):
        best_parent = -1
        best_area = float('inf')
        for j in range(n):
            if i == j:
                continue
            if areas[j] <= areas[i] * 1.1:
                continue
            if _module_contains(modules[j], modules[i]):
                if areas[j] < best_area:
                    best_area = areas[j]
                    best_parent = j
        modules[i].parent_idx = best_parent
        if best_parent >= 0:
            modules[best_parent].children_indices.append(i)

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
