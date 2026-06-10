from .glyph_module import GlyphModule
from .contour_geometry import contour_sign, is_outer_contour, fix_contour, point_in_poly
from .contour_classifier import classify_contours, build_modules, build_module_hierarchy, module_hierarchy_tree
from .contour_subdivision import subdivide_contour
from .triangulation import triangulate_module, merge_modules_results
from .freetype_parser import char_to_contours_ft


def glyph_triangulate_all(glyph_contours, outer_indices=None, max_seg_length=None, 
                          bisect_level=None, n_subdiv=None):
    """Unified public entry point for glyph triangulation."""
    outer_list, inner_dict = classify_contours(glyph_contours, outer_indices)
    modules = build_modules(outer_list, inner_dict)
    tri_res_list = [triangulate_module(m, max_seg_length=max_seg_length, 
                                        bisect_level=bisect_level, 
                                        n_subdiv=n_subdiv) for m in modules]
    final_mesh = merge_modules_results(tri_res_list)
    return final_mesh, modules


__all__ = [
    'GlyphModule',
    'contour_sign',
    'is_outer_contour',
    'fix_contour',
    'point_in_poly',
    'classify_contours',
    'build_modules',
    'build_module_hierarchy',
    'module_hierarchy_tree',
    'subdivide_contour',
    'triangulate_module',
    'merge_modules_results',
    'char_to_contours_ft',
    'glyph_triangulate_all'
]
