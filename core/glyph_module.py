import numpy as np


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
                f"parent={self.parent_idx}, children={self.children_indices}, depth={self.depth})")
