import freetype
import numpy as np


def _normalize_ttf_points(points, tags):
    """Normalize a contour point sequence according to TrueType spec."""
    if len(points) == 0:
        return [], []

    pts = [(float(x), float(y)) for (x, y) in points]
    tags = list(tags)
    n = len(pts)

    def on_curve(t):
        return bool(t & 1)

    first_on = 0
    for i in range(n):
        if on_curve(tags[i]):
            first_on = i
            break
    
    pts = pts[first_on:] + pts[:first_on]
    tags = tags[first_on:] + tags[:first_on]
    pts.append(pts[0])
    tags.append(tags[0])

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
    """Sample quadratic Bezier curve."""
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
    
    Correctly parses quadratic Bezier curves (TrueType spec), 
    supporting multiple contours and holes.
    """
    face = freetype.Face(font_path)
    face.set_char_size(resolution * 64)

    char_code = ord(char)
    face.load_char(char_code, freetype.FT_LOAD_DEFAULT | 
                   freetype.FT_LOAD_NO_BITMAP | 
                   freetype.FT_LOAD_NO_HINTING)
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

        pts_norm, tags_norm = _normalize_ttf_points(raw_pts, raw_tags)
        if len(pts_norm) < 3:
            start = end + 1
            continue

        out_pts = []
        i = 0
        n_norm = len(pts_norm)
        while i < n_norm - 1:
            cur = pts_norm[i]
            nxt = pts_norm[i + 1]
            if tags_norm[i + 1] & 1:
                out_pts.append([cur[0] * scale, cur[1] * scale])
                i += 1
            else:
                if i + 2 < n_norm:
                    next_next = pts_norm[i + 2]
                else:
                    next_next = pts_norm[0]
                out_pts.append([cur[0] * scale, cur[1] * scale])
                samples = _sample_quadratic(cur, nxt, next_next, n_samples=bezier_samples)
                for (sx, sy) in samples:
                    out_pts.append([sx * scale, sy * scale])
                i += 2

        if out_pts and (out_pts[0][0] != out_pts[-1][0] or out_pts[0][1] != out_pts[-1][1]):
            out_pts.append(out_pts[0])
        if len(out_pts) >= 3:
            contours.append(out_pts)

        start = end + 1

    return contours
