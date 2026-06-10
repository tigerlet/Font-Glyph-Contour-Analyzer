import os
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from core import (
    char_to_contours_ft,
    classify_contours,
    build_modules,
    build_module_hierarchy,
    triangulate_module,
    merge_modules_results,
    contour_sign
)


class GlyphAnalyzerApp:
    DEFAULT_FONT_CANDIDATES = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simkai.ttf",
    ]

    def __init__(self, root):
        self.root = root

        self.font_path = self._auto_font_path()
        try:
            self.font_prop = FontProperties(fname=self.font_path)
        except Exception:
            self.font_prop = FontProperties(family="SimHei")

        try:
            import matplotlib
            matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        self._build_ui()

        self.char_var.set("国")
        self.root.after(300, self.analyze)

    def _auto_font_path(self):
        local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simhei.ttf")
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

        main_wrap = ttk.Frame(self.root)
        main_wrap.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 8))
        main_wrap.columnconfigure(0, weight=2)
        main_wrap.columnconfigure(1, weight=1)
        main_wrap.rowconfigure(0, weight=1)

        left_frame = tk.LabelFrame(main_wrap, text="  Glyph Contours  ", padx=6, pady=6,
                                   font=("Arial", 10))
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)

        self.contour_fig = plt.Figure(figsize=(8, 7), dpi=100, facecolor="white")
        self.contour_canvas = FigureCanvasTkAgg(self.contour_fig, master=left_frame)
        canvas_widget = self.contour_canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")

        tb_frame = ttk.Frame(left_frame)
        tb_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        toolbar1 = NavigationToolbar2Tk(self.contour_canvas, tb_frame)
        toolbar1.update()

        self.contour_ax = self.contour_fig.add_subplot(111)
        self.contour_ax.set_aspect("equal")
        self.contour_ax.text(0.5, 0.5, "Enter a character, then click -> Analyze",
                             ha="center", va="center", fontsize=14, color="gray",
                             transform=self.contour_ax.transAxes)
        self.contour_ax.set_xlim(0, 1)
        self.contour_ax.set_ylim(0, 1)
        self.contour_ax.axis("off")
        self.contour_fig.tight_layout()
        self.contour_canvas.draw()

        right_frame = tk.LabelFrame(main_wrap, text="  Analysis Results  ", padx=6, pady=6,
                                    font=("Arial", 10))
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_frame.rowconfigure(2, weight=1)
        right_frame.rowconfigure(4, weight=1)
        right_frame.columnconfigure(0, weight=1)

        self.summary_label = tk.Label(right_frame, text="(No analysis yet)",
                                      justify="left", anchor="w", font=("Arial", 9),
                                      relief="solid", bd=1, padx=8, pady=6)
        self.summary_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))

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

        tab_t.columnconfigure(0, weight=1)
        tab_t.rowconfigure(0, weight=1)
        self.result_fig = plt.Figure(figsize=(4, 3), dpi=100, facecolor="white")
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, master=tab_t)
        self.result_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.result_ax = self.result_fig.add_subplot(111)
        self.result_ax.set_aspect("equal")
        self.result_ax.axis("off")
        self.result_ax.text(0.5, 0.5, "(No triangulation yet)", ha="center", va="center",
                            fontsize=10, color="gray", transform=self.result_ax.transAxes)
        self.result_fig.tight_layout()
        self.result_canvas.draw()

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

        self.status_var = tk.StringVar(value="Ready. Enter a character and click -> Analyze.")
        status = tk.Label(self.root, textvariable=self.status_var, anchor="w",
                          relief="sunken", bd=1, padx=8, pady=2, font=("Arial", 9))
        status.pack(side="bottom", fill="x")

        self.current_char = None
        self.raw_contours_np = []
        self.contour_info = []
        self.modules = []
        self.mesh = {"vertices": np.array([]), "triangles": np.array([])}
        self.selected_module = -1
        self.selected_contour = -1
        self._contour_to_module = {}

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

        cinfo = []
        normalized = []
        outer_palette = ["#E53935", "#FB8C00", "#F4511E", "#D81B60", "#8E24AA"]
        inner_palette = ["#1E88E5", "#00ACC1", "#43A047", "#3949AB", "#00897B"]
        oi, ii = 0, 0
        for i, cnt in enumerate(contours):
            arr = np.array(cnt, dtype=np.float32)
            s = contour_sign(arr)
            ctype = "outer" if s < 0 else "inner"
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

    def _on_contour_select(self, _event):
        sel = self.tree_contour.selection()
        if not sel:
            return
        idx = int(self.tree_contour.item(sel[0], "values")[0])
        self.selected_contour = idx
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
        values = self.tree_hierarchy.item(sel[0], "values")
        if not values:
            return
        idx = int(values[0]) - 1
        self.selected_module = idx
        self._redraw_contour()
        self._update_detail_for_module(idx)

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

        legend_lines = []
        legend_labels = []
        x_min, y_min = np.min(all_pts, axis=0)
        x_max, y_max = np.max(all_pts, axis=0)
        x_range = (x_max - x_min) if x_max > x_min else 1.0
        y_range = (y_max - y_min) if y_max > y_min else 1.0
        char_size = max(x_range, y_range)

        for i, info in enumerate(self.contour_info):
            if info["type"] != "inner":
                continue
            cnt = self.raw_contours_np[i]
            line, = ax.plot(cnt[:, 0], cnt[:, 1], color=info["color"], lw=2.2, ls="--")
            legend_lines.append(line)
            legend_labels.append(f"Contour {i} (inner)")
            self._draw_arrow_and_label(ax, cnt, info, i, char_size, is_inner=True)

        for i, info in enumerate(self.contour_info):
            if info["type"] != "outer":
                continue
            cnt = self.raw_contours_np[i]
            line, = ax.plot(cnt[:, 0], cnt[:, 1], color=info["color"], lw=2.8, ls="-")
            legend_lines.append(line)
            legend_labels.append(f"Contour {i} (outer)")
            self._draw_arrow_and_label(ax, cnt, info, i, char_size, is_inner=False)

        if hl_outer is not None:
            ax.plot(hl_outer[:, 0], hl_outer[:, 1], color="#FFEB3B", lw=5, alpha=0.9, zorder=10)
        for ic in hl_inners:
            ax.plot(ic[:, 0], ic[:, 1], color="#FFEB3B", lw=4, alpha=0.9, zorder=10)

        margin = max((x_max - x_min), (y_max - y_min)) * 0.10
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

        title = f"Glyph: '{self.current_char}' (TrueType)" if self.current_char else "Glyph Contours"
        ax.set_title(title, fontsize=13, pad=8)
        if legend_lines:
            ax.legend(legend_lines, legend_labels, loc="upper left",
                      bbox_to_anchor=(1.0, 1.0),
                      fontsize=8, framealpha=0.9, ncol=1)

        self.contour_fig.tight_layout()
        self.contour_canvas.draw()

    def _draw_arrow_and_label(self, ax, cnt, info, idx, char_size, is_inner=False):
        n = len(cnt)
        if n < 2:
            return

        seg_lens = []
        for i in range(n - 1):
            dx = cnt[i + 1, 0] - cnt[i, 0]
            dy = cnt[i + 1, 1] - cnt[i, 1]
            seg_lens.append((dx * dx + dy * dy) ** 0.5)
        if not seg_lens:
            return

        max_len = max(seg_lens)
        best_idx = seg_lens.index(max_len)
        min_required = char_size * 0.04

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

        scale = 0.015 if is_inner else 0.025
        hw = char_size * scale
        hl = char_size * scale * 1.6

        arrow_start_x = sp[0] + dx * 0.25
        arrow_start_y = sp[1] + dy * 0.25
        arrow_dx = dx * 0.5
        arrow_dy = dy * 0.5

        arrow_len = (arrow_dx * arrow_dx + arrow_dy * arrow_dy) ** 0.5
        if arrow_len < hl:
            arrow_dx = dx * 0.3
            arrow_dy = dy * 0.3

        ax.arrow(arrow_start_x, arrow_start_y, arrow_dx, arrow_dy,
                 head_width=hw, head_length=min(hl, arrow_len * 0.6),
                 fc=info["color"], ec=info["color"],
                 length_includes_head=True, alpha=0.95, zorder=8)

        if getattr(self, "show_label_var", None) and self.show_label_var.get():
            label_text = f"#{idx}"
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

        for item in self.tree_contour.get_children():
            self.tree_contour.delete(item)
        for c in self.contour_info:
            cdir = "CW" if c["type"] == "outer" else "CCW"
            self.tree_contour.insert(
                "", "end",
                values=(c["idx"], "outer" if c["type"] == "outer" else "inner",
                        c["points"], f"{c['area']:.2f}", cdir),
            )

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
