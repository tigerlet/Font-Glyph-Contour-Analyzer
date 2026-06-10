import sys
import tkinter as tk
from tkinter import ttk
from ui.main_window import GlyphAnalyzerApp


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
