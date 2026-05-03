"""
Практическая работа №7. Вариант 15.
RGB <-> HSL преобразование с интерактивным GUI и историей Undo/Redo.

Формулы перевода взяты из методички (слайды 23-24 PDF):
  RGB -> HSL: L = (MAX+MIN)/2,
              S = (MAX-MIN)/(MAX+MIN)        при L <= 0.5
                = (MAX-MIN)/(2-MAX-MIN)      при L >  0.5
              H по веткам в зависимости от того, какой канал MAX.
  HSL -> RGB: через Q, P и пороги 1/6, 1/2, 2/3 (см. слайд 24).

Готовые функции colorsys / cv2.cvtColor НЕ используются - всё вручную на NumPy.
"""

from __future__ import annotations

import os
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk


# ============================================================
# 1. RGB <-> HSL (ручная векторная реализация)
# ============================================================

def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """rgb: float[H,W,3] в [0,1]  ->  hsl: float[H,W,3], H в [0,360), S,L в [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    delta = mx - mn

    L = (mx + mn) / 2.0

    S = np.zeros_like(L)
    nz = delta > 1e-12
    # S по двум веткам (см. слайд 23)
    low = nz & (L <= 0.5)
    high = nz & (L > 0.5)
    S[low] = delta[low] / (mx[low] + mn[low])
    S[high] = delta[high] / (2.0 - mx[high] - mn[high])

    H = np.zeros_like(L)
    # MAX = R
    mask = nz & (mx == r)
    H[mask] = 60.0 * ((g[mask] - b[mask]) / delta[mask] % 6)
    # MAX = G
    mask = nz & (mx == g) & ~(mx == r)
    H[mask] = 60.0 * ((b[mask] - r[mask]) / delta[mask] + 2.0)
    # MAX = B
    mask = nz & (mx == b) & ~(mx == r) & ~(mx == g)
    H[mask] = 60.0 * ((r[mask] - g[mask]) / delta[mask] + 4.0)

    H = np.where(H < 0, H + 360.0, H)
    return np.stack([H, S, L], axis=-1)


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """hsl: H в [0,360), S,L в [0,1]  ->  rgb: float[H,W,3] в [0,1]."""
    H, S, L = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    Q = np.where(L < 0.5, L * (1.0 + S), L + S - L * S)
    P = 2.0 * L - Q
    Hk = (H / 360.0) % 1.0

    def channel(t: np.ndarray) -> np.ndarray:
        t = np.where(t < 0, t + 1.0, t)
        t = np.where(t > 1, t - 1.0, t)
        out = np.where(
            t < 1.0 / 6.0, P + (Q - P) * 6.0 * t,
            np.where(
                t < 1.0 / 2.0, Q,
                np.where(
                    t < 2.0 / 3.0, P + (Q - P) * (2.0 / 3.0 - t) * 6.0,
                    P,
                ),
            ),
        )
        return out

    R = channel(Hk + 1.0 / 3.0)
    G = channel(Hk)
    B = channel(Hk - 1.0 / 3.0)

    # Если S=0 — серый: R=G=B=L
    achrom = S < 1e-12
    R = np.where(achrom, L, R)
    G = np.where(achrom, L, G)
    B = np.where(achrom, L, B)

    return np.stack([R, G, B], axis=-1)


def apply_hsl_shift(hsl: np.ndarray, dh: float, ds: float, dl: float) -> np.ndarray:
    """Сдвигаем H по кругу, S и L клиппим в [0,1]."""
    H = (hsl[..., 0] + dh) % 360.0
    S = np.clip(hsl[..., 1] + ds, 0.0, 1.0)
    L = np.clip(hsl[..., 2] + dl, 0.0, 1.0)
    return np.stack([H, S, L], axis=-1)


# ============================================================
# 2. GUI (tkinter)
# ============================================================

PREVIEW_MAX = 480  # макс. сторона превью, чтобы не тормозило


class HSLEditor(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("HSL-редактор - Вариант 15 (Undo/Redo)")
        self.geometry("900x640")

        # Состояние изображения
        self.src_rgb: np.ndarray | None = None     # исходный (полный размер) [0,1]
        self.src_hsl: np.ndarray | None = None
        self.preview_rgb: np.ndarray | None = None # уменьшенный для UI
        self.preview_hsl: np.ndarray | None = None
        self.tk_img: ImageTk.PhotoImage | None = None

        # История Undo/Redo
        self.history: deque[tuple[float, float, float]] = deque(maxlen=200)
        self.future: deque[tuple[float, float, float]] = deque(maxlen=200)
        self._committing = False  # флаг, чтобы не плодить записи во время undo/redo
        self._after_id: str | None = None  # отложенная фиксация в историю

        self._build_ui()
        self._update_history_buttons()

        # Горячие клавиши
        self.bind("<Control-z>", lambda _e: self.undo())
        self.bind("<Control-y>", lambda _e: self.redo())
        self.bind("<Control-Shift-Z>", lambda _e: self.redo())

    # ---------- UI ----------
    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(top, text="Открыть...", command=self.open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Сохранить как...", command=self.save_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Сброс", command=self.reset_sliders).pack(side=tk.LEFT, padx=8)

        self.btn_undo = ttk.Button(top, text="← Undo (Ctrl+Z)", command=self.undo)
        self.btn_undo.pack(side=tk.LEFT, padx=2)
        self.btn_redo = ttk.Button(top, text="Redo (Ctrl+Y) →", command=self.redo)
        self.btn_redo.pack(side=tk.LEFT, padx=2)

        self.lbl_hist = ttk.Label(top, text="История: 0 / 0")
        self.lbl_hist.pack(side=tk.RIGHT, padx=6)

        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Панель ползунков
        ctrl = ttk.LabelFrame(body, text="Параметры HSL")
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        self.var_h = tk.DoubleVar(value=0.0)
        self.var_s = tk.DoubleVar(value=0.0)
        self.var_l = tk.DoubleVar(value=0.0)

        self._make_slider(ctrl, "ΔHue (град.)", self.var_h, -180, 180, 1)
        self._make_slider(ctrl, "ΔSaturation", self.var_s, -1.0, 1.0, 0.01)
        self._make_slider(ctrl, "ΔLightness", self.var_l, -1.0, 1.0, 0.01)

        ttk.Label(
            ctrl,
            text=(
                "Подсказки:\n"
                "  • H — поворот цветового круга\n"
                "  • S — насыщенность (0 = серый)\n"
                "  • L — светлота (0 = чёрный, 1 = белый)\n"
                "  • Ctrl+Z / Ctrl+Y — Undo / Redo"
            ),
            justify=tk.LEFT,
            foreground="#555",
        ).pack(side=tk.TOP, anchor="w", padx=6, pady=8)

        # Канвас под изображение
        self.canvas = tk.Canvas(body, bg="#222")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())

        self.status = ttk.Label(self, text="Откройте изображение, чтобы начать.")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _make_slider(self, parent, label: str, var: tk.DoubleVar,
                     lo: float, hi: float, resolution: float) -> None:
        frm = ttk.Frame(parent)
        frm.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Label(frm, text=label).pack(side=tk.TOP, anchor="w")

        val_lbl = ttk.Label(frm, text=f"{var.get():+.2f}")
        val_lbl.pack(side=tk.TOP, anchor="e")

        def on_change(_v: str) -> None:
            val_lbl.config(text=f"{var.get():+.2f}")
            self._on_slider_change()

        scale = tk.Scale(
            frm, from_=lo, to=hi, resolution=resolution,
            orient=tk.HORIZONTAL, variable=var,
            length=240, showvalue=False, command=on_change,
        )
        scale.pack(side=tk.TOP, fill=tk.X)

        # Фиксируем в историю по отпусканию мыши - так одно перетаскивание = одна запись
        scale.bind("<ButtonRelease-1>", lambda _e: self._commit_history())

    # ---------- Файлы ----------
    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Открыть изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{exc}")
            return

        self.src_rgb = np.asarray(img, dtype=np.float32) / 255.0
        self.src_hsl = rgb_to_hsl(self.src_rgb)

        # уменьшенная копия для отзывчивого превью
        prev = img.copy()
        prev.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
        self.preview_rgb = np.asarray(prev, dtype=np.float32) / 255.0
        self.preview_hsl = rgb_to_hsl(self.preview_rgb)

        self.history.clear()
        self.future.clear()
        self.history.append(self._slider_state())
        self.reset_sliders()
        self._update_history_buttons()
        self.status.config(text=f"Открыто: {os.path.basename(path)} ({img.width}x{img.height})")
        self._redraw()

    def save_image(self) -> None:
        if self.src_hsl is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте изображение.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить как...",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if not path:
            return
        dh, ds, dl = self._slider_state()
        shifted = apply_hsl_shift(self.src_hsl, dh, ds, dl)
        rgb = hsl_to_rgb(shifted)
        out = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(out, "RGB").save(path)
        self.status.config(text=f"Сохранено: {path}")

    # ---------- Ползунки ----------
    def _slider_state(self) -> tuple[float, float, float]:
        return (float(self.var_h.get()), float(self.var_s.get()), float(self.var_l.get()))

    def _set_sliders(self, state: tuple[float, float, float]) -> None:
        self._committing = True
        self.var_h.set(state[0])
        self.var_s.set(state[1])
        self.var_l.set(state[2])
        self._committing = False
        self._redraw()

    def reset_sliders(self) -> None:
        self._set_sliders((0.0, 0.0, 0.0))
        self._commit_history()

    def _on_slider_change(self) -> None:
        if self._committing:
            return
        # отрисовываем сразу (быстро благодаря preview)
        self._redraw()

    def _commit_history(self) -> None:
        if self._committing or self.src_hsl is None:
            return
        state = self._slider_state()
        if self.history and self.history[-1] == state:
            return
        self.history.append(state)
        self.future.clear()
        self._update_history_buttons()

    # ---------- Undo / Redo ----------
    def undo(self) -> None:
        if len(self.history) < 2:
            return
        current = self.history.pop()
        self.future.append(current)
        prev = self.history[-1]
        self._set_sliders(prev)
        self._update_history_buttons()
        self.status.config(text=f"Undo → ΔH={prev[0]:+.0f}°, ΔS={prev[1]:+.2f}, ΔL={prev[2]:+.2f}")

    def redo(self) -> None:
        if not self.future:
            return
        nxt = self.future.pop()
        self.history.append(nxt)
        self._set_sliders(nxt)
        self._update_history_buttons()
        self.status.config(text=f"Redo → ΔH={nxt[0]:+.0f}°, ΔS={nxt[1]:+.2f}, ΔL={nxt[2]:+.2f}")

    def _update_history_buttons(self) -> None:
        self.btn_undo.state(["!disabled"] if len(self.history) > 1 else ["disabled"])
        self.btn_redo.state(["!disabled"] if self.future else ["disabled"])
        # позиция в истории (1-based)
        depth = len(self.history)
        total = depth + len(self.future)
        self.lbl_hist.config(text=f"История: {depth} / {total}")

    # ---------- Рисование ----------
    def _redraw(self) -> None:
        if self.preview_hsl is None:
            return
        dh, ds, dl = self._slider_state()
        shifted = apply_hsl_shift(self.preview_hsl, dh, ds, dl)
        rgb = hsl_to_rgb(shifted)
        out = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(out, "RGB")

        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        img.thumbnail((cw - 10, ch - 10))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img)


def main() -> None:
    app = HSLEditor()
    app.mainloop()


if __name__ == "__main__":
    main()
