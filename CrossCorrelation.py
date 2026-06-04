"""
Cross-Correlation Analysis – ROI Selection + Manual Alignment
=============================================================
Stage 1 – Select ROI on LIF and RAY images
Stage 2 – Manually position RAY crop over LIF crop using arrow keys
Stage 3 – Apply fixed offset to all frames and compute corr2

Controls (Stage 2):
  Arrow keys         – move RAY by 1 pixel
  Shift + Arrow key  – move RAY by 10 pixels
  +/-                – adjust RAY transparency
  1/2                – decrease/increase contrast clipping
  Enter              – confirm offset and run correlation
  Q                  – cancel

Requires:  numpy scikit-image matplotlib tqdm
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from skimage.transform import resize
from tqdm import tqdm

# =============================================================================
# USER SETTINGS
# =============================================================================

LIF_FILE      = r""   # selected via startup window
RAY_FILE      = r""   # selected via startup window

RAY_SCALE     = 0.92
PREVIEW_FRAME = 2

MIE_MEAN_THRESHOLD = 0.85
MIE_STD_THRESHOLD  = 0.1

# =============================================================================
# SPE READER
# =============================================================================

class SPEReader:
    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            self.header = f.read(4100)
        self.xdim     = struct.unpack_from("<H", self.header,   42)[0]
        self.ydim     = struct.unpack_from("<H", self.header,  656)[0]
        dtype_id      = struct.unpack_from("<H", self.header,  108)[0]
        self.n_frames = struct.unpack_from("<i", self.header, 1446)[0]
        dtype_map     = {0: np.float32, 1: np.int32,
                         2: np.int16,   3: np.uint16}
        self.np_dtype    = dtype_map[dtype_id]
        self.itemsize    = np.dtype(self.np_dtype).itemsize
        self.frame_bytes = self.ydim * self.xdim * self.itemsize
        self.data_start  = 4100

    def read_frame(self, idx):
        offset = self.data_start + idx * self.frame_bytes
        with open(self.path, "rb") as f:
            f.seek(offset)
            buf = f.read(self.frame_bytes)
        return np.frombuffer(buf, dtype=self.np_dtype)\
                 .reshape(self.ydim, self.xdim).astype(np.float32)

    def __len__(self):
        return self.n_frames

    def read_all(self):
        return np.stack([self.read_frame(i) for i in range(self.n_frames)], axis=0)

# =============================================================================
# HELPERS
# =============================================================================

def normalize(image, clip_pct=0):
    img = image.astype(np.float64)
    if clip_pct > 0:
        lo = np.percentile(img, clip_pct)
        hi = np.percentile(img, 100 - clip_pct)
    else:
        lo, hi = img.min(), img.max()
    if hi == lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def corr2(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return 0.0 if denom == 0 else (a * b).sum() / denom


def prepare_lif(frame, lif_crop):
    r0, r1, c0, c1 = lif_crop
    return normalize(frame[r0:r1, c0:c1].astype(np.float64))


def prepare_ray(frame, ray_crop, scale=None):
    r0, r1, c0, c1 = ray_crop
    s = scale if scale is not None else RAY_SCALE
    new_shape = (int(frame.shape[0] * s),
                 int(frame.shape[1] * s))
    scaled   = resize(frame.astype(np.float64), new_shape, anti_aliasing=True)
    cropped  = scaled[r0:r1, c0:c1]
    inverted = cropped.max() - cropped
    return normalize(inverted)

# =============================================================================
# STAGE 1 — ROI SELECTOR
# =============================================================================

class ROISelector:
    def __init__(self, image, title, cmap="gray"):
        self.coords = None
        self.image  = image
        self.cmap   = cmap

        self.fig = plt.figure(figsize=(10, 8))
        self.ax  = self.fig.add_axes([0.05, 0.12, 0.90, 0.82])
        self.fig.canvas.manager.set_window_title(title)

        self.vmin = float(np.percentile(image, 2))
        self.vmax = float(np.percentile(image, 98))
        self.im = self.ax.imshow(image, cmap=cmap, origin="upper",
                                  vmin=self.vmin, vmax=self.vmax)
        self.ax.set_title(
            f"{title}\n"
            "► Draw a rectangle over the region of interest\n"
            "► Press ENTER to confirm  |  R to reset  |  Q to skip",
            fontsize=10
        )

        # Brightness sliders
        from matplotlib.widgets import Slider as _Slider
        ax_lo = self.fig.add_axes([0.10, 0.04, 0.33, 0.02], facecolor='#333333')
        ax_hi = self.fig.add_axes([0.55, 0.04, 0.33, 0.02], facecolor='#333333')
        self.sl_lo = _Slider(ax_lo, 'B low',  0, 100, valinit=2,  color='#555555')
        self.sl_hi = _Slider(ax_hi, 'B high', 0, 100, valinit=98, color='#555555')
        for sl in [self.sl_lo, self.sl_hi]:
            sl.label.set_fontsize(8)
        self.sl_lo.on_changed(self._on_brightness)
        self.sl_hi.on_changed(self._on_brightness)

        self.selector = RectangleSelector(
            self.ax, self._on_select,
            useblit=True, button=[1],
            minspanx=5, minspany=5,
            spancoords="pixels", interactive=True,
        )

        self.coord_text = self.fig.text(
            0.5, 0.01, "No region selected yet",
            ha="center", fontsize=9, color="gray"
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.show()

    def _on_brightness(self, val):
        self.vmin = float(np.percentile(self.image, self.sl_lo.val))
        self.vmax = float(np.percentile(self.image, self.sl_hi.val))
        self.im.set_clim(self.vmin, self.vmax)
        self.fig.canvas.draw_idle()

    def _on_select(self, eclick, erelease):
        x1 = int(min(eclick.xdata, erelease.xdata))
        x2 = int(max(eclick.xdata, erelease.xdata))
        y1 = int(min(eclick.ydata, erelease.ydata))
        y2 = int(max(eclick.ydata, erelease.ydata))
        self.coords = (y1, y2, x1, x2)
        self.coord_text.set_text(
            f"Selected: rows {y1}:{y2}, cols {x1}:{x2}  →  "
            f"{y2-y1} x {x2-x1} px  |  Press ENTER to confirm"
        )
        self.coord_text.set_color("steelblue")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "enter":
            if self.coords:
                plt.close(self.fig)
            else:
                print("  Please select a region first!")
        elif event.key == "r":
            self.coords = None
            self.coord_text.set_text("Reset – please select again")
            self.coord_text.set_color("orange")
            self.fig.canvas.draw_idle()
        elif event.key == "q":
            self.coords = None
            plt.close(self.fig)


def confirm_selection(image, coords, title, cmap="gray", vmin=None, vmax=None):
    r0, r1, c0, c1 = coords
    cropped = image[r0:r1, c0:c1]

    # Use provided vmin/vmax for left plot, auto for right (cropped)
    if vmin is None: vmin = float(np.percentile(image,   2))
    if vmax is None: vmax = float(np.percentile(image,  98))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title(f"Preview: {title}")

    axes[0].imshow(image, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    rect = patches.Rectangle((c0, r0), c1-c0, r1-r0,
                               linewidth=2, edgecolor="red", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].set_title("Full image (red = selection)")
    axes[0].axis("off")

    axes[1].imshow(normalize(cropped), cmap=cmap, origin="upper")
    axes[1].set_title(f"Cropped region\n{r1-r0} x {c1-c0} px")
    axes[1].axis("off")

    fig.suptitle(f"{title}  –  Close window to continue", fontsize=11)
    plt.tight_layout()
    plt.show()

# =============================================================================
# STAGE 2 — MANUAL ALIGNER
# =============================================================================

class ManualAligner:
    def __init__(self, lif, ray):
        self.lif       = lif
        self.ray       = ray
        self.lif_h, self.lif_w = lif.shape
        self.ray_h, self.ray_w = ray.shape
        self.alpha     = 0.5
        self.clip_pct  = 0
        self.confirmed = False

        self.dr = (self.lif_h - self.ray_h) // 2
        self.dc = (self.lif_w - self.ray_w) // 2
        self._clamp()

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Manual Alignment")
        self._draw()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.tight_layout()
        plt.show()

    def _clamp(self):
        self.dr = int(np.clip(self.dr, 0, self.lif_h - self.ray_h))
        self.dc = int(np.clip(self.dc, 0, self.lif_w - self.ray_w))

    def _draw(self):
        self.ax.clear()
        lif_disp = normalize(self.lif, self.clip_pct)
        ray_disp = normalize(self.ray, self.clip_pct)

        self.ax.imshow(lif_disp, cmap="inferno", origin="upper",
                       extent=[0, self.lif_w, self.lif_h, 0])
        self.ax.imshow(ray_disp, cmap="gray", alpha=self.alpha, origin="upper",
                       extent=[self.dc, self.dc + self.ray_w,
                                self.dr + self.ray_h, self.dr])

        rect = patches.Rectangle(
            (self.dc, self.dr), self.ray_w, self.ray_h,
            linewidth=1.5, edgecolor="cyan", facecolor="none"
        )
        self.ax.add_patch(rect)

        lif_patch = self.lif[self.dr:self.dr + self.ray_h,
                              self.dc:self.dc + self.ray_w]
        c = corr2(lif_patch, self.ray)

        self.ax.set_title(
            f"Δrow={self.dr}, Δcol={self.dc}  |  corr2={c:.3f}  |  "
            f"α={self.alpha:.1f}  |  contrast clip={self.clip_pct}%\n"
            "Arrows: 1px  |  Shift+Arrows: 10px  |  +/-: transparency  |  "
            "1/2: contrast  |  Enter: confirm  |  Q: cancel",
            fontsize=8
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        step = 10 if event.key in ("shift+up", "shift+down",
                                    "shift+left", "shift+right") else 1
        if event.key in ("up", "shift+up"):         self.dr -= step
        elif event.key in ("down", "shift+down"):   self.dr += step
        elif event.key in ("left", "shift+left"):   self.dc -= step
        elif event.key in ("right", "shift+right"): self.dc += step
        elif event.key in ("+", "="):
            self.alpha = min(1.0, round(self.alpha + 0.1, 1))
        elif event.key == "-":
            self.alpha = max(0.0, round(self.alpha - 0.1, 1))
        elif event.key == "1":
            self.clip_pct = max(0, self.clip_pct - 1)
        elif event.key == "2":
            self.clip_pct = min(49, self.clip_pct + 1)
        elif event.key == "enter":
            self.confirmed = True
            plt.close(self.fig)
            return
        elif event.key == "q":
            plt.close(self.fig)
            return
        self._clamp()
        self._draw()

# =============================================================================
# MAIN
# =============================================================================

def main():
    import os
    from pathlib import Path
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import sys

    OUT_DIR     = Path(__file__).parent
    CONFIG_FILE = OUT_DIR / "last_used.json"
    print(f"  Output folder: {OUT_DIR}")

    # Load last used paths
    last_lif, last_ray = "", ""
    if CONFIG_FILE.exists():
        import json
        try:
            cfg      = json.loads(CONFIG_FILE.read_text())
            last_lif = cfg.get("lif_file", "")
            last_ray = cfg.get("ray_file", "")
        except Exception:
            pass

    # ── Startup window ────────────────────────────────────
    root = tk.Tk()
    root.title("Cross-Correlation Analysis")
    root.resizable(False, False)
    root.attributes('-topmost', True)

    tk.Label(root, text="Cross-Correlation Analysis",
             font=("Arial", 13, "bold"), pady=8).grid(
        row=0, column=0, columnspan=3, padx=20)

    # File fields
    lif_var = tk.StringVar(value=last_lif)
    ray_var = tk.StringVar(value=last_ray)

    def browse_lif():
        p = filedialog.askopenfilename(title="Select LIF (OH) file",
            filetypes=[("SPE files", "*.spe"), ("All files", "*.*")])
        if p: lif_var.set(p)

    def browse_ray():
        p = filedialog.askopenfilename(title="Select Rayleigh file",
            filetypes=[("SPE files", "*.spe"), ("All files", "*.*")])
        if p: ray_var.set(p)

    for row, (label, var, cmd) in enumerate([
        ("LIF (OH) file:", lif_var, browse_lif),
        ("Rayleigh file:", ray_var, browse_ray),
    ], start=1):
        tk.Label(root, text=label, anchor="w", width=18).grid(
            row=row, column=0, padx=(20,5), pady=4, sticky="w")
        tk.Entry(root, textvariable=var, width=50).grid(
            row=row, column=1, padx=5, pady=4)
        tk.Button(root, text="Browse", command=cmd).grid(
            row=row, column=2, padx=(5,20), pady=4)

    # Separator
    tk.Frame(root, height=1, bg="#cccccc").grid(
        row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=6)

    # Settings
    tk.Label(root, text="Settings",
             font=("Arial", 10, "bold")).grid(
        row=4, column=0, columnspan=3, padx=20, pady=(4,2), sticky="w")

    preview_var  = tk.StringVar(value=str(PREVIEW_FRAME))
    rayscale_var = tk.StringVar(value=str(RAY_SCALE))
    mie_mean_var = tk.StringVar(value=str(MIE_MEAN_THRESHOLD))
    mie_std_var  = tk.StringVar(value=str(MIE_STD_THRESHOLD))

    settings = [
        ("Preview frame index:", preview_var),
        ("RAY scale factor:",    rayscale_var),
        ("Mie mean threshold:",  mie_mean_var),
        ("Mie std threshold:",   mie_std_var),
    ]
    for i, (label, var) in enumerate(settings):
        tk.Label(root, text=label, anchor="w", width=22).grid(
            row=5+i, column=0, padx=(20,5), pady=3, sticky="w")
        tk.Entry(root, textvariable=var, width=12).grid(
            row=5+i, column=1, padx=5, pady=3, sticky="w")

    # Confirm / Cancel
    confirmed = [False]
    def on_confirm():
        if not lif_var.get().strip() or not ray_var.get().strip():
            messagebox.showerror("Error", "Please select both LIF and RAY files.")
            return
        confirmed[0] = True
        root.quit()
    def on_cancel():
        root.quit()

    btn_frame = tk.Frame(root)
    btn_frame.grid(row=10, column=0, columnspan=3, pady=15)
    tk.Button(btn_frame, text="Confirm", width=12,
              bg="#336633", fg="white", command=on_confirm).pack(
        side="left", padx=10)
    tk.Button(btn_frame, text="Cancel", width=12,
              command=on_cancel).pack(side="left", padx=10)

    root.mainloop()

    # Read all values BEFORE destroying
    _confirmed   = confirmed[0]
    _lif         = lif_var.get().strip()
    _ray         = ray_var.get().strip()
    _preview     = preview_var.get()
    _rayscale    = rayscale_var.get()
    _mie_mean    = mie_mean_var.get()
    _mie_std     = mie_std_var.get()
    try:
        root.destroy()
    except Exception:
        pass

    if not _confirmed:
        sys.exit(0)

    # Save last used paths
    import json
    CONFIG_FILE.write_text(json.dumps({
        "lif_file": _lif,
        "ray_file": _ray
    }, indent=2))

    # Read settings from dialog
    LIF_FILE_RUN  = _lif
    RAY_FILE_RUN  = _ray
    PREVIEW_RUN   = int(_preview)
    RAYSCALE_RUN  = float(_rayscale)
    MIE_MEAN_RUN  = float(_mie_mean)
    MIE_STD_RUN   = float(_mie_std)

    # ── Load preview frames ───────────────────────────────
    print(f"\n[1/3] Loading preview frames (frame {PREVIEW_RUN})...")
    lif_reader = SPEReader(LIF_FILE_RUN)
    ray_reader = SPEReader(RAY_FILE_RUN)
    lif_frame_raw = lif_reader.read_frame(PREVIEW_RUN).astype(np.float64)
    ray_frame_raw = ray_reader.read_frame(PREVIEW_RUN).astype(np.float64)
    print(f"  LIF: {lif_frame_raw.shape[0]} x {lif_frame_raw.shape[1]} px")
    print(f"  RAY: {ray_frame_raw.shape[0]} x {ray_frame_raw.shape[1]} px")



    # Scale RAY for display — show raw (no inversion) for ROI selection
    new_shape = (int(ray_frame_raw.shape[0] * RAYSCALE_RUN),
                 int(ray_frame_raw.shape[1] * RAYSCALE_RUN))
    ray_scaled  = resize(ray_frame_raw, new_shape, anti_aliasing=True)
    ray_display = ray_scaled   # raw — easier to see the laser sheet for cropping

    # ── Stage 1: ROI selection ────────────────────────────
    print("\n[1/3] Stage 1 — Select ROI on LIF image...")
    lif_selector = ROISelector(lif_frame_raw,
                                title="LIF – Select region of interest",
                                cmap="inferno")
    lif_crop = lif_selector.coords
    if lif_crop is None:
        print("  LIF: No region selected — exiting.")
        return
    confirm_selection(lif_frame_raw, lif_crop, "LIF", cmap="inferno")
    r0, r1, c0, c1 = lif_crop
    print(f"  LIF_CROP = ({r0}, {r1}, {c0}, {c1})  →  {r1-r0} x {c1-c0} px")

    print("\n       Stage 1 — Select ROI on RAY image (scaled & inverted)...")
    ray_selector = ROISelector(ray_display,
                                title="RAY (scaled) – Select region of interest",
                                cmap="gray")
    ray_crop = ray_selector.coords
    if ray_crop is None:
        print("  RAY: No region selected — exiting.")
        return
    confirm_selection(ray_display, ray_crop, "RAY (scaled)", cmap="gray", vmin=ray_selector.vmin, vmax=ray_selector.vmax)
    r0, r1, c0, c1 = ray_crop
    print(f"  RAY_CROP = ({r0}, {r1}, {c0}, {c1})  →  {r1-r0} x {c1-c0} px")

    # ── Stage 2: Manual alignment ─────────────────────────
    print(f"\n[2/3] Stage 2 — Manual alignment (frame {PREVIEW_RUN})...")
    lif_prev = prepare_lif(lif_frame_raw, lif_crop)
    ray_prev = prepare_ray(ray_frame_raw, ray_crop, scale=RAYSCALE_RUN)

    print(f"  LIF crop: {lif_prev.shape[0]} x {lif_prev.shape[1]} px")
    print(f"  RAY crop: {ray_prev.shape[0]} x {ray_prev.shape[1]} px")

    if ray_prev.shape[0] > lif_prev.shape[0] or ray_prev.shape[1] > lif_prev.shape[1]:
        print("  ERROR: RAY crop is larger than LIF crop — adjust coordinates.")
        return

    aligner = ManualAligner(lif_prev, ray_prev)
    if not aligner.confirmed:
        print("  Cancelled.")
        return

    dr_fix = aligner.dr
    dc_fix = aligner.dc
    lif_patch  = lif_prev[dr_fix:dr_fix + ray_prev.shape[0],
                           dc_fix:dc_fix + ray_prev.shape[1]]
    c_preview  = corr2(lif_patch, ray_prev)
    print(f"\n  Confirmed offset: Δrow={dr_fix}, Δcol={dc_fix} px")
    print(f"  corr2 on preview frame: {c_preview:.4f}")

    # ── Stage 3: corr2 for all frames ────────────────────
    print(f"\n[3/3] Stage 3 — Computing corr2 for all frames...")
    lif_frames = lif_reader.read_all()
    ray_frames = ray_reader.read_all()
    n_frames   = min(len(lif_frames), len(ray_frames))
    print(f"  {n_frames} frames")

    corr_values, ray_means, ray_stds = [], [], []
    for i in tqdm(range(n_frames), desc="Correlation"):
        lif  = prepare_lif(lif_frames[i], lif_crop)
        ray  = prepare_ray(ray_frames[i], ray_crop, scale=RAYSCALE_RUN)
        patch = lif[dr_fix:dr_fix + ray.shape[0],
                    dc_fix:dc_fix + ray.shape[1]]
        corr_values.append(corr2(patch, ray))
        ray_means.append(ray.mean())
        ray_stds.append(ray.std())

    corr_values = np.array(corr_values)
    ray_means   = np.array(ray_means)
    ray_stds    = np.array(ray_stds)

    mie_mask = (ray_means > MIE_MEAN_RUN) & (ray_stds < MIE_STD_RUN)
    n_mie    = mie_mask.sum()
    clean    = corr_values[~mie_mask]

    print(f"\n{'─'*45}")
    print(f"LIF_CROP             : {lif_crop}")
    print(f"RAY_CROP             : {ray_crop}")
    print(f"Fixed offset         : Δrow={dr_fix}, Δcol={dc_fix} px")
    print(f"Mie frames detected  : {n_mie} / {n_frames} ({100*n_mie/n_frames:.1f}%)")
    print(f"Mean correlation     : {corr_values.mean():.4f}  (all frames)")
    print(f"Mean correlation     : {clean.mean():.4f}  (excl. Mie)")
    print(f"{'─'*45}\n")

    np.save(str(OUT_DIR / "corr_values_manual.npy"), corr_values)
    np.save(str(OUT_DIR / "mie_mask.npy"), mie_mask)
    print("Saved: corr_values_manual.npy, mie_mask.npy")

    # ── GIF ──────────────────────────────────────────────
    print("\nGenerating overlay GIF...")
    import matplotlib.animation as animation
    import matplotlib.gridspec as gridspec

    lif_disp = normalize(lif_prev)
    ray_disp = normalize(ray_prev)
    alphas   = list(np.linspace(0, 1, 30)) + list(np.linspace(1, 0, 30))

    fig_gif, ax_gif = plt.subplots(figsize=(8, 6))
    ax_gif.axis("off")
    ax_gif.imshow(lif_disp, cmap="inferno", origin="upper",
                  extent=[0, lif_prev.shape[1], lif_prev.shape[0], 0])
    im_ray = ax_gif.imshow(ray_disp, cmap="gray", alpha=0.0, origin="upper",
                            extent=[dc_fix, dc_fix + ray_prev.shape[1],
                                    dr_fix + ray_prev.shape[0], dr_fix])
    ax_gif.add_patch(patches.Rectangle(
        (dc_fix, dr_fix), ray_prev.shape[1], ray_prev.shape[0],
        linewidth=1.5, edgecolor="cyan", facecolor="none"))
    title_gif = ax_gif.set_title("", fontsize=9)

    def update(alpha):
        im_ray.set_alpha(alpha)
        title_gif.set_text(
            f"LIF ↔ RAY  |  α={alpha:.2f}  |  "
            f"Δrow={dr_fix}, Δcol={dc_fix}  |  corr2={c_preview:.3f}"
        )
        return im_ray, title_gif

    ani = animation.FuncAnimation(fig_gif, update, frames=alphas,
                                   interval=50, blit=False)

    gif_path = str(OUT_DIR / "overlay_alpha_sweep.gif")
    ani.save(gif_path, writer="pillow", fps=20)
    plt.close(fig_gif)
    print(f"GIF saved: {gif_path}")

    # ── Plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(corr_values, color="steelblue", linewidth=0.8, label="corr2")
    if n_mie > 0:
        ax.scatter(np.where(mie_mask)[0], corr_values[mie_mask],
                   color="red", zorder=5, s=20,
                   label=f"Mie scattering ({n_mie} frames)")
    ax.axhline(corr_values.mean(), color="gray", linestyle="--", linewidth=0.8,
               label=f"Mean all = {corr_values.mean():.3f}")
    ax.axhline(clean.mean(), color="red", linestyle="--", linewidth=0.8,
               label=f"Mean excl. Mie = {clean.mean():.3f}")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Correlation coefficient (corr2)")
    ax.set_title(f"LIF ↔ RAY  |  Δrow={dr_fix}, Δcol={dc_fix} px")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "corr_plot_manual.png"), dpi=150)
    plt.show()
    print(f"Plot saved: {OUT_DIR / 'corr_plot_manual.png'}")


if __name__ == "__main__":
    main()
