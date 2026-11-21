"""The README figures, each in a light and a dark version, and the numbers the README quotes.

  decomposition-{light,dark}.png  the artificial signal of the ICEEMDAN paper (a sinusoid plus a
                                  short burst of a higher frequency) decomposed by this library,
                                  beside the authors' reference code when its output is there;
                                  the amplitude of the true components lies underneath
  speed-{light,dark}.png          time per decomposition against signal length, this library
                                  and the reference code (drawn only with the reference times)
  numbers.json                    the errors and times read off the files below, written
                                  beside this script

Usage: python make_figures.py <run dir> <image dir> [machine name]

The run dir holds, from make_signals.py: burst.csv, burst_truth.csv; from figure_tool:
lib_burst_<seed>.csv and lib_times.csv (the lines "figure_tool time" prints); and, optional,
from run_reference.m: ref_burst_<seed>.csv and ref_times.csv.
"""
import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

run_dir, out_dir = sys.argv[1], sys.argv[2]
machine = sys.argv[3] if len(sys.argv) > 3 else ""

# Colours by role: this library is blue, the reference code is neutral ink, text never takes a
# data colour. Light and dark steps of one palette, checked for contrast on their own surface.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", lib="#2a78d6", ref="#52514e", truth="#e3e2db",
                  lib_1t="#6da7ec", lib_8t="#1c5cab"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", lib="#3987e5", ref="#c3c2b7", truth="#343431",
                 lib_1t="#256abf", lib_8t="#86b6ef"),
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------------------------
def load_modes(path):
    """Columns: IMF 1 .. IMF K, residue."""
    return np.loadtxt(path, delimiter=",")


truth = np.loadtxt(f"{run_dir}/burst_truth.csv", delimiter=",")  # burst, sinusoid


def burst_error(prefix):
    """Relative error of the IMF nearest each true component, mean over the seeds."""
    files = sorted(glob.glob(f"{run_dir}/{prefix}_burst_*.csv"))
    if not files:
        return None
    errors = [[min(np.linalg.norm(m - c) / np.linalg.norm(c) for m in load_modes(p)[:, :-1].T) for c in truth.T]
              for p in files]
    return [round(float(v), 4) for v in np.mean(errors, axis=0)]


lib_ms = {}  # (n, threads) -> best median of the passes
for row in csv.reader(open(f"{run_dir}/lib_times.csv")):
    key = (int(row[0]), int(row[1]))
    lib_ms[key] = min(lib_ms.get(key, float("inf")), float(row[2]))

ref_s = {}  # (signal, n) -> best run
if os.path.exists(f"{run_dir}/ref_times.csv"):
    for row in csv.DictReader(open(f"{run_dir}/ref_times.csv")):
        key = (row["signal"], int(row["n"]))
        ref_s[key] = min(ref_s.get(key, float("inf")), float(row["seconds"]))

lengths = [n for n in (1024, 2048, 4096, 8192) if (n, 1) in lib_ms and (n, 8) in lib_ms]
have_ref_times = all(("tones", n) in ref_s for n in lengths) and bool(lengths)
numbers = {
    "machine": machine,
    "burst": {
        "library_error": burst_error("lib"),
        "reference_error": burst_error("ref"),
        "library_ms_1_thread": lib_ms.get((1000, 1)),
        "library_ms_8_threads": lib_ms.get((1000, 8)),
        "reference_s": ref_s.get(("burst", 1000)),
    },
    "speed": {
        "n": lengths,
        "library_ms_1_thread": [lib_ms[(n, 1)] for n in lengths],
        "library_ms_8_threads": [lib_ms[(n, 8)] for n in lengths],
        "reference_s": [ref_s[("tones", n)] for n in lengths] if have_ref_times else None,
    },
}
json.dump(numbers, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers.json"), "w"), indent=2)


# ---------------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------------
def style_axes(ax, c):
    ax.set_facecolor(c["surface"])
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=c["muted"], labelcolor=c["ink2"], length=3, width=0.8, labelsize=8.5)


def human_time(seconds):
    if seconds >= 100:
        return f"{seconds:.0f} s"
    if seconds >= 1:
        return f"{seconds:.1f} s"
    return f"{seconds * 1000:.0f} ms"


def percent(v):
    return f"{100 * v:.0f} %" if v >= 0.1 else f"{100 * v:.1f} %"


def decomposition(mode):
    c = THEMES[mode]
    x = np.loadtxt(f"{run_dir}/burst.csv")
    n = len(x)
    t = np.arange(1, n + 1)
    b = numbers["burst"]

    def subtitle(err, seconds, threads):
        text = f"error against the true components: burst {percent(err[0])}, sinusoid {percent(err[1])}"
        if seconds:
            text += f"  ·  {human_time(seconds)}" + (f" on {threads} threads" if threads else "")
        return text

    columns = []
    if b["reference_error"]:
        columns.append(("Authors' reference code", subtitle(b["reference_error"], b["reference_s"], 0),
                        f"{run_dir}/ref_burst_1.csv", c["ref"]))
    columns.append(("This library" if columns else "ICEEMDAN of a sinusoid with a short burst",
                    subtitle(b["library_error"], (b["library_ms_8_threads"] or 0) / 1000, 8),
                    f"{run_dir}/lib_burst_1.csv", c["lib"]))

    ncol = len(columns)
    fig, axes = plt.subplots(4, ncol, figsize=(10, 6.1 if ncol > 1 else 5.6), sharex=True, sharey="row",
                             facecolor=c["surface"], squeeze=False)
    rows = ["Signal", "IMF 1", "IMF 2", "The rest"]
    # amplitude of the two true components: the burst lasts from sample 501 to 750
    burst_amp = np.where((t >= 501) & (t <= 750), 1.0, 0.0)
    sine_amp = np.ones(n)
    for j, (title, sub, path, colour) in enumerate(columns):
        modes = load_modes(path)
        series = [x, modes[:, 0], modes[:, 1], modes[:, 2:].sum(axis=1)]
        bands = [None, burst_amp, sine_amp, None]
        for i, (y, band) in enumerate(zip(series, bands)):
            ax = axes[i, j]
            style_axes(ax, c)
            last = i == len(series) - 1
            if band is not None:
                ax.fill_between(t, -band, band, color=c["truth"], linewidth=0, step="mid", zorder=1)
            ax.plot(t, y, color=colour, linewidth=1.0, solid_joinstyle="round", zorder=2)
            if i == 0:
                ax.set_ylim(-2.4, 2.4)
                ax.set_yticks([-2, 0, 2])
            else:
                ax.set_ylim(-1.7, 1.7)
                ax.set_yticks([-1, 0, 1])
            ax.set_xlim(1, n)
            ax.xaxis.set_major_locator(FixedLocator([250, 500, 750, 1000]))
            ax.spines["bottom"].set_visible(last)
            ax.tick_params(bottom=last)
            if j == 0:
                ax.set_ylabel(rows[i], rotation=0, ha="right", va="center", labelpad=12, color=c["ink"], fontsize=10)
        axes[0, j].set_title(title, loc="left", color=c["ink"], fontsize=12, fontweight="semibold", pad=24)
        axes[0, j].text(0, 1.12, sub, transform=axes[0, j].transAxes, color=c["ink2"], fontsize=9, va="bottom")
        axes[-1, j].set_xlabel("sample", color=c["ink2"], fontsize=9)

    # the pale band, named once
    key = axes[1, -1]
    key.fill_between([], [], [], color=c["truth"], linewidth=0, label="amplitude of the true component")
    legend = key.legend(loc="upper right", frameon=False, fontsize=8.5, handlelength=1.4, handleheight=0.9,
                        borderaxespad=0.0)
    for text in legend.get_texts():
        text.set_color(c["ink2"])

    fig.subplots_adjust(left=0.1, right=0.985, top=0.865 if ncol > 1 else 0.855, bottom=0.09, hspace=0.2, wspace=0.06)
    fig.savefig(f"{out_dir}/decomposition-{mode}.png", dpi=200, facecolor=c["surface"])
    plt.close(fig)


def speed(mode):
    c = THEMES[mode]
    s = numbers["speed"]
    lengths = s["n"]
    series = [
        ("Authors' reference code", s["reference_s"], c["ref"]),
        ("This library, 1 thread", [v / 1000 for v in s["library_ms_1_thread"]], c["lib_1t"]),
        ("This library, 8 threads", [v / 1000 for v in s["library_ms_8_threads"]], c["lib_8t"]),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=c["surface"])
    style_axes(ax, c)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    for name, seconds, colour in series:
        ax.plot(lengths, seconds, color=colour, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                marker="o", markersize=7, markeredgecolor=c["surface"], markeredgewidth=1.6, label=name, zorder=3)
        ax.annotate(name, (lengths[-1], seconds[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", ha="left", color=c["ink"], fontsize=9.5)

    ax.set_xlim(lengths[0] / 1.12, lengths[-1] * 1.12)
    ax.xaxis.set_major_locator(FixedLocator(lengths))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{v:,}" for v in lengths])
    lo = min(min(v[1]) for v in series)
    hi = max(max(v[1]) for v in series)
    ticks = [v for v in (0.01, 0.1, 1, 10, 100, 1000) if lo / 3 <= v <= hi * 3]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([human_time(v).replace(".0 s", " s") for v in ticks])
    ax.set_ylim(lo / 2.2, hi * 2.2)
    ax.grid(axis="y", color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("signal length (samples)", color=c["ink2"], fontsize=9.5)

    # the gap, stated once: at the longest signal
    ref_last, lib_last = series[0][1][-1], series[2][1][-1]
    ax.annotate("", xy=(lengths[-1], lib_last * 1.6), xytext=(lengths[-1], ref_last / 1.6),
                arrowprops=dict(arrowstyle="<->", color=c["muted"], linewidth=1, shrinkA=0, shrinkB=0))
    ax.annotate(f"{ref_last / lib_last:,.0f}× faster".replace(",", " "), (lengths[-1], (ref_last * lib_last) ** 0.5),
                xytext=(10, 0), textcoords="offset points", va="center", ha="left", color=c["ink"],
                fontsize=10.5, fontweight="semibold")

    ax.set_title("Time per decomposition", loc="left", color=c["ink"], fontsize=12, fontweight="semibold", pad=24)
    sub = "ensemble of 100, default configuration" + (f"  ·  {machine}" if machine else "")
    ax.text(0, 1.045, sub, transform=ax.transAxes, color=c["ink2"], fontsize=9, va="bottom")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0, 0.99), frameon=False, fontsize=8.5, handlelength=1.8,
                       ncol=3, columnspacing=1.6, borderaxespad=0)
    for text in legend.get_texts():
        text.set_color(c["ink2"])

    fig.subplots_adjust(left=0.085, right=0.70, top=0.84, bottom=0.125)
    fig.savefig(f"{out_dir}/speed-{mode}.png", dpi=200, facecolor=c["surface"])
    plt.close(fig)


made = []
for mode in ("light", "dark"):
    decomposition(mode)
    made.append(f"decomposition-{mode}.png")
    if numbers["speed"]["reference_s"]:
        speed(mode)
        made.append(f"speed-{mode}.png")
print("written to", out_dir + ":", ", ".join(made))
