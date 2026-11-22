"""The README figures, each in a light and a dark version, and the numbers the README quotes.

  decomposition-{light,dark}.png  the artificial signal of the ICEEMDAN paper (a sinusoid plus a
                                  short burst of a higher frequency) decomposed by this library,
                                  beside the authors' reference code when its output is there;
                                  the amplitude of the true components lies underneath
  accuracy-{light,dark}.png       this library beside the authors' code and beside the true
                                  components, in six panels: the numbers ../compare.py writes
                                  to accuracy.json, and the burst's from the files below
  speed.svg, speed.png            time per decomposition as a speedup matrix over the reference
                                  code, in the style of VectorFFT's heatmaps (black on white, one
                                  version for both themes; drawn only with the reference times)
  numbers.json                    the errors and times read off the files below, written
                                  beside this script

Usage: python make_figures.py <run dir> <image dir> [machine name] [what ran the authors' code]

The run dir holds, from make_signals.py: burst.csv, burst_truth.csv; from figure_tool:
lib_burst_<seed>.csv, lib_times.csv (the lines "figure_tool time" prints) and, optional,
lib_times_first.csv (the same with --no-keep) and fast_burst_<seed>.csv (the fast option);
and, optional, from run_reference.m: ref_burst_<seed>.csv and ref_times.csv.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator

run_dir, out_dir = sys.argv[1], sys.argv[2]
machine = sys.argv[3] if len(sys.argv) > 3 else ""
reference_runtime = sys.argv[4] if len(sys.argv) > 4 else "GNU Octave"  # what ran the authors' code

# Colours by role: this library is blue, its fast option orange (and a diamond), the reference
# code is neutral ink, text never takes a data colour. Light and dark steps of one palette,
# checked for contrast on their own surface.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", lib="#2a78d6", ref="#52514e", truth="#e3e2db", fast="#eb6834",
                  lib_1t="#6da7ec", lib_8t="#1c5cab"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", lib="#3987e5", ref="#c3c2b7", truth="#343431", fast="#d95926",
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
    """Relative error of the mode nearest each true component, mean over the seeds. The residue
    counts as a mode: on this signal the authors' code stops after IMF 1 and returns the
    sinusoid as its residue."""
    files = sorted(glob.glob(f"{run_dir}/{prefix}_burst_*.csv"))
    if not files:
        return None
    errors = [[min(np.linalg.norm(m - c) / np.linalg.norm(c) for m in load_modes(p).T) for c in truth.T]
              for p in files]
    return [round(float(v), 6) for v in np.mean(errors, axis=0)]


def burst_modes(prefix):
    """Number of IMFs of every seed (the residue not counted)."""
    return [load_modes(p).shape[1] - 1 for p in sorted(glob.glob(f"{run_dir}/{prefix}_burst_*.csv"))]


def read_times(name):
    """(n, threads) -> best median of the passes, from the lines "figure_tool time" prints."""
    times = {}
    if os.path.exists(f"{run_dir}/{name}"):
        for row in csv.reader(open(f"{run_dir}/{name}")):
            key = (int(row[0]), int(row[1]))
            times[key] = min(times.get(key, float("inf")), float(row[2]))
    return times


# The library keeps the decomposed noise between calls. lib_times.csv is timed that way: what
# every further signal of a length costs. lib_times_first.csv is timed with --no-keep: what the
# first signal of a length costs, which is the like-for-like time against the reference code.
lib_ms = read_times("lib_times.csv")
first_ms = read_times("lib_times_first.csv")

ref_s = {}  # (signal, n) -> best run
if os.path.exists(f"{run_dir}/ref_times.csv"):
    for row in csv.DictReader(open(f"{run_dir}/ref_times.csv")):
        key = (row["signal"], int(row["n"]))
        ref_s[key] = min(ref_s.get(key, float("inf")), float(row["seconds"]))

lengths = [n for n in (1024, 2048, 4096, 8192) if (n, 1) in lib_ms and (n, 8) in lib_ms]
have_ref_times = all(("tones", n) in ref_s for n in lengths) and bool(lengths)
have_first = all((n, t) in first_ms for n in lengths for t in (1, 8)) and bool(lengths)
numbers = {
    "machine": machine,
    "burst": {
        "library_error": burst_error("lib"),
        "reference_error": burst_error("ref"),
        "fast_error": burst_error("fast"),
        "library_modes": burst_modes("lib"),
        "reference_modes": burst_modes("ref"),
        "library_first_ms_1_thread": first_ms.get((1000, 1)),
        "library_first_ms_8_threads": first_ms.get((1000, 8)),
        "library_ms_1_thread": lib_ms.get((1000, 1)),
        "library_ms_8_threads": lib_ms.get((1000, 8)),
        "reference_s": ref_s.get(("burst", 1000)),
    },
    "speed": {
        "n": lengths,
        "library_first_ms_1_thread": [first_ms[(n, 1)] for n in lengths] if have_first else None,
        "library_first_ms_8_threads": [first_ms[(n, 8)] for n in lengths] if have_first else None,
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
    if seconds >= 0.01:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds * 1000:.1f} ms"


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
            text += f"\n{human_time(seconds)} per decomposition" + (f" on {threads} threads" if threads else "")
        return text

    columns = []
    if b["reference_error"]:
        columns.append(("Authors' reference code", subtitle(b["reference_error"], b["reference_s"], 0),
                        f"{run_dir}/ref_burst_1.csv", c["ref"]))
    # one decomposition from nothing: the time of a first signal
    lib_time = b["library_first_ms_8_threads"] or b["library_ms_8_threads"] or 0
    columns.append(("This library" if columns else "ICEEMDAN of a sinusoid with a short burst",
                    subtitle(b["library_error"], lib_time / 1000, 8),
                    f"{run_dir}/lib_burst_1.csv", c["lib"]))

    ncol = len(columns)
    # IMF 1 and the residue only (the decomposition stopped where the residue needed no more
    # sifting): three rows; otherwise IMF 2 and the sum of the rest as well
    two_only = all(load_modes(path).shape[1] == 2 for _, _, path, _ in columns)
    rows = ["Signal", "IMF 1", "Residue"] if two_only else ["Signal", "IMF 1", "IMF 2", "The rest"]
    height = (5.0 if ncol > 1 else 4.6) if two_only else (6.4 if ncol > 1 else 5.9)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(10, height), sharex=True, sharey="row",
                             facecolor=c["surface"], squeeze=False)
    # amplitude of the two true components: the burst lasts from sample 501 to 750
    burst_amp = np.where((t >= 501) & (t <= 750), 1.0, 0.0)
    sine_amp = np.ones(n)
    for j, (title, sub, path, colour) in enumerate(columns):
        modes = load_modes(path)
        if two_only:
            series = [x, modes[:, 0], modes[:, 1]]
            bands = [None, burst_amp, sine_amp]
        else:
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
        axes[0, j].set_title(title, loc="left", color=c["ink"], fontsize=12, fontweight="semibold", pad=40)
        axes[0, j].text(0, 1.1, sub, transform=axes[0, j].transAxes, color=c["ink2"], fontsize=9, va="bottom",
                        linespacing=1.5)
        axes[-1, j].set_xlabel("sample", color=c["ink2"], fontsize=9)

    # the pale band, named once
    key = axes[1, -1]
    key.fill_between([], [], [], color=c["truth"], linewidth=0, label="amplitude of the true component")
    legend = key.legend(loc="upper right", frameon=False, fontsize=8.5, handlelength=1.4, handleheight=0.9,
                        borderaxespad=0.0)
    for text in legend.get_texts():
        text.set_color(c["ink2"])

    top = (0.8 if ncol > 1 else 0.79) if two_only else (0.84 if ncol > 1 else 0.83)
    fig.subplots_adjust(left=0.1, right=0.985, top=top, bottom=0.1 if two_only else 0.085, hspace=0.2, wspace=0.06)
    fig.savefig(f"{out_dir}/decomposition-{mode}.png", dpi=200, facecolor=c["surface"])
    plt.close(fig)


def speed_matrix():
    """Time per decomposition as a speedup matrix, in the style of VectorFFT's speedup heatmaps
    (src/tools/plots/gen_2d_pow2.py there): Computer Modern, black on white, five steps of blue
    binned by the ratio, the number in every cell. Columns: the signal length. The first row is
    the authors' code, its time per decomposition; below it this library on 1 and 8 threads, for
    the first signal of a length (it decomposes its noise, as the authors' code does on every
    call) and for further signals (the noise kept). A cell holds the speedup over the authors'
    code and, under it, the library's time."""
    s = numbers["speed"]
    lengths = s["n"]
    ref = s["reference_s"]
    one_first = s["library_first_ms_1_thread"] or s["library_ms_1_thread"]
    eight_first = s["library_first_ms_8_threads"] or s["library_ms_8_threads"]
    rows = [("1 thread", one_first), ("8 threads", eight_first),
            ("1 thread", s["library_ms_1_thread"]), ("8 threads", s["library_ms_8_threads"])]

    EDGES = [500, 1000, 2000, 4000]            # speedup bin edges
    BLUE = ["#c5dffe", "#9ec5f4", "#6da7ec", "#2a78d6", "#184f95"]   # VectorFFT's blue arm, palest first
    REF_FILL = "#efefef"

    def step(x):
        return sum(x >= e for e in EDGES)

    def three(v):                              # three significant figures, thousands separated
        if v >= 1000:
            return f"{round(v, -1):,.0f}"
        if v >= 100:
            return f"{v:.0f}"
        return f"{v:.1f}" if v >= 10 else f"{v:.2f}"

    def latency(ms):
        return f"{three(ms / 1000)} s" if ms >= 1000 else f"{three(ms)} ms"

    style = {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
             "axes.unicode_minus": False, "svg.fonttype": "path", "axes.formatter.use_mathtext": True}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(8.0, 5.6), facecolor="white")
        ax = fig.add_axes([0.33, 0.28, 0.6, 0.57])
        ncol, nrow = len(lengths), 1 + len(rows)
        ax.set_xlim(0.5, ncol + 0.5)
        ax.set_ylim(nrow + 0.5, 0.5)
        ax.axis("off")

        # the authors' code: its time
        for j, seconds in enumerate(ref, 1):
            ax.add_patch(Rectangle((j - .5, 0.5), 1, 1, facecolor=REF_FILL, edgecolor="white", lw=1.2))
            ax.text(j, 1.02, latency(seconds * 1000), ha="center", va="center", fontsize=11.5)
        # this library: the speedup, and its time under it
        for i, (_, ms) in enumerate(rows, 2):
            for j, (seconds, t) in enumerate(zip(ref, ms), 1):
                x = seconds * 1000 / t
                ink = "white" if step(x) == 4 else "black"
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=BLUE[step(x)], edgecolor="white", lw=1.2))
                ax.text(j, i - .1, three(x), ha="center", va="center", fontsize=12, color=ink)
                ax.text(j, i + .24, latency(t), ha="center", va="center", fontsize=8.4, color=ink)

        # rules: the authors' code above, first and further signals apart; a thin frame
        for y in (1.5, 3.5):
            ax.plot([0.5, ncol + .5], [y, y], color="black", lw=2.0, solid_capstyle="projecting")
        ax.plot([0.5, ncol + .5, ncol + .5, 0.5, 0.5], [0.5, 0.5, nrow + .5, nrow + .5, 0.5], color="black", lw=0.8)

        # the lengths across the top, the rows down the left
        for j, n in enumerate(lengths, 1):
            ax.text(j, 0.36, f"{n:,}", ha="center", va="bottom", fontsize=10)
        ax.text((ncol + 1) / 2, -0.12, "signal length $N$", ha="center", va="bottom", fontsize=11.5)
        ax.text(0.38, 0.9, "authors' code", ha="right", va="center", fontsize=10)
        ax.text(0.38, 1.22, "GNU Octave", ha="right", va="center", fontsize=8.6, fontstyle="italic", color="0.35")
        for i, (name, _) in enumerate(rows, 2):
            ax.text(0.38, i, name, ha="right", va="center", fontsize=10)

        def vbracket(i0, i1, x, label):
            ax.plot([x + .07, x, x, x + .07], [i0 - .42, i0 - .42, i1 + .42, i1 + .42], color="black", lw=0.7,
                    clip_on=False)
            ax.text(x - .1, (i0 + i1) / 2, label, ha="center", va="center", rotation=90, fontsize=9.5,
                    fontstyle="italic")
        vbracket(2, 3, -0.32, "first signal")
        vbracket(4, 5, -0.32, "further signals")
        ax.text(-0.72, 3.5, "this library", ha="center", va="center", rotation=90, fontsize=11)

        # legend: the blue arm, edges labelled in the printed ratio
        lg = fig.add_axes([0.33, 0.145, 0.6, 0.034])
        lg.axis("off")
        lg.set_xlim(0, 100)
        lg.set_ylim(0, 1)
        w = 9.0
        for n, colour in enumerate(BLUE):
            lg.add_patch(Rectangle((n * w, 0), w, 1, facecolor=colour, edgecolor="white", lw=0.8))
        for n, edge in enumerate(EDGES, 1):
            lg.plot([n * w, n * w], [0, -0.18], color="black", lw=0.5)
            lg.text(n * w, -0.3, f"{edge:,}", ha="center", va="top", fontsize=8.2)
        lg.text(5 * w + 0.8, 0.5, "faster", ha="left", va="center", fontsize=8.8)
        lg.text(0, 1.45, "speedup over the authors' code", fontsize=9, va="bottom")
        notes = ["large number: the speedup; under it, the time",
                 "first signal: the noise decomposed, as the",
                 "authors' code does on every call",
                 "further signals: the noise kept"]
        for k, note in enumerate(notes):
            lg.text(58, 1.45 - 0.9 * k, note, fontsize=8.2, va="bottom", fontstyle="italic")

        fig.text(0.06, 0.042, "ICEEMDAN, ensemble of 100, the default configuration, a sum of tones"
                              + (f"; {machine}." if machine else "."), fontsize=8.6)
        fig.text(0.06, 0.014, f"This library: median of 7 runs, GCC 16.2. The authors' code (ceemdan.m, emd.m): "
                              f"the faster of 2 runs, {reference_runtime}.", fontsize=8.6)
        fig.savefig(f"{out_dir}/speed.svg", facecolor="white")
        fig.savefig(f"{out_dir}/speed.png", dpi=170, facecolor="white")
        plt.close(fig)


def accuracy(mode):
    """Six panels in two rows. Against the authors' code: the mean frequency of every mode of
    white noise, the number of modes, and the authors' code on the library's own noise. Against
    the true components of three signals: the authors' code (an open ring), this library (a
    dot: where it sits in the ring, the two agree) and the library's fast option (a diamond)."""
    c = THEMES[mode]
    a = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy.json")))
    b = numbers["burst"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.9), facecolor=c["surface"],
                             gridspec_kw=dict(width_ratios=[1.05, 1, 1], height_ratios=[1.35, 1]))

    def panel(ax, names, title, sub, slots=None):
        """The rows of a panel, first on top, centred in its slots; returns their heights."""
        slots = slots or len(names)
        rows = (slots - 1) / 2 + (len(names) - 1) / 2 - np.arange(len(names))
        style_axes(ax, c)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.set_yticks(rows)
        ax.set_yticklabels(names, color=c["ink"], fontsize=9)
        ax.set_ylim(-0.7, slots - 0.3)
        ax.grid(axis="x", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=c["ink"], fontsize=11, fontweight="semibold", pad=22)
        ax.text(0, 1.035, sub, transform=ax.transAxes, color=c["ink2"], fontsize=8.5, va="bottom")
        return rows

    def ring(ax, x, y):
        ax.plot(x, y, linestyle="none", marker="o", markersize=10.5, markerfacecolor=c["surface"],
                markeredgecolor=c["ref"], markeredgewidth=1.6, zorder=2)

    def dot(ax, x, y):
        ax.plot(x, y, linestyle="none", marker="o", markersize=5.8, markerfacecolor=c["lib"],
                markeredgecolor=c["lib"], zorder=4)

    def diamond(ax, x, y):
        ax.plot(x, y, linestyle="none", marker="D", markersize=5.6, markerfacecolor=c["fast"],
                markeredgecolor=c["surface"], markeredgewidth=0.9, zorder=3)

    def link(ax, xs, y):
        xs = [v for v in xs if v is not None]
        if len(xs) > 1:
            ax.plot([min(xs), max(xs)], [y, y], color=c["axis"], linewidth=1.1, solid_capstyle="round", zorder=1)

    def label(ax, text):
        ax.set_xlabel(text, color=c["ink2"], fontsize=8.5)

    no_reference = "the authors' code was not run"

    # white noise: every mode's mean frequency, one octave apart
    ax = axes[0, 0]
    wn = a["white_noise"]
    lib, ref = wn.get("library", []), wn.get("reference", [])
    k = max(len(lib), len(ref))
    sub = (f"the {min(len(lib), len(ref))} bands agree within {wn['mean_octave_distance']:g} octave on average"
           if "mean_octave_distance" in wn else no_reference)
    rows = panel(ax, [f"IMF {i}" for i in range(1, k + 1)], "White noise: every band", sub)
    for i in range(min(len(lib), len(ref))):
        link(ax, [lib[i], ref[i]], rows[i])
    ring(ax, ref, rows[:len(ref)])
    dot(ax, lib, rows[:len(lib)])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([2.0 ** -e for e in (2, 4, 6, 8, 10)]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1/4", "1/16", "1/64", "1/256", "1/1024"])
    ax.set_xlim(2.0 ** -11, 0.5)
    label(ax, "mean frequency of the mode, cycles per sample")

    # how many modes: every count a seed gave
    ax = axes[0, 1]
    m = a["modes"]
    counts = {"Burst\n1,000": (b.get("reference_modes") or [], b.get("library_modes") or [])}
    for name, text in (("tones", "Tones\n2,048"), ("baseline", "Baseline\n2,048"), ("white noise", "White noise\n4,096")):
        i = m["names"].index(name)
        counts[text] = (m["reference"][i] if "reference" in m else [], m["library"][i] if "library" in m else [])
    if all(r and v for r, v in counts.values()):
        gap = max(abs(np.mean(r) - np.mean(v)) for r, v in counts.values())
        sub = ("the same counts as the authors' code" if all(set(r) == set(v) for r, v in counts.values())
               else f"mean counts within {gap:.1f} of the authors' code")
    else:
        sub = "the counts of every seed"
    rows = panel(ax, list(counts), "Number of modes", sub)
    for y, (r, v) in zip(rows, counts.values()):
        r, v = sorted(set(r)), sorted(set(v))
        link(ax, r + v, y)
        ring(ax, r, [y] * len(r))
        dot(ax, v, [y] * len(v))
    most = max([1] + [max(r + v) for r, v in counts.values() if r + v])
    ax.set_xlim(0, max(13, most + 1.5))
    ax.xaxis.set_major_locator(FixedLocator(range(0, int(max(13, most + 1.5)) + 1, 2)))
    label(ax, "IMFs (the residue not counted)")

    # the authors' code on the library's noise: every mode's largest difference
    ax = axes[0, 2]
    same = a.get("same_noise", {"names": [], "difference": []})
    names = {"burst": "Burst", "sines": "Tones", "baseline": "Baseline", "wn": "White noise"}
    shown = [s for s in names if s in same["names"]] or list(names)
    worst = max((max(d) for d in same["difference"]), default=None)
    if worst is not None:
        mantissa, exponent = f"{worst:.0e}".split("e")
        power = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        sub = f"no mode differs by more than {mantissa} × 10{power}"
    else:
        sub = no_reference
    rows = panel(ax, [names[s] for s in shown], "Same noise, sample by sample", sub)
    for y, s in zip(rows, shown):
        if s in same["names"]:
            d = np.maximum(same["difference"][same["names"].index(s)], 1e-17)
            link(ax, list(d), y)
            ax.plot(d, [y] * len(d), linestyle="none", marker="o", markersize=5.0, markerfacecolor=c["lib"],
                    markeredgecolor=c["surface"], markeredgewidth=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1e-17, 3)
    ax.xaxis.set_major_locator(FixedLocator([1e-16, 1e-12, 1e-8, 1e-4, 1]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["10\u207b\u00b9\u2076", "10\u207b\u00b9\u00b2", "10\u207b\u2078", "10\u207b\u2074", "1"])
    label(ax, "largest difference of a mode / its peak")

    # known components: how well each is recovered
    def truth(ax, title, names, lib, ref, fast):
        pairs = [(r, v) for r, v in zip(ref or [], lib) if r is not None]
        sub = (f"within {np.ceil(max(abs(r - v) for r, v in pairs) * 100) / 100:g} of the authors' code"
               if pairs else no_reference)
        rows = panel(ax, names, title, sub, slots=4)
        for i, y in enumerate(rows):
            r = ref[i] if ref else None
            f = fast[i] if fast else None
            link(ax, [lib[i], r, f], y)
            if r is not None:
                ring(ax, [r], [y])
            if f is not None:
                diamond(ax, [f], [y])
            dot(ax, [lib[i]], [y])
        ax.set_xscale("log")
        ax.set_xlim(0.0012, 1.4)
        ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["1 %", "10 %", "100 %"])
        label(ax, "error of the recovered component")

    truth(axes[1, 0], "A burst on a sinusoid", ["Burst", "Sinusoid"], b["library_error"], b["reference_error"],
          b.get("fast_error"))
    tones, base = a["tones"], a["baseline"]
    truth(axes[1, 1], "Three tones on a trend", ["128 cycles", "32 cycles", "8 cycles", "Trend"], tones["library"],
          tones.get("reference"), tones.get("fast"))
    truth(axes[1, 2], "Rhythm on a wandering baseline", ["Rhythm", "Baseline"], base["library"], base.get("reference"),
          base.get("fast"))

    # the two sections, and the marks named once
    fig.subplots_adjust(left=0.085, right=0.975, top=0.855, bottom=0.075, wspace=0.46, hspace=0.72)
    for r, text in ((0, "Against the authors' code"), (1, "Against the true components")):
        box = axes[r, 0].get_position()
        fig.text(0.012, box.y1 + (0.082 if r == 0 else 0.09), text, color=c["ink"], fontsize=12.5,
                 fontweight="semibold", va="bottom")
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=10.5, markerfacecolor=c["surface"],
               markeredgecolor=c["ref"], markeredgewidth=1.6, label="Authors' code"),
        Line2D([], [], linestyle="none", marker="o", markersize=5.8, markerfacecolor=c["lib"],
               markeredgecolor=c["lib"], label="This library"),
        Line2D([], [], linestyle="none", marker="D", markersize=5.6, markerfacecolor=c["fast"],
               markeredgecolor=c["surface"], markeredgewidth=0.9, label="This library, fast option"),
    ]
    legend = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.992), frameon=False,
                        fontsize=9, ncol=3, columnspacing=1.6, handletextpad=0.4, borderaxespad=0)
    for text in legend.get_texts():
        text.set_color(c["ink2"])

    fig.savefig(f"{out_dir}/accuracy-{mode}.png", dpi=200, facecolor=c["surface"])
    plt.close(fig)

made = []
for mode in ("light", "dark"):
    decomposition(mode)
    made.append(f"decomposition-{mode}.png")
    accuracy(mode)
    made.append(f"accuracy-{mode}.png")
if numbers["speed"]["reference_s"]:
    speed_matrix()
    made += ["speed.svg", "speed.png"]
print("written to", out_dir + ":", ", ".join(made))
