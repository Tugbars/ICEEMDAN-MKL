"""The library against the authors' code on the same noise, sample by sample.

Usage: python compare_same_noise.py <folder> [library prefix ...]

Reads, in <folder>: same_<signal>.csv from run_same_noise.m (the authors' code) and
<prefix>_<signal>.csv from "same_noise iceemdan" (the library; prefix "lib" if none is given),
both with the modes one per column and the residue last; and same_emd_<N>_<i>.csv against
lib_emd_<N>_<i>.csv (the EMD of one noise realization).

For each mode: the largest difference from the authors' mode, relative to the largest value of
that mode, and where it is. A difference near 1e-12 is rounding; the sifting decides some
stops by thresholds, so rounding can grow to 1e-6 or so in a few trials.
"""
import glob
import os
import sys

import numpy as np

folder = sys.argv[1]
prefixes = sys.argv[2:] or ["lib"]


def load(path):
    a = np.loadtxt(path, delimiter=",")
    return a[:, None] if a.ndim == 1 else a


def compare(ref, lib, label):
    k = min(ref.shape[1], lib.shape[1])
    print(f"  {label}: authors {ref.shape[1] - 1} IMFs + residue, library {lib.shape[1] - 1} IMFs + residue")
    rows = []
    for c in range(k):
        scale = np.max(np.abs(ref[:, c])) or 1.0
        d = np.abs(ref[:, c] - lib[:, c])
        name = "residue" if c == ref.shape[1] - 1 and c == lib.shape[1] - 1 else f"IMF {c + 1}"
        rows.append(f"{name} {d.max() / scale:.1e}")
    print("    " + " | ".join(rows))
    # the whole decomposition: the sum of the modes from the third on, to see where the extra
    # or missing modes went
    if ref.shape[1] != lib.shape[1]:
        tail = min(ref.shape[1], lib.shape[1]) - 1
        a, b = ref[:, tail:].sum(axis=1), lib[:, tail:].sum(axis=1)
        print(f"    modes {tail + 1} to the end, summed: {np.max(np.abs(a - b)) / (np.max(np.abs(a)) or 1):.1e}")


for path in sorted(glob.glob(os.path.join(folder, "same_*.csv"))):
    name = os.path.basename(path)[5:-4]
    if name.startswith("emd_"):
        continue
    ref = load(path)
    print(f"{name}:")
    for p in prefixes:
        lp = os.path.join(folder, f"{p}_{name}.csv")
        if os.path.exists(lp):
            compare(ref, load(lp), p)

emd = sorted(glob.glob(os.path.join(folder, "same_emd_*.csv")))
if emd:
    print("EMD of single noise realizations (emd.m against the library's noise-bank sifting):")
    for path in emd:
        lp = os.path.join(folder, "lib_" + os.path.basename(path)[5:])
        if os.path.exists(lp):
            compare(load(path), load(lp), os.path.basename(path)[9:-4])
