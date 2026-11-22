"""The Python module (python/iceemdan.py) against the shared library: everything it offers."""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import iceemdan  # noqa: E402

failures = 0


def check(ok, what):
    global failures
    print(f"  {what:<72} {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1


def raises(kind, f):
    try:
        f()
    except kind:
        return True
    except Exception:
        return False
    return False


rng = np.random.default_rng(4)
n, count = 800, 6
t = np.arange(n) / 250.0
eeg = np.array([np.sin(2 * np.pi * (8 + c % 5) * t) + 0.5 * np.sin(2 * np.pi * (20 + c) * t)
                + (0.2 + 0.1 * (c % 3)) * rng.standard_normal(n) for c in range(count)])
x = eeg[0]

print("The library")
header = open(os.path.join(ROOT, "include", "iceemdan.h"), encoding="utf-8").read()
check(iceemdan.version() == re.search(r'ICEEMDAN_VERSION_STRING "([0-9.]+)"', header).group(1),
      f"version {iceemdan.version()}, as the header")
check(iceemdan.info()["processors"] >= 1, f"processors: {iceemdan.info()['processors']}")

print("One signal")
imfs, residue = iceemdan.decompose(x)
check(imfs.ndim == 2 and imfs.shape[1] == n and residue.shape == (n,), f"decompose(): imfs {imfs.shape}, residue {residue.shape}")
check(np.abs(imfs.sum(axis=0) + residue - x).max() < 1e-10, "the IMFs and the residue add up to the signal")
d = iceemdan.ICEEMDAN()
a, ra = d.decompose(x)
check(np.array_equal(a, imfs) and np.array_equal(ra, residue), "a decomposer gives the same")
facts = d.info()
check(facts["threads"] == iceemdan.info()["processors"] and facts["signals"] == 1 and facts["n"] == n and
      facts["imfs"] == len(imfs) and not facts["noise_kept"] and facts["repaired"] == 0,
      "info(): threads, the last decomposition")
b, rb = d.decompose(x)
check(np.array_equal(b, imfs) and d.info()["noise_kept"], "again: the noise kept, the same result")
check(np.array_equal(iceemdan.decompose(x, ensemble_size=100, seed=42, threads=1)[0], imfs),
      "ensemble_size 100, seed 42, 1 thread: the defaults' result")
other = iceemdan.decompose(x, seed=7)[0]
check(other.shape != imfs.shape or not np.array_equal(other, imfs), "another seed: another result")
two, r2 = iceemdan.decompose(x, max_imfs=2)
check(two.shape == (2, n) and np.abs(two.sum(axis=0) + r2 - x).max() < 1e-10, "max_imfs=2: two IMFs, the rest in the residue")
with iceemdan.ICEEMDAN(fast=True) as fast:
    f_imfs, f_res = fast.decompose(x)
    check(np.abs(f_imfs.sum(axis=0) + f_res - x).max() < 1e-10 and
          (f_imfs.shape != imfs.shape or not np.array_equal(f_imfs, imfs)), "fast=True: another result, which adds up")

print("Many signals")
d40 = iceemdan.ICEEMDAN(ensemble_size=40)
batch = d40.decompose_batch(eeg)
rows = batch.imfs.shape[1]
check(batch.imfs.shape == (count, rows, n) and batch.residues.shape == (count, n) and batch.num_imfs.shape == (count,) and
      rows == batch.num_imfs.max(), f"decompose_batch(): imfs {batch.imfs.shape}, IMFs per channel {batch.num_imfs.tolist()}")
same = True
for c in range(count):
    one_imfs, one_res = d40.decompose(eeg[c])
    k = batch.num_imfs[c]
    same = (same and len(one_imfs) == k and np.array_equal(one_imfs, batch.imfs[c, :k]) and
            np.array_equal(one_res, batch.residues[c]) and not batch.imfs[c, k:].any())
check(same, "every channel bit for bit as decompose() alone; zeros after its own IMFs")
listed = iceemdan.decompose_batch([eeg[0], eeg[1]], ensemble_size=40)
check(np.array_equal(listed.imfs[0, :listed.num_imfs[0]], batch.imfs[0, :batch.num_imfs[0]]),
      "the module function, on a list of arrays")

print("The audit trail, NaN and Inf")
d40.decompose_batch(eeg)
g = d40.diagnostics(2)
k2 = batch.num_imfs[2]
check(g["orthogonality"] >= 0 and g["repaired"] == 0 and all(len(g[key]) == k2 for key in
                                                              ("copies", "iterations_max", "iterations_mean", "unconverged"))
      and set(g["copies"]) == {40}, "diagnostics(signal): per IMF, the copies and the sifting iterations")
holed = x.copy()
holed[[100, 300, 301]] = [np.nan, np.inf, -np.inf]
h_imfs, h_res = d40.decompose(holed)
filled = x.copy()
filled[100] = 0.5 * (x[99] + x[101])
filled[300:302] = x[299] + np.array([1, 2]) / 3 * (x[302] - x[299])
check(d40.info()["repaired"] == 3 and d40.diagnostics()["repaired"] == 3 and
      np.abs(h_imfs.sum(axis=0) + h_res - filled).max() < 1e-10, "3 bad samples: filled in by straight lines and counted")

print("Refused")
huge = iceemdan.ICEEMDAN(ensemble_size=2 ** 30)
try:
    huge.decompose(x)
    message = ""
except ValueError as e:
    message = str(e)
check("GiB" in message and "fast option" in message, "noise that does not fit in half the memory: refused, with the numbers")
check(raises(ValueError, lambda: iceemdan.ICEEMDAN(ensemble_size=-1)) and raises(ValueError, lambda: iceemdan.ICEEMDAN(max_imfs=99))
      and raises(ValueError, lambda: iceemdan.ICEEMDAN(noise=float("nan"))), "a setting out of range: ValueError")
check(raises(ValueError, lambda: d.decompose(eeg)) and raises(ValueError, lambda: d.decompose(x[:3])) and
      raises(ValueError, lambda: d.decompose_batch(x)) and raises(ValueError, lambda: d.decompose_batch(np.zeros((2, 3)))) and
      raises(ValueError, lambda: d40.diagnostics(99)), "a wrong shape, 3 samples, no such signal: ValueError")
d.close()
d.close()
check(raises(ValueError, lambda: d.decompose(x)), "a closed decomposer: ValueError")

print("Hilbert-Huang analysis")
fs = 1000.0
tone_t = np.arange(1000) / fs  # 1 s: 6 and 30 whole cycles
tones = np.array([4.0 * np.cos(2 * np.pi * 6 * tone_t), 1.0 * np.cos(2 * np.pi * 30 * tone_t)])
amp, freq, phase = iceemdan.instantaneous(tones, fs)
check(amp.shape == freq.shape == phase.shape == tones.shape and abs(amp[0, 500] - 4) < 1e-6 and abs(freq[1, 500] - 30) < 1e-6,
      "instantaneous(): amplitude, frequency, phase, shaped like the IMFs")
a1, f1, p1 = iceemdan.instantaneous(tones[0], fs)
check(a1.shape == (1000,) and np.array_equal(a1, amp[0]), "one IMF in, one row out")
mf = iceemdan.mean_frequency(tones, fs)
check(np.allclose(mf, [6, 30], atol=1e-6) and isinstance(iceemdan.mean_frequency(tones[0], fs), float),
      "mean_frequency(): per IMF, and a float for one")
hs = iceemdan.hilbert_spectrum(tones, fs, frequency_bins=16, fmax=64)  # bands of 4 Hz: 6 Hz in band 1, 30 Hz in band 7
check(hs.spectrum.shape == (16, 1000) and hs.frequencies.shape == (16,) and hs.times.shape == (1000,) and
      abs(hs.frequencies[1] - 6) < 1e-12 and abs(hs.times[1] - 0.001) < 1e-12, "hilbert_spectrum(): shapes and axes")
check(abs(hs.marginal[1] - 16000) < 1e-3 and abs(hs.marginal[7] - 1000) < 1e-3 and
      abs(hs.marginal.sum() - 17000) < 1e-3, "the marginal spectrum: each tone's energy in its band, nothing else")
coarse = iceemdan.hilbert_spectrum(tones, fs, frequency_bins=16, fmax=64, time_step=10, amplitude=True)
check(coarse.spectrum.shape == (16, 100) and abs(coarse.times[0] - 0.0045) < 1e-12 and abs(coarse.marginal[1] - 4000) < 1e-3,
      "time_step 10, amplitude: 100 columns centred on their samples")
check(raises(ValueError, lambda: iceemdan.hilbert_spectrum(tones, fs, fmin=50, fmax=40)) and
      raises(ValueError, lambda: iceemdan.instantaneous(tones, 0.0)) and
      raises(ValueError, lambda: iceemdan.instantaneous(np.zeros(1))), "fmin above fmax, no sample rate, one sample: ValueError")

print("ALL PASSED" if not failures else f"FAILED ({failures})")
sys.exit(1 if failures else 0)
