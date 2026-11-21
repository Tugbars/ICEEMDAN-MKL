"""Compare decompositions with the authors' reference ICEEMDAN (run in Octave).

Reads <prefix>_<signal>_<seed>.csv files from the working directory (or the
directory given as the first argument): ref_* from run_reference.m, after_*,
aftersd_* and single_* from decompose_csv, before_* from an older build.
Missing sets are skipped."""
import glob, numpy as np, os, sys
os.chdir(sys.argv[1] if len(sys.argv) > 1 else '.')

def load(path, ref=False):
    a = np.loadtxt(path, delimiter=',')
    if a.ndim == 1:
        a = a[:, None]
    return a[:, :-1].T  # IMFs only (last column = residue)

def centroid(m):
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m))
    return (P * f).sum() / P.sum()

names = {'ref': 'Reference (authors\' code, Octave)', 'before': 'Before (1.0)',
         'after': 'After: default (Sifted, Rilling)', 'aftersd': 'After: Sifted, SD stop rule',
         'single': 'After: SingleEnvelope, SD'}

# ---------------- white noise: the dyadic filter-bank structure --------------
print('White noise, N = 4096: mean frequency (cycles/sample) of IMF 1..6, mean over seeds')
rows = {}
for key in ['ref', 'before', 'after', 'aftersd', 'single']:
    files = sorted(glob.glob(f'{key}_wn_*.csv'))
    if not files:
        continue
    F = []
    for p in files:
        imfs = load(p)
        F.append([centroid(m) for m in imfs[:6]])
    K = min(len(f) for f in F)
    rows[key] = np.mean([f[:K] for f in F], axis=0)
    print(f"  {names[key]:36s} " + ' '.join(f'{v:7.4f}' for v in rows[key]) + f'   ({len(files)} seeds)')
if 'ref' in rows:
    print('  mean |log2(f_k / f_k,ref)| over IMF 1..6 (0 = same frequency bands as the reference):')
    for key in ['before', 'after', 'aftersd', 'single']:
        if key in rows:
            K = min(len(rows[key]), len(rows['ref']))
            d = np.mean(np.abs(np.log2(rows[key][:K] / rows['ref'][:K])))
            print(f"    {names[key]:34s} {d:.3f}")

# ---------------- known tones: mode mixing --------------------------------------
n = 2048; t = np.arange(n) / n
truth = {'128 cycles': 0.5 * np.sin(2 * np.pi * 128 * t), '32 cycles': 2 * np.sin(2 * np.pi * 32 * t),
         '8 cycles': 5 * np.sin(2 * np.pi * 8 * t)}
inner = slice(128, n - 128)
print('\nSum of tones (8, 32, 128 cycles + trend), N = 2048: relative error of the best single IMF')
print(f"  {'':36s}" + ''.join(f'{k:>12s}' for k in truth))
for key in ['ref', 'before', 'after', 'aftersd', 'single']:
    files = sorted(glob.glob(f'{key}_sines_*.csv'))
    if not files:
        continue
    E = []
    for p in files:
        imfs = load(p)
        E.append([min(np.linalg.norm((m - c)[inner]) / np.linalg.norm(c[inner]) for m in imfs) for c in truth.values()])
    E = np.mean(E, axis=0)
    print(f"  {names[key]:36s}" + ''.join(f'{v:12.3f}' for v in E) + f'   ({len(files)} seeds)')
