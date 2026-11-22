"""This library beside the authors' reference ICEEMDAN (run in Octave) and beside the true
components of the validation signals: the numbers of the accuracy figure of the main README.

Usage: python compare.py [run dir] [--json accuracy.json]

Reads, in the run directory (default: the working directory), <prefix>_<signal>_<seed>.csv
with the modes one per column and the residue last:

  ref_*      the authors' code                      (run_reference.m)
  after_*    this library, the published algorithm  (decompose_csv)
  fast_*     this library, its fast option          (decompose_csv --sd --single --circular)
  scatter_*  this library over many seeds, to tell how much a number varies from seed to seed

for the signals of make_signals.py (wn, sines, baseline). And, in its folder same_noise/, the
authors' code on the library's own noise, same_<signal>.csv (run_same_noise.m), against
lib_<signal>.csv (the same_noise tool). Sets that are not there are skipped.

--json writes the numbers to a file as well. validation/figures/accuracy.json is that file:
make_figures.py draws the accuracy figure from it.
"""
import glob
import json
import os
import sys

import numpy as np

args = [a for a in sys.argv[1:] if a != '--json']
json_path = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
if json_path in args:
    args.remove(json_path)
run = args[0] if args else '.'

WHO = {'reference': ('ref', "Authors' code (Octave)"), 'library': ('after', 'This library'),
       'fast': ('fast', 'This library, fast option')}


def modes(path):
    """Rows: IMF 1 .. IMF K, the residue."""
    a = np.loadtxt(path, delimiter=',')
    return (a[:, None] if a.ndim == 1 else a).T


def runs(who, signal):
    return [modes(p) for p in sorted(glob.glob(f'{run}/{WHO[who][0]}_{signal}_*.csv'))]


def centroid(m):
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m))
    return (P * f).sum() / P.sum()


def error(a, c, part, own_variation=False):
    """Relative L2 error of a against the component c on the samples of part; own_variation:
    against c without its mean level (a trend, a baseline)."""
    scale = c - c.mean() if own_variation else c
    return np.linalg.norm((a - c)[part]) / np.linalg.norm(scale[part])


def row(label, values, fmt, width=36):
    return f'  {label:{width}s}' + ''.join(fmt.format(v) for v in values)


result = {'seeds': {}}

# ---------------- white noise: the dyadic filter bank, every mode ----------------------------
print('White noise, N = 4096: mean frequency (cycles/sample) of every mode, mean over the seeds')
wn = {}
for who in WHO:
    r = runs(who, 'wn')
    if not r:
        continue
    k = min(len(m) - 1 for m in r)  # the modes every seed has
    wn[who] = np.mean([[centroid(v) for v in m[:k]] for m in r], axis=0)
    result['seeds'][who] = len(r)
    print(row(WHO[who][1], wn[who], '{:8.4f}') + f'   ({len(r)} seeds)')
entry = {'what': 'white noise, N = 4096: mean frequency of every mode in cycles per sample (the modes every seed has)'}
entry.update({who: [round(float(v), 6) for v in f] for who, f in wn.items()})
if 'reference' in wn and 'library' in wn:
    k = min(len(wn['reference']), len(wn['library']))
    d = float(np.mean(np.abs(np.log2(wn['library'][:k] / wn['reference'][:k]))))
    entry['mean_octave_distance'] = round(d, 3)
    print(f'  mean |log2(f_k / f_k,ref)| over the first {k} modes (0 = the same bands): {d:.3f}')
result['white_noise'] = entry

# ---------------- known components ---------------------------------------------------------------
n = 2048
t = np.arange(n) / n
inner = slice(128, n - 128)  # away from the edges
tones = {'128 cycles': 0.5 * np.sin(2 * np.pi * 128 * t), '32 cycles': 2 * np.sin(2 * np.pi * 32 * t),
         '8 cycles': 5 * np.sin(2 * np.pi * 8 * t)}
trend = 100 + 10 * t
rhythm = np.sin(2 * np.pi * 40 * t)
baseline = 1.5 * np.sin(2 * np.pi * 1.5 * t + 0.7) + 2 * t * t


def tones_errors(m):
    # a tone: the nearest IMF. The trend: the nearest mode, the residue included, against the
    # trend's own variation (its level of 100 would hide any error)
    return [min(error(v, c, inner) for v in m[:-1]) for c in tones.values()] + \
           [min(error(v, trend, inner, own_variation=True) for v in m)]


def baseline_errors(m):
    # the rhythm: the nearest IMF. The baseline: the slow modes together, from the best mode on
    tails = [m[k:].sum(axis=0) for k in range(len(m))]
    return [min(error(v, rhythm, inner) for v in m[:-1]),
            min(error(v, baseline, inner, own_variation=True) for v in tails)]


tests = [
    ('tones', 'sines', ['128 cycles', '32 cycles', '8 cycles', 'trend'], tones_errors,
     'tones of 128, 32 and 8 cycles on a trend, N = 2048: relative error of the mode nearest each '
     'tone, and of the mode nearest the trend against the trend\'s own variation; edges excluded'),
    ('baseline', 'baseline', ['rhythm', 'baseline'], baseline_errors,
     'a rhythm of 40 cycles on a baseline that wanders and drifts, N = 2048: relative error of the '
     'mode nearest the rhythm, and of the slow modes together against the baseline\'s own variation; '
     'edges excluded'),
]
for key, signal, names, errors, what in tests:
    print(f'\n{what[0].upper() + what[1:]}')
    print(row('', names, '{:>12s}'))
    entry = {'what': what, 'names': names}
    for who in WHO:
        r = runs(who, signal)
        if not r:
            continue
        e = np.mean([errors(m) for m in r], axis=0)
        entry[who] = [round(float(v), 6) for v in e]
        result['seeds'][who] = max(result['seeds'].get(who, 0), len(r))
        print(row(WHO[who][1], e, '{:12.4f}') + f'   ({len(r)} seeds)')
    result[key] = entry

# ---------------- how many modes; that the modes add up to the signal -----------------------
print('\nNumber of IMFs (the residue not counted), every seed; and the largest difference between')
print('the sum of the modes and the signal, relative to the signal\'s peak')
signals = [('tones', 'sines'), ('baseline', 'baseline'), ('white noise', 'wn')]
entry = {'what': 'number of IMFs of every seed, the residue not counted', 'names': [name for name, _ in signals]}
complete = {}
for who in WHO:
    counts = []
    for name, signal in signals:
        r = runs(who, signal)
        counts.append([len(m) - 1 for m in r])
        if r and os.path.exists(f'{run}/{signal}.csv'):
            x = np.loadtxt(f'{run}/{signal}.csv')
            complete[who] = max([complete.get(who, 0.0)] + [float(np.max(np.abs(m.sum(axis=0) - x)) / np.max(np.abs(x))) for m in r])
    if any(counts):
        entry[who] = counts
        print(row(WHO[who][1], [f'{name}: {" ".join(map(str, c))}' for (name, _), c in zip(signals, counts)], '{:s}   ') +
              (f'sum: {complete[who]:.1e}' if who in complete else ''))
result['modes'] = entry
if 'library' in complete:
    result['completeness'] = float(f"{max(complete.get('library', 0.0), complete.get('fast', 0.0)):.1e}")

# ---------------- the authors' code on the library's noise: sample by sample --------------------
same = sorted(p for p in glob.glob(f'{run}/same_noise/same_*.csv') if not os.path.basename(p).startswith('same_emd_'))
if same:
    print('\nThe same noise: largest difference of each mode from the authors\' mode, relative to its peak')
    entry = {'what': 'the authors\' code on the library\'s noise realizations: largest difference of each mode '
                     '(IMF 1 .., the residue last) from the authors\' mode, relative to that mode\'s peak',
             'names': [], 'difference': [], 'reference_modes': [], 'library_modes': []}
    for path in same:
        name = os.path.basename(path)[5:-4]
        lib = f'{run}/same_noise/lib_{name}.csv'
        if not os.path.exists(lib):
            continue
        a, b = modes(path), modes(lib)
        k = min(len(a), len(b))
        d = [float(np.max(np.abs(a[i] - b[i])) / (np.max(np.abs(a[i])) or 1.0)) for i in range(k)]
        entry['names'].append(name)
        entry['difference'].append([float(f'{v:.1e}') for v in d])
        entry['reference_modes'].append(len(a) - 1)
        entry['library_modes'].append(len(b) - 1)
        print(f'  {name:10s} IMFs {len(a) - 1} (authors) / {len(b) - 1} (library): ' + ' '.join(f'{v:.0e}' for v in d))
    result['same_noise'] = entry

# ---------------- how much each number varies from seed to seed --------------------------------
# On the same noise the two codes return the same modes, so the standard deviation over many
# seeds of this library is the scatter of one seed of either code, and a mean over n seeds
# scatters by that over sqrt(n): the standard error of the authors' mean. A gap of up to about
# two standard errors is what different noise alone produces.
def seeds_of(prefix, signal):
    return [modes(p) for p in sorted(glob.glob(f'{run}/{prefix}_{signal}_*.csv'))]


many = {signal: seeds_of('scatter', signal) for signal in ('sines', 'baseline', 'wn')}
reference = {signal: seeds_of('ref', signal) for signal in many}
if all(many.values()) and all(reference.values()):
    k = min(len(m) - 1 for m in many['wn'] + reference['wn'])  # the bands every seed has
    groups = [
        ('sines', ['tone of 128 cycles', 'tone of 32 cycles', 'tone of 8 cycles', 'trend under the tones',
                   'tones: number of IMFs'], lambda m: tones_errors(m) + [len(m) - 1]),
        ('baseline', ['rhythm on the baseline', 'wandering baseline', 'baseline: number of IMFs'],
         lambda m: baseline_errors(m) + [len(m) - 1]),
        ('wn', [f'white noise: band {i + 1}, in octaves' for i in range(k)] + ['white noise: number of IMFs'],
         lambda m: [np.log2(centroid(v)) for v in m[:k]] + [len(m) - 1]),
    ]
    n_many, n_ref = len(many['sines']), len(reference['sines'])
    print(f'\nSeed-to-seed scatter: this library over {n_many} seeds, and the authors\' mean of {n_ref} seeds in')
    print('standard errors of such a mean (the bands as log2 of the mean frequency)')
    print(row('', ['mean', 'one seed', 'authors', 'std errors'], '{:>12s}'))
    entry = {'what': f'this library over {n_many} seeds: the mean of each number and its standard deviation from seed '
                     f'to seed; the mean of the authors\' {n_ref} seeds, and its distance from the library\'s mean '
                     'in standard errors of a mean of that many seeds. Bands: log2 of the mean frequency (octaves)',
             'seeds': n_many, 'reference_seeds': n_ref, 'names': [], 'mean': [], 'one_seed': [], 'reference': [],
             'standard_errors': []}
    for signal, names, measure in groups:
        v = np.array([measure(m) for m in many[signal]], dtype=float)
        r = np.array([measure(m) for m in reference[signal]], dtype=float)
        mean, sd, ref_mean = v.mean(axis=0), v.std(axis=0, ddof=1), r.mean(axis=0)
        z = (ref_mean - mean) / (sd / np.sqrt(len(r)))
        for j, name in enumerate(names):
            entry['names'].append(name)
            entry['mean'].append(round(float(mean[j]), 6))
            entry['one_seed'].append(round(float(sd[j]), 6))
            entry['reference'].append(round(float(ref_mean[j]), 6))
            entry['standard_errors'].append(round(float(z[j]), 4))
            print(row(name, [mean[j], sd[j], ref_mean[j], z[j]], '{:12.4f}'))
    result['scatter'] = entry
    print(f"  largest distance: {max(abs(v) for v in entry['standard_errors']):.2f} standard errors")

if json_path:
    seeds = result['seeds']
    result = {'source': 'validation/compare.py: the authors\' code run in GNU Octave' +
                        (f" ({seeds['reference']} seeds)" if 'reference' in seeds else '') +
                        f", this library ({seeds.get('library', 0)} seeds)", **result}
    json.dump(result, open(json_path, 'w'), indent=2)
    print(f'\nwritten: {json_path}')
