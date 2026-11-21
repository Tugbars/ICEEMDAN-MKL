# The figure and the timings of the main README

These scripts make the figure of the main README (`docs/img/decomposition-*.png`) and measure
its table of times. Once the authors' code has been run as well, they add its decomposition
beside the library's in the figure and draw the speed comparison (`docs/img/speed.svg`, a
speedup matrix in the style of VectorFFT's heatmaps, with a PNG copy).

Requirements: Python 3 with NumPy and Matplotlib, and a build of this repository
(`cmake --build build` makes `build/figure_tool`). For the authors' code: GNU Octave or MATLAB.

```bash
mkdir run && cd run
python3 ../make_signals.py        # burst.csv, burst_truth.csv, tones_<N>.csv

# this library: the decompositions of the figure, then the times at 1 and 8 threads
for s in 1 2 3 4 5; do
  ../../../build/figure_tool decompose burst.csv lib_burst_$s.csv --seed $s
done
for sig in burst tones_1024 tones_2048 tones_4096 tones_8192; do for t in 1 8; do
  ../../../build/figure_tool time $sig.csv 7 --threads $t           >> lib_times.csv
  ../../../build/figure_tool time $sig.csv 7 --threads $t --no-keep >> lib_times_first.csv
done; done

# the authors' code (optional)
git clone --depth 1 https://github.com/macolominas/CEEMDAN.git ref
cp ref/ceemdan.m ref/emd.m ../run_reference.m .
octave-cli -q run_reference.m     # ref_burst_<seed>.csv, ref_times.csv

python3 ../make_figures.py . ../../../docs/img "name of the machine" "GNU Octave 11.3"
```

`make_figures.py` also writes `numbers.json` beside itself: the errors and times it read, which
are the ones the main README quotes.

The accuracy figure (`docs/img/accuracy-*.png`) is drawn from `accuracy.json`, which holds the
numbers `../compare.py` prints for the authors' code and for this library (see
[`../README.md`](../README.md)). After a new validation run, copy its numbers there.

## What is measured

- **The signal of the figure** is the artificial signal of Torres et al. (2011) and Colominas
  et al. (2014): `sin(2π·0.065·(n−1))` on 1000 samples plus `sin(2π·0.255·(n−501))` on samples
  501 to 750. The error quoted for a component is the relative L2 error of the mode nearest to
  it, averaged over the seeds. The residue counts as a mode: on this signal the authors' code
  stops after IMF 1 and returns the sinusoid as its residue, where this library carries on and
  puts it in IMF 2 with small modes after it.
- **The times** are medians of 7 decompositions after one warm-up, on the sum of tones of the
  validation (8, 32 and 128 cycles on a trend) at four lengths, in the default configuration
  with an ensemble of 100. `figure_tool time` prints `n, threads, median ms, fastest ms, IMFs`;
  when the loop is run more than once, the best median counts.
- **Two times for each length.** The library keeps the decomposed noise of the ensemble between
  calls. `lib_times.csv` is timed that way: what every further signal of a length costs.
  `lib_times_first.csv` is timed with `--no-keep`, which decomposes the noise on every call:
  what the first signal of a length costs. The first-signal time is the one in the figure and
  the one set against the authors' code, which decomposes its noise on every call too.
- **The speed options** of the main README are timed the same way, with
  `--sd --single --circular` added.
- **The authors' code** runs with the same settings (noise 0.2, 100 realizations, at most 5000
  sifting iterations, `SNRFlag` 1) and is timed with `tic`/`toc` around `ceemdan`, twice per
  length; the faster run counts. The times of the main README are from GNU Octave 11.3.

The times of the main README were taken on Windows with the process held on the performance
cores, at high priority, for example:

```bat
start /b /wait /high /affinity FFFF ..\..\..\build\figure_tool.exe time tones_1024.csv 7 --threads 8
start /b /wait /high /affinity FFFF "" "<Octave folder>\mingw64\bin\octave-cli.exe" -q run_reference.m
```

Files in `run/` are not in git.
