# Validation against the reference ICEEMDAN

Compares this library with the ICEEMDAN code of Colominas et al.
([github.com/macolominas/CEEMDAN](https://github.com/macolominas/CEEMDAN): `ceemdan.m`, which
needs Rilling & Flandrin's `emd.m`, shipped in the same repository) and with the true components
of test signals. Results are in the main README, section *Accuracy*.

Requirements: GNU Octave (or MATLAB), Python 3 with NumPy, and a build of this repository
(`cmake --build build` makes `build/decompose_csv` and `build/same_noise`).

```bash
mkdir run && cd run
python3 ../make_signals.py                       # wn.csv (N = 4096), sines.csv and baseline.csv (N = 2048)

# reference: ~1-2 minutes per decomposition
git clone --depth 1 https://github.com/macolominas/CEEMDAN.git ref
cp ../run_reference.m ref/ && cp wn.csv sines.csv baseline.csv ref/
(cd ref && octave-cli -q run_reference.m) && cp ref/ref_*.csv .

# this library: the published algorithm, and its fast option
for s in 1 2 3 4 5; do for sig in wn sines baseline; do
  ../../build/decompose_csv $sig.csv after_${sig}_$s.csv --seed $s
  ../../build/decompose_csv $sig.csv fast_${sig}_$s.csv  --seed $s --sd --single --circular
done; done

# this library over many seeds: how much each number varies from seed to seed (under a minute)
for s in $(seq 1 200); do for sig in wn sines baseline; do
  ../../build/decompose_csv $sig.csv scatter_${sig}_$s.csv --seed $s
done; done

# the same noise: the authors' code on the noise realizations of this library
mkdir same_noise && cd same_noise
cp ../ref/ceemdan.m ../ref/emd.m ../../same_noise/randn.m ../../same_noise/run_same_noise.m .
cp ../wn.csv ../sines.csv ../baseline.csv . && python3 ../../figures/make_signals.py   # and burst.csv
for n in 1000 2048 4096; do
  ../../../build/same_noise noise $n 100 1 noise_$n.csv
  for i in 1 2 3 4 5; do ../../../build/same_noise emd $n 1 $((i - 1)) lib_emd_${n}_$i.csv; done
done
for sig in burst sines baseline wn; do
  ../../../build/same_noise iceemdan $sig.csv lib_$sig.csv --seed 1
done
octave-cli -q run_same_noise.m                   # same_<signal>.csv, same_emd_<N>_<i>.csv
python3 ../../same_noise/compare_same_noise.py . # every mode, and the EMD of single realizations
cd ..

python3 ../compare.py . --json ../figures/accuracy.json
```

`compare.py` prints the numbers below; with `--json` it also writes them to
`figures/accuracy.json`, from which the accuracy figure of the main README is drawn
([`figures/README.md`](figures/README.md)).

What is compared:

- **White noise**: the mean (spectral-centroid) frequency of every mode. ICEEMDAN of white noise
  behaves as a dyadic filter bank; matching these bands is the core check. The summary number is
  the mean |log2(f_k / f_k,ref)|, in octaves. The last modes vary most from seed to seed: they
  are the ones a seed may or may not have.
- **Number of modes** of every seed, on the three signals.
- **Sum of tones** (8, 32 and 128 cycles plus a trend): the relative L2 error of the IMF that
  best matches each tone, away from the edges. Lower means less mode mixing. For the trend: the
  error of the mode nearest to it (the residue is one), against the trend's own variation.
- **A rhythm on a wandering baseline** (40 cycles on one and a half slow cycles plus a drift):
  the error of the IMF nearest the rhythm, and of the slow modes summed, from the best mode on,
  against the baseline's own variation. A baseline slower than two cycles is not one mode; what
  matters to someone who removes it is the sum.
- **The fast option** of this library on the same signals: what it gives up for its speed.
- **The scatter of one seed**: the standard deviation of each number over the 200 seeds of this
  library. A mean of n seeds scatters by that over √n, its standard error; the gap between the
  authors' mean and the library's is given in standard errors of the authors' mean. Different
  noise alone produces gaps of up to about two. More seeds in `run_reference.m` shrink the gaps
  with the square root of their number.
- **The same noise**: in the folder `same_noise`, `randn.m` stands in for Octave's `randn` and
  hands the unchanged `ceemdan.m` the noise realizations this library draws for seed 1. Both then
  decompose the same signal with the same noise, and every mode is compared sample by sample:
  the largest difference, relative to the mode's peak. Differences near 10⁻¹³ are rounding.

With their own random numbers the two implementations agree statistically; on the same noise,
sample by sample. The authors' code is called with at most 5000 sifting iterations for a local
mean and decomposes its noise with `emd.m`'s default of 2000; this library allows 2000 for both.
No trial of these signals needs more than a few hundred.
