# Validation against the reference ICEEMDAN

Compares this library with the ICEEMDAN code of Colominas et al.
([github.com/macolominas/CEEMDAN](https://github.com/macolominas/CEEMDAN): `ceemdan.m`, which
needs Rilling & Flandrin's `emd.m`, shipped in the same repository). Results are in the main
README, section *Accuracy*.

Requirements: GNU Octave (or MATLAB), Python 3 with NumPy, and a build of this repository
(`cmake --build build` makes `build/decompose_csv`).

```bash
mkdir run && cd run
python3 ../make_signals.py                       # wn.csv (N = 4096), sines.csv (N = 2048)

# reference: ~1-2 minutes per decomposition
git clone --depth 1 https://github.com/macolominas/CEEMDAN.git ref
cp ../run_reference.m ref/ && cp wn.csv sines.csv ref/
(cd ref && octave-cli -q run_reference.m) && cp ref/ref_*.csv .

# this library: default, SD stop rule, single envelope
for s in 1 2 3 4 5; do for sig in wn sines; do
  ../../build/decompose_csv $sig.csv after_${sig}_$s.csv   --seed $s
  ../../build/decompose_csv $sig.csv aftersd_${sig}_$s.csv --seed $s --sd
  ../../build/decompose_csv $sig.csv single_${sig}_$s.csv  --seed $s --sd --single
done; done

python3 ../compare.py
```

What is compared:

- **White noise**: mean (spectral-centroid) frequency of IMFs 1–6. ICEEMDAN of white noise
  behaves as a dyadic filter bank; matching these bands is the core check. The summary number is
  the mean |log2(f_k / f_k,ref)|, in octaves.
- **Sum of tones** (8, 32 and 128 cycles plus a trend): the relative L2 error of the IMF that
  best matches each tone, away from the edges. Lower means less mode mixing.

The two implementations use different random numbers, so results agree statistically, not
sample by sample.
