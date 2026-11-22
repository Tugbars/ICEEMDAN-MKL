# VectorEMD

**ICEEMDAN (Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise), EEMD and EMD in C++, with C and Python interfaces.** The repository, the CMake package and the Python module are named `iceemdan`.

It follows the published algorithm of Colominas et al. (2014), agrees with the authors' own code, and decomposes a signal of 1,024 samples with an ensemble of 100 in 8 ms on eight threads, where the authors' code takes 36 s. Every further signal of that length takes 3 ms.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/decomposition-dark.png">
  <img src="docs/img/decomposition-light.png" width="900" alt="A sinusoid with a short burst of a higher frequency, decomposed by the authors' reference code (left, 21.5 s) and by this library (right, 5.3 ms on 8 threads). Both put the burst in IMF 1 and leave the sinusoid as the residue.">
</picture>

The test signal of the ICEEMDAN paper: a sinusoid with a short burst of a higher frequency, the standard example of mode mixing in plain EMD. Both codes put the burst in IMF 1 and stop there: what is left, the sinusoid, is already a mode and becomes the residue. Both are within 8 % of the true components.

## What is ICEEMDAN?

**Empirical Mode Decomposition (EMD)** splits a nonlinear, non-stationary signal into oscillatory components called Intrinsic Mode Functions (IMFs). It uses no predefined basis: the components come from the data. Its weakness is **mode mixing**: one IMF can hold very different frequencies, as the burst and the sinusoid above would.

| Method | Year | Idea | Limitation |
|--------|------|------|------------|
| **EMD** | Huang 1998 | Adaptive decomposition | Mode mixing |
| **EEMD** | Wu & Huang 2009 | Average the EMD of many noisy copies | Residual noise, incomplete decomposition |
| **CEEMDAN** | Torres 2011 | Add noise stage by stage | Spurious modes, residual noise |
| **ICEEMDAN** | Colominas 2014 | Average the local means of the noisy copies | Cleaner IMFs, complete decomposition |

ICEEMDAN averages the *local mean* of each noisy copy instead of its modes. The IMFs carry less noise, separate better in frequency, and add up to the signal exactly.

## Accuracy

The default configuration is the published algorithm as the authors' own code computes it (`ceemdan.m` with `emd.m`, run in GNU Octave). Given the same noise, it returns the authors' modes sample by sample. With its own noise it agrees with their code statistically: in the frequency bands, in the number of modes, and in how well the true components of test signals are recovered.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/accuracy-dark.png">
  <img src="docs/img/accuracy-light.png" width="900" alt="Six panels. Against the authors' code: on white noise the mean frequencies of the 10 modes, one octave apart, agree within 0.079 octave on average; the number of modes is the same on four signals; and on the same noise no mode of any signal differs from the authors' by more than 5 × 10⁻¹³ of its peak. Against the true components of three signals (a burst on a sinusoid, three tones on a trend, a rhythm on a wandering baseline): the error of this library is within 0.03 of that of the authors' code on every component, and its fast option is further from the truth on 7 of the 8.">
</picture>

<details>
<summary>The numbers</summary>

The same noise: the authors' code run on this library's noise realizations. Largest difference of any mode from the authors' mode, relative to the mode's peak.

| Signal | Modes, both codes | Largest difference |
|---|---|---|
| Burst on a sinusoid | 1 IMF and the residue | 1 × 10⁻¹⁵ |
| Tones on a trend | 6 IMFs and the residue | 5 × 10⁻¹³ |
| Rhythm on a wandering baseline | 8 IMFs and the residue | 5 × 10⁻¹⁴ |
| White noise | 11 IMFs and the residue | 3 × 10⁻¹⁵ |

White noise, N = 4096: mean frequency of every mode, in cycles per sample.

| | IMF 1 | IMF 2 | IMF 3 | IMF 4 | IMF 5 | IMF 6 | IMF 7 | IMF 8 | IMF 9 | IMF 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Authors' code | 0.3421 | 0.1568 | 0.0788 | 0.0422 | 0.0201 | 0.0095 | 0.0053 | 0.0028 | 0.0020 | 0.0008 |
| This library | 0.3422 | 0.1568 | 0.0789 | 0.0422 | 0.0203 | 0.0095 | 0.0055 | 0.0027 | 0.0017 | 0.0006 |

Number of IMFs over the seeds (the residue not counted).

| | Authors' code | This library |
|---|---|---|
| Burst on a sinusoid, 1,000 | 1 | 1 |
| Tones on a trend, 2,048 | 6 to 7 | 6 to 7 |
| Rhythm on a wandering baseline, 2,048 | 7 to 8 | 7 to 8 |
| White noise, 4,096 | 10 to 11 | 10 to 11 |

True components: relative error of the recovered component, edges excluded.

| | Authors' code | This library | This library, fast option |
|---|---|---|---|
| Burst | 7.7 % | 7.7 % | 59.1 % |
| Sinusoid under the burst | 3.8 % | 3.9 % | 29.3 % |
| Tone of 128 cycles | 26.9 % | 26.5 % | 41.0 % |
| Tone of 32 cycles | 30.2 % | 30.6 % | 52.1 % |
| Tone of 8 cycles | 5.8 % | 8.1 % | 18.0 % |
| Trend under the tones | 5.8 % | 7.7 % | 7.4 % |
| Rhythm on the baseline | 5.0 % | 5.5 % | 7.2 % |
| Wandering baseline | 0.3 % | 0.3 % | 1.0 % |

</details>

With their own random numbers the two codes agree statistically: the first six bands of white noise within 0.005 octave on average, all 10 within 0.079, the last ones varying most because not every seed has them. The modes and the residue add up to the signal to rounding (2 × 10⁻¹⁶). A baseline slower than two cycles is not a single mode; its error is that of the slow modes summed. Method: [`validation/README.md`](validation/README.md).

## Speed

![Time per decomposition as a matrix over signal length 1,024 to 8,192. The authors' code takes 35.7 to 111 s. For the first signal of a length, this library is 778 to 169 times faster on one thread and 4,680 to 1,020 times faster on eight; for further signals, 1,930 to 616 times on one thread and 10,860 to 3,460 times on eight.](docs/img/speed.svg)

The authors' code (`ceemdan.m` with `emd.m`, at the same settings) ran in GNU Octave 11.3 on the same cores. Like the library's first signal of a length, it decomposes the noise of the ensemble on every call. The library keeps that noise, so further signals of the same length and settings skip the step, with the same result as they would get alone.

<details>
<summary>The numbers</summary>

Time per decomposition, ensemble of 100, default configuration:

| Signal length | Authors' code | First signal, 1 thread | First signal, 8 threads | Further signals, 1 thread | Further signals, 8 threads |
|---------------|---------------|------------------------|-------------------------|---------------------------|----------------------------|
| 1,024 | 35.7 s | 46.0 ms | 7.64 ms | 18.5 ms | 3.29 ms |
| 2,048 | 44.3 s | 100 ms | 17.2 ms | 36.5 ms | 6.34 ms |
| 4,096 | 64.9 s | 240 ms | 43.7 ms | 78.5 ms | 14.8 ms |
| 8,192 | 111 s | 657 ms | 108 ms | 180 ms | 32.0 ms |

Intel Core i9-14900KF, threads on the performance cores, built with GCC 16.2; median of 7 runs on a sum of tones for the library, the faster of 2 runs for the authors' code. The measurement is described in [`validation/figures/README.md`](validation/figures/README.md).

</details>

The speed options (a simpler stop rule, one envelope mean for each noisy copy, one shared noise decomposition) are 8 to 12 times faster again on further signals: 0.4 ms for 1,024 samples on 8 threads. They no longer reproduce the published algorithm: the accuracy figure shows what they give up.

Many signals of one length in one call, such as the channels of a recording, are 1.1 to 1.5 times faster than one call per signal (8 to 128 signals of 1,000 to 4,096 samples, on 8 or 16 threads), with the same results bit for bit. `benchmarks/batch_bench.cpp` measures it.

## Features

**The decomposition**

- **ICEEMDAN as published**, by default, computed the way the authors' code computes it: its stop rule for sifting, its boundary extension, cubic spline envelopes, independent noise for every copy.
- **EMD and EEMD** as well, in a header of their own: plain EMD by the rules of `emd.m` (Rilling, Flandrin and Gonçalves), the EMD the ICEEMDAN authors' code is built on, and EEMD as the mean of such EMDs of noisy copies.
- **Complete**: the IMFs and the residue add up to the signal to rounding.
- **Reproducible**: for a given seed, the same result bit for bit at any number of threads, for ICEEMDAN and EEMD alike. The noise is the library's own (Philox4x32-10 and the inverse normal distribution function) and comes out the same bit for bit from GCC and MSVC, on Windows and Linux.
- **Gaps in the data**: NaN and Inf samples are filled in from their neighbours by straight lines, and counted.
- **Batches and channels**: one call decomposes many signals of one length, such as the channels of an EEG or its epochs, from C++, C and Python. They share one noise bank, and their trials are interleaved on the threads, so no thread waits for the end of one signal's stage; every result is the same, bit for bit, as decomposing that signal alone. The decomposed noise is also kept between calls, so further signals of the same length take half the time or less.
- **A fast option** for when exact agreement with the paper matters less than time: a simpler stop rule, one envelope mean for each noisy copy and one shared noise decomposition, 8 to 12 times faster.
- **Never another algorithm than the one asked for**: a signal whose decomposed noise would not fit in half of the machine's memory is refused, with the numbers, instead of being computed some other way.
- **Diagnostics** for every IMF: sifting iterations, trials that did not converge, the number of trials averaged, the orthogonality index.
- **Hilbert–Huang analysis**: the instantaneous amplitude, frequency and phase of every IMF, the Hilbert spectrum (energy over time and frequency) and the marginal spectrum, from C++, C and Python.

**The library**

- **Three interfaces with the same settings and the same results**:
  - C++17, header-only: [`include/iceemdan.hpp`](include/iceemdan.hpp) (example: [`examples/cpp/decompose.cpp`](examples/cpp/decompose.cpp));
  - C, eight functions in one header: [`include/iceemdan.h`](include/iceemdan.h) (example: [`examples/c/decompose.c`](examples/c/decompose.c)), built as a shared library that any language can call;
  - Python, a module on NumPy arrays: [`python/iceemdan.py`](python/iceemdan.py).

  EMD and EEMD have headers of the same shape, [`include/emd.hpp`](include/emd.hpp) and [`include/emd.h`](include/emd.h), and `emd()` and `eemd()` in Python.
- **Multithreaded without OpenMP**: the ensemble runs on the library's own thread pool. The thread count is one setting, and idle threads use no processor.
- **Little to install**: a C++17 compiler and CMake. The splines, the noise and the threads are the library's own; the one library it uses, [VectorFFT](https://github.com/Tugbars/VectorFFT) for the FFTs, is downloaded by CMake as a ready-built binary.
- **Windows and Linux** on x86-64 processors with AVX2. Developed and tested with GCC (MinGW included); it also builds with MSVC, and CMake has configurations for Clang and Intel ICX.
- **Tested**: regression tests run with `ctest`, and the validation against the authors' code is in the repository.

## References

1. **ICEEMDAN**: Colominas, M. A., Schlotthauer, G., & Torres, M. E. (2014). Improved complete ensemble EMD: A suitable tool for biomedical signal processing. *Biomedical Signal Processing and Control*, 14, 19-29.

2. **CEEMDAN**: Torres, M. E., Colominas, M. A., Schlotthauer, G., & Flandrin, P. (2011). A complete ensemble empirical mode decomposition with adaptive noise. *ICASSP 2011*.

3. **EEMD**: Wu, Z., & Huang, N. E. (2009). Ensemble empirical mode decomposition: a noise-assisted data analysis method. *Advances in Adaptive Data Analysis*, 1(01), 1-41.

4. **EMD**: Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. *Proceedings of the Royal Society A*, 454(1971), 903-995.

5. **Sifting stop rule and boundary extension**: Rilling, G., Flandrin, P., & Gonçalves, P. (2003). On empirical mode decomposition and its algorithms. *IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing (NSIP-03)*.

6. **The noise generator**: Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. (2011). Parallel random numbers: as easy as 1, 2, 3. *SC '11: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*.

7. **The inverse normal distribution function**: Wichura, M. J. (1988). Algorithm AS 241: The percentage points of the normal distribution. *Applied Statistics*, 37(3), 477-484.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{vectoremd,
  author = {Tugbars Heptaskin},
  title = {VectorEMD: ICEEMDAN, EEMD and EMD in C++},
  year = {2025},
  url = {https://github.com/Tugbars/iceemdan}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

- Colominas, Schlotthauer and Torres for the ICEEMDAN algorithm and its reference code
- Rilling, Flandrin and Gonçalves for `emd.m`, whose stop rule and boundary extension the defaults follow
- [VectorFFT](https://github.com/Tugbars/VectorFFT) for the FFTs
