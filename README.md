# ICEEMDAN-MKL
**High-Performance EEMD/ICEEMDAN Implementation using Intel MKL**

The first public C/C++ implementation of Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (ICEEMDAN). Optimized for both research reproducibility and production performance.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is ICEEMDAN?

**Empirical Mode Decomposition (EMD)** decomposes nonlinear, non-stationary signals into oscillatory components called Intrinsic Mode Functions (IMFs): no predefined basis functions, fully data-driven. However, EMD suffers from **mode mixing**: a single IMF may contain wildly different frequencies.

| Method | Year | Key Innovation | Limitation |
|--------|------|----------------|------------|
| **EMD** | Huang 1998 | Adaptive decomposition | Mode mixing |
| **EEMD** | Wu & Huang 2009 | Ensemble averaging with white noise | Residual noise, incomplete decomposition |
| **CEEMDAN** | Torres 2011 | Adds noise to each IMF stage separately | Spurious modes, residual noise |
| **ICEEMDAN** | Colominas 2014 | Computes local means of noise realizations | Cleaner IMFs, near-zero reconstruction error |

**ICEEMDAN** improves on CEEMDAN by adding noise in a fundamentally different way:
- Instead of adding noise to the signal, it computes the **local mean of noise realizations**
- This yields IMFs with less noise contamination and better spectral separation
- Reconstruction error drops from ~1% (CEEMDAN) to **floating-point rounding** (~10⁻¹⁴)

For financial and scientific applications, this means cleaner trend extraction, better denoising, and more reliable frequency analysis.

---

## Features

- **Complete ICEEMDAN Algorithm** — Full implementation of Colominas et al. (2014), checked against the authors' own reference code (see [Accuracy](#accuracy))
- **Intel MKL Acceleration** — Spline interpolation via MKL Data Fitting, VSL random number generation
- **VectorFFT** — FFTs (Hilbert transform, instantaneous frequency, spectral entropy) via [VectorFFT](https://github.com/Tugbars/VectorFFT)
- **Reproducible** — for a given seed, identical results at any number of threads
- **Multiple Spline Methods** — Cubic (smooth), Akima (outlier-resistant), Linear (fast)
- **OpenMP Parallelization** — Scales across CPU cores with lock-free reduction
- **Multiple Processing Modes** — Standard (audio/seismic), Finance (trading), Scientific (reproducible research)
- **Header-Only** — headers only; the FFTs link VectorFFT's ready-built library (CMake downloads and links it)
- **Cross-Platform** — Windows and Linux on x86-64 (GCC, Clang, ICX, MSVC)

## Changes in 1.1

Version 1.1 makes the default decomposition follow the published algorithm and the authors'
reference code. **Results differ from 1.0**, and the default is slower (see [Performance](#performance)):

| Area | 1.0 | 1.1 default |
|------|-----|-------------|
| Local mean M(·) | one envelope mean | first mode sifted to convergence, M(x) = x − E₁(x) |
| Sifting stop rule | SD < 0.05 (2–3 iterations) | Rilling, Flandrin & Gonçalves (2003), as `emd.m` |
| Noise amplitude | ε·std(x) at every stage | β₀ = ε·std(x)/std(E₁(w)), β_k = ε·std(r_k) |
| Noise realizations | windows of one long realization | independent realizations, the same one at every stage |
| Envelope ends | mirrored extrema + linear end points; natural splines | symmetric extension of `emd.m`; not-a-knot splines |
| Threads | results depended on the thread count | identical at any thread count |
| FFT | MKL DFTI | VectorFFT |

The 1.0 behaviour is still available: see [Speed options](#speed-options). Also fixed in 1.1:
compilation with GCC/Clang, NaN/Inf sanitizing under `-ffast-math`, the Hilbert transform for
odd lengths, flat extrema (tick-quantized prices), per-IMF convergence diagnostics, a memory cap
for the noise bank, and loading the Python binding on Linux.

---

## Performance

Time per decomposition, ensemble size 100, sum-of-tones test signal, median of 5 runs, on a
2-core Intel Xeon @ 2.1 GHz (AVX-512), 2 threads:

| Signal Length | Default (paper-exact) | Sifted, SD stop rule | SingleEnvelope, SD | SingleEnvelope, SD, circular bank |
|---------------|-----------------------|----------------------|--------------------|-----------------------------------|
| 1024 | 122.5 ms | 16.6 ms | 12.6 ms | 10.9 ms |
| 2048 | 223.6 ms | 41.4 ms | 28.0 ms | 25.6 ms |
| 4096 | 566.2 ms | 90.6 ms | 56.5 ms | 46.3 ms |
| 8192 | 1480.9 ms | 188.7 ms | 124.7 ms | 113.0 ms |

The default follows the published algorithm, whose stop rule typically needs 10–15 sifting
iterations per local mean. The last column is close to the configuration of versions before
1.1 (113 ms at N = 8192 on the same machine), which was faster but did not reproduce
ICEEMDAN (see [Accuracy](#accuracy) and [Speed options](#speed-options)).

*Earlier versions of this README quoted 6.6–61.6 ms on an Intel Core i9-14900KF with 8 threads;
those numbers were measured with the pre-1.1 algorithm.*

---

## Examples

### ICEEMDAN Decomposition

<img width="1153" height="1389" alt="ICEEMDAN_DECOMPOSITION" src="https://github.com/user-attachments/assets/3adf61ed-d5c2-45c1-8ab0-bc9a90dc0e28" />

**Signal**: Simulated GARCH(1,1) price series exhibiting volatility clustering — note the increased fluctuations around samples 1800-2000.

| IMF | Frequency | Trading Interpretation |
|-----|-----------|----------------------|
| **IMF 0** | Highest | Market microstructure noise, bid-ask bounce. Amplitude scales with volatility. |
| **IMF 1** | High | Tick-level noise, HFT activity. Still shows volatility clustering. |
| **IMF 2** | Medium-High | Intraday oscillations, mean-reversion at short scales. |
| **IMF 3** | Medium | Swing patterns, ~10-15 bar cycles. Short-term momentum. |
| **IMF 4-5** | Low | Multi-day/week cycles. Institutional rebalancing, options expiry effects. |
| **IMF 6-7** | Very Low | Monthly/quarterly cycles. Earnings, macro regimes. |
| **Residue** | Trend | Underlying drift component. |

Each IMF captures roughly half the frequency of the previous, octave-spaced bands without manual tuning.

---

### Trading Application: Denoising

<img width="1393" height="890" alt="ICEEMDAN_DENOISING" src="https://github.com/user-attachments/assets/ada9fa0f-47c9-4f91-b3d8-b3b6ec2533f3" />

**Top**: Original price (blue) vs smoothed (red) with high-frequency IMFs removed — ideal for trend-following.

**Middle**: Detrended price oscillates around zero — suitable for mean-reversion strategies.

**Bottom**: Extracted noise (IMF 0-1). Note the **heteroskedasticity** — amplitude increases during high-vol regimes (samples 1500-2000). For volatility estimation, this "noise" is the signal.

---

### Reconstruction Verification

<img width="1389" height="590" alt="ICEEMDAN_RECONSTRUCTION" src="https://github.com/user-attachments/assets/e70d26ad-5e3b-4720-a0db-40e25611d8d2" />

Proof that ICEEMDAN is a **complete decomposition**: ∑IMFs + Residue = Original Signal.

- **Max error**: at floating-point rounding level (about 10⁻¹⁴ relative in the test suite)
- **No information loss**: Selectively reconstruct using any IMF subset

---

### 3D Hilbert Spectrum

<img width="1123" height="989" alt="ICEEMDAN_HILBERT" src="https://github.com/user-attachments/assets/218252fd-e6d5-4e98-a614-94b0510b9c03" />

Time-frequency energy distribution via Hilbert-Huang Transform.

- **Yellow region**: Trend component, >95% of energy
- **Terraced ridges**: Clean frequency separation per IMF
- **Energy bulge at t=1500-2000**: GARCH volatility clustering visible across all bands
- **1/f decay**: Characteristic pink noise structure of financial markets

---

## Quick Start

### Prerequisites

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+, Intel ICX)
- Intel oneAPI MKL (free download from Intel)
- CMake 3.20+
- [VectorFFT](https://github.com/Tugbars/VectorFFT) for the FFTs: CMake downloads its ready-built
  library (Windows x64 or Linux x64, a CPU with AVX2 and FMA), nothing to install

### Installation

```bash
git clone https://github.com/Tugbars/iceemdan-mkl.git
cd iceemdan-mkl
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build      # regression tests
```

The first configure downloads the VectorFFT 0.8.1 release (25 MB), checks its SHA-256 and unpacks
it into `third_party/vectorfft-0.8.1/`, which git ignores; later configures, in any build folder,
use that copy. CMake prints what it links (`FFT backend: ...`). Two options take VectorFFT from
elsewhere:

- `-DVECTORFFT_ROOT=<folder>`: an unpacked release you already have (no download)
- `-DVectorFFT_DIR=<prefix>/lib/cmake/VectorFFT`: a VectorFFT you built from source, which is also
  the way on a platform the release does not cover

On Windows the build copies `libvfft.dll`, `libvfft_avx2.dll` and `libvfft_avx512.dll` beside the
programs and into `python/`. Keep the three together with any program you move: `libvfft.dll`
loads one of the other two, by what the CPU runs.

MKL is taken from `-DMKL_DIR=<folder of MKLConfig.cmake>` when you pass it, else from `MKLROOT`
(oneAPI's `setvars` sets it), else from oneAPI's standard install folder, else from CMake's own
search. With MinGW GCC on Windows the build links MKL's single dynamic library (`mkl_rt`); the
compiler must come with its OpenMP runtime (libgomp).

### Basic Usage

```cpp
#include "iceemdan_mkl.hpp"

int main() {
    // Your signal data
    std::vector<double> signal(1024);
    // ... fill signal ...
    
    // Decompose
    eemd::ICEEMDAN decomposer;
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    
    decomposer.decompose(signal.data(), signal.size(), imfs, residue);
    
    // imfs[0] = highest frequency component
    // imfs[n-1] = lowest frequency component
    // residue = trend
    
    return 0;
}
```

## Processing Modes

Three built-in modes optimize for different use cases:

### Standard Mode (Default)
Best for: Audio processing, seismic analysis, biomedical signals

```cpp
eemd::ICEEMDAN decomposer;  // Standard mode by default
// or explicitly:
eemd::ICEEMDAN decomposer(eemd::ProcessingMode::Standard);
```

- Cubic splines (smooth C² interpolation, not-a-knot ends)
- Global volatility scaling
- Symmetric boundary extension (as in Rilling & Flandrin's `emd.m`)

### Scientific Mode
Best for: Research requiring reproducibility and audit trails

```cpp
eemd::ICEEMDAN decomposer(eemd::ProcessingMode::Scientific);
decomposer.config().rng_seed = 42;  // Fixed seed for reproducibility

eemd::DecompositionDiagnostics diag;
decomposer.decompose_with_diagnostics(signal, n, imfs, residue, diag);

// Audit trail
std::cout << "Seed used: " << diag.rng_seed_used << "\n";
std::cout << "Orthogonality index: " << diag.orthogonality_index << "\n";
```

- Cubic splines
- No variance reduction tricks (pure algorithm): independent noise realizations, no antithetic pairs
- Convergence diagnostics per IMF (see [Diagnostics](#diagnostics))

### Finance Mode
Best for: Quantitative trading, real-time analysis

```cpp
eemd::ICEEMDAN decomposer(eemd::ProcessingMode::Finance);

// Features:
// - Akima splines (outlier-resistant, no overshoot on gaps/crashes)
// - EMA volatility scaling (adapts to regime changes)
// - AR(1) boundary extrapolation (causal, no look-ahead)
// - NaN/Inf sanitization (handles bad data feeds)
// - Convergence tracking
```

**Why Akima splines for finance?**

Standard cubic splines are smooth but "overshoot" on sharp price moves (flash crashes, earnings gaps). This creates phantom high-frequency oscillations in IMF 1-2 that don't exist in the data.

Akima splines use local 5-point slope estimation — they're virtually immune to distant outliers:

| Spline | Continuity | Behavior on Gaps | Best For |
|--------|------------|------------------|----------|
| Cubic | C² (smooth) | Overshoots | Smooth signals |
| **Akima** | C¹ (local) | Tight, stable | Financial data |
| Linear | C⁰ | No overshoot | Very sparse extrema |

## Configuration

### Common Parameters

```cpp
eemd::ICEEMDAN decomposer;
auto& cfg = decomposer.config();

cfg.ensemble_size = 100;      // Number of noise realizations (default: 100)
cfg.noise_std = 0.2;          // Noise amplitude as fraction of signal std (default: 0.2)
cfg.max_imfs = 10;            // Maximum IMFs to extract (default: 10)
cfg.max_sift_iters = 100;     // Sifting iterations per IMF (default: 100)
cfg.sift_threshold = 0.05;    // Sifting convergence threshold (default: 0.05)
cfg.rng_seed = 42;            // Random seed for reproducibility (default: 42)
```

### Spline Methods

Controls envelope interpolation:

```cpp
// Cubic: Smooth C² spline (default for Standard/Scientific)
cfg.spline_method = eemd::SplineMethod::Cubic;

// Akima: Local C¹ spline, outlier-resistant (default for Finance)
cfg.spline_method = eemd::SplineMethod::Akima;

// Linear: Simple linear interpolation (fastest, for very sparse extrema)
cfg.spline_method = eemd::SplineMethod::Linear;
```

**Note:** Akima requires minimum 5 data points and strict geometric constraints. If these aren't met, the implementation automatically falls back to Cubic splines for that envelope — no manual handling needed.

### Volatility Methods

Controls how noise is scaled across the signal:

```cpp
// Global: Single std-dev (fastest, good for stationary signals)
cfg.volatility_method = eemd::VolatilityMethod::Global;

// SMA: Rolling window std-dev (adapts to local variance)
cfg.volatility_method = eemd::VolatilityMethod::SMA;
cfg.vol_window = 100;  // Lookback window

// EMA: Exponential moving std-dev (fastest adaptation)
cfg.volatility_method = eemd::VolatilityMethod::EMA;
cfg.vol_ema_span = 20;  // EMA span
```

### Boundary Methods

Controls extrapolation at signal edges:

```cpp
// Symmetric: emd.m's symmetric extension (default for Standard/Scientific)
cfg.boundary_method = eemd::BoundaryMethod::Symmetric;

// Mirror: Reflects extrema plus a linear end point (the pre-1.1 default)
cfg.boundary_method = eemd::BoundaryMethod::Mirror;

// AR: Autoregressive extrapolation (causal, no look-ahead)
cfg.boundary_method = eemd::BoundaryMethod::AR;
cfg.ar_damping = 0.5;  // 0=mean-revert, 1=full AR

// Linear: Simple linear extrapolation
cfg.boundary_method = eemd::BoundaryMethod::Linear;
```

### Local Mean, Stop Rule and Noise Bank

```cpp
// The published algorithm (defaults)
cfg.local_mean_method = eemd::LocalMeanMethod::Sifted;   // M(x) = x - E1(x)
cfg.sift_stop_rule    = eemd::SiftStopRule::Rilling;     // emd.m criterion
cfg.rilling           = {0.05, 0.5, 0.05};               // theta1, theta2, alpha (emd.m defaults)
cfg.use_circular_bank = false;                           // independent noise realizations
cfg.noise_bank_max_bytes = size_t(2) << 30;              // above it: circular bank, else fail
```

### Speed Options

Each option trades fidelity to the published algorithm for speed (timings in
[Performance](#performance)):

```cpp
cfg.sift_stop_rule    = eemd::SiftStopRule::SD;                // stop at SD < sift_threshold (~2-3 iterations)
cfg.local_mean_method = eemd::LocalMeanMethod::SingleEnvelope; // one envelope mean per trial
cfg.use_circular_bank = true;                                  // one long noise decomposition
cfg.boundary_method   = eemd::BoundaryMethod::Mirror;          // 1.0 envelope ends
```

All four together reproduce the 1.0 configuration (except for the noise amplitude, which
follows the paper in every configuration).

## Algorithm Details

### ICEEMDAN vs EEMD vs EMD

| Method | Mode Mixing | Completeness | Computation |
|--------|-------------|--------------|-------------|
| EMD | Severe | Complete | Fast |
| EEMD | Reduced | Incomplete | Slow |
| CEEMDAN | Reduced | Complete | Slow |
| **ICEEMDAN** | **Minimal** | **Complete** | **Slow** |

ICEEMDAN adds noise to the *residue* at each stage (not the original signal), producing cleaner IMFs with better frequency separation.

### Accuracy

The default configuration was compared with the ICEEMDAN code of Colominas et al.
(`ceemdan.m` with Rilling & Flandrin's `emd.m`, run in GNU Octave, ε = 0.2, 100 realizations;
3 seeds for the reference, 5 for this library).

**White noise, N = 4096**: mean frequency (cycles/sample) of IMFs 1–6. ICEEMDAN of white noise
is a dyadic filter bank, so this is the core property to match:

| | IMF 1 | IMF 2 | IMF 3 | IMF 4 | IMF 5 | IMF 6 | mean distance to the reference |
|---|---|---|---|---|---|---|---|
| Reference (authors' code) | 0.3421 | 0.1568 | 0.0788 | 0.0422 | 0.0201 | 0.0095 | — |
| 1.1 default | 0.3421 | 0.1566 | 0.0787 | 0.0424 | 0.0202 | 0.0094 | 0.007 octave |
| 1.1, SD stop rule | 0.3256 | 0.1157 | 0.0476 | 0.0196 | 0.0074 | 0.0032 | 0.89 octave |
| 1.0 | 0.3175 | 0.1019 | 0.0395 | 0.0163 | 0.0066 | 0.0027 | 1.09 octave |

**Tones of 128, 32 and 8 cycles plus a trend, N = 2048**: relative error of the IMF that best
matches each tone (lower = less mode mixing; edges excluded):

| | 128 cycles | 32 cycles | 8 cycles |
|---|---|---|---|
| Reference (authors' code) | 0.269 | 0.302 | 0.058 |
| 1.1 default | 0.260 | 0.323 | 0.086 |
| 1.0 | 0.405 | 0.519 | 0.210 |

To reproduce: [`validation/README.md`](validation/README.md).

### Reproducibility

For a given `rng_seed` and configuration, `ICEEMDAN::decompose()` returns bit-identical IMFs at
any number of OpenMP threads (tested at 1–4 threads in every mode): each realization's noise
comes from its own Philox4x32-10 stream keyed by (seed, trial), and trials are summed in fixed
blocks in trial order. `EEMD::decompose()` is bit-identical between runs at a fixed thread
count; across thread counts it differs only by summation rounding.

### Diagnostics

```cpp
eemd::DecompositionDiagnostics diag;
decomposer.decompose_with_diagnostics(signal, n, imfs, residue, diag);
// per IMF k:
//   diag.sift_iterations[k]       most sifting iterations any trial needed
//   diag.mean_sift_iterations[k]  mean iterations per trial
//   diag.nonconverged_trials[k]   trials that hit max_sift_iters
//   diag.convergence_flags[k]     true if none did
//   diag.trials_per_imf[k]        local means averaged
// diag.nan_count, diag.rng_seed_used, diag.orthogonality_index, diag.noise_bank_multiplier
```

`decomposer.last_diagnostics()` returns the same after a plain `decompose()`.

### FFT Backend

The analysis utilities (`compute_hilbert_transform`, `compute_instantaneous_frequency`,
`compute_spectral_entropy`) use [VectorFFT](https://github.com/Tugbars/VectorFFT) real
transforms at every length from 2 up; a single sample needs no transform. Plans are made
once per thread and length, with `VFFT_FLAG_ESTIMATE` (no measuring, no wisdom files written),
and the caller's floating-point mode (MXCSR) is restored after every VectorFFT call.
`eemd::fft::backend_for(n)` returns `Backend::VectorFFT` when the transform of length `n` is
ready, and `Backend::None` when VectorFFT could not be loaded (its libraries are not beside the
program, or the CPU has no AVX2); the utilities then return `false`.

### S-Number Sifting Criterion

With `sift_stop_rule = SiftStopRule::SD`, the SD (stopping difference) criterion is combined with the **S-number criterion** for faster convergence (the default Rilling rule does not use it):

```cpp
cfg.s_number = 6;  // Stop after 6 consecutive iterations with stable extrema count
```

When the number of extrema remains unchanged for S iterations, the IMF is considered converged. This provides ~15% speedup with identical decomposition quality. Set to 0 to disable.

### Orthogonality Index

Lower is better. Measures mode mixing:

```cpp
eemd::DecompositionDiagnostics diag;
decomposer.decompose_with_diagnostics(signal, n, imfs, residue, diag);
std::cout << "IO: " << diag.orthogonality_index << "\n";  // Good: < 0.1
```

### IMF Interpretation

```
IMF 1: Highest frequency oscillations
IMF 2: Second highest frequency
...
IMF N: Lowest frequency oscillations  
Residue: Monotonic trend
```

For analysis:

```cpp
// Hurst exponent (persistence)
// H < 0.5: Mean-reverting
// H = 0.5: Random walk
// H > 0.5: Trending

// Compute instantaneous frequency via Hilbert transform
// (not included, use your preferred implementation)
```

## Examples

### Detrending a Signal

```cpp
eemd::ICEEMDAN decomposer;
decomposer.decompose(signal, n, imfs, residue);

// Detrended = signal - residue
std::vector<double> detrended(n);
for (int i = 0; i < n; ++i) {
    detrended[i] = signal[i] - residue[i];
}
```

### Extracting Specific Frequency Bands

```cpp
decomposer.decompose(signal, n, imfs, residue);

// High-frequency component (first few IMFs)
std::vector<double> high_freq(n, 0.0);
for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < n; ++i) {
        high_freq[i] += imfs[k][i];
    }
}

// Low-frequency component (remaining IMFs + residue)
std::vector<double> low_freq(n, 0.0);
for (int k = 3; k < imfs.size(); ++k) {
    for (int i = 0; i < n; ++i) {
        low_freq[i] += imfs[k][i];
    }
}
for (int i = 0; i < n; ++i) {
    low_freq[i] += residue[i];
}
```

### Reconstruction Verification

```cpp
// IMFs + residue should exactly reconstruct the signal
std::vector<double> reconstructed(n, 0.0);
for (const auto& imf : imfs) {
    for (int i = 0; i < n; ++i) {
        reconstructed[i] += imf[i];
    }
}
for (int i = 0; i < n; ++i) {
    reconstructed[i] += residue[i];
}

// Verify
double max_error = 0.0;
for (int i = 0; i < n; ++i) {
    max_error = std::max(max_error, std::abs(signal[i] - reconstructed[i]));
}
std::cout << "Reconstruction error: " << max_error << "\n";  // Should be ~1e-15
```

### Finance Mode with Akima Splines

```cpp
eemd::ICEEMDAN decomposer(eemd::ProcessingMode::Finance);

// Finance mode automatically sets:
// - spline_method = Akima (with Cubic fallback)
// - volatility_method = EMA
// - boundary_method = AR (causal)
// - sanitize_input = true

decomposer.decompose(prices, n, imfs, residue);

// IMF 0-1: Market microstructure noise
// IMF 2-3: Short-term mean-reversion
// IMF 4+: Swing/trend components
// Residue: Long-term drift
```

## Benchmarking

Build and run the benchmarks:

```bash
./build/iceemdan_bench         # General performance
./build/iceemdan_finance_bench # Finance-specific tests
./build/bench_snumber          # S-number optimization benchmark
```

## Visualization (Jupyter Notebook)

Generate CSV files and visualize with Python:

```bash
# 1. Build and run CSV export example (and see python/ for the ctypes binding:
#    it loads build/libiceemdan.so, or iceemdan.dll on Windows, built by the
#    iceemdan_shared target)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/example_finance_csv

# 2. Open Jupyter notebook
cd jupyter
jupyter notebook iceemdan_visualization.ipynb
```

The notebook shows:
- Full decomposition plots (signal + all IMFs + residue)
- Energy distribution across IMFs
- Reconstruction verification
- Frequency analysis per IMF
- Trading applications (detrending, denoising)

## References

1. **ICEEMDAN**: Colominas, M. A., Schlotthauer, G., & Torres, M. E. (2014). Improved complete ensemble EMD: A suitable tool for biomedical signal processing. *Biomedical Signal Processing and Control*, 14, 19-29.

2. **CEEMDAN**: Torres, M. E., Colominas, M. A., Schlotthauer, G., & Flandrin, P. (2011). A complete ensemble empirical mode decomposition with adaptive noise. *ICASSP 2011*.

3. **EEMD**: Wu, Z., & Huang, N. E. (2009). Ensemble empirical mode decomposition: a noise-assisted data analysis method. *Advances in Adaptive Data Analysis*, 1(01), 1-41.

4. **EMD**: Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. *Proceedings of the Royal Society A*, 454(1971), 903-995.

5. **Sifting stop rule and boundary extension**: Rilling, G., Flandrin, P., & Gonçalves, P. (2003). On empirical mode decomposition and its algorithms. *IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing (NSIP-03)*.

6. **Akima Splines**: Akima, H. (1970). A new method of interpolation and smooth curve fitting based on local procedures. *Journal of the ACM*, 17(4), 589-602.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{iceemdan_mkl,
  author = Tugbars Heptaskin,
  title = {ICEEMDAN-MKL: High-Performance ICEEMDAN Implementation},
  year = {2025},
  url = {https://github.com/Tugbars/iceemdan-mkl}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

- Intel MKL for high-performance spline fitting
- [VectorFFT](https://github.com/Tugbars/VectorFFT) for the FFTs
- Rilling, Flandrin & Gonçalves for `emd.m`, whose stop rule and boundary extension the defaults follow
- Original ICEEMDAN algorithm by Colominas et al.
- Akima spline method for robust envelope construction
