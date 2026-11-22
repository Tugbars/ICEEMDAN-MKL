"""
VectorEMD for Python: ICEEMDAN decomposition and Hilbert-Huang analysis.

    import iceemdan

    imfs, residue = iceemdan.decompose(signal)             # one signal: imfs (k, n), residue (n,)
    batch = iceemdan.decompose_batch(eeg)                   # (channels, n): batch.imfs, .residues, .num_imfs

    amplitude, frequency, phase = iceemdan.instantaneous(imfs, sample_rate=250.0)
    hs = iceemdan.hilbert_spectrum(imfs, sample_rate=250.0, fmax=40.0)   # hs.spectrum, .frequencies, .times, .marginal

    d = iceemdan.ICEEMDAN(ensemble_size=200)                # a decomposer to use again: it keeps its noise,
    imfs, residue = d.decompose(signal)                     # so further signals of a length take half the time

With no arguments the decomposition is ICEEMDAN as published (Colominas et
al., 2014), computed as the authors' code computes it. The module calls the
shared library's C interface (include/iceemdan.h).
"""

import ctypes
import platform
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

__all__ = ["ICEEMDAN", "BatchDecomposition", "HilbertSpectrum", "decompose", "decompose_batch", "instantaneous",
           "mean_frequency", "hilbert_spectrum", "version", "info"]


# ============================================================================
# The shared library
# ============================================================================

def _load_library() -> ctypes.CDLL:
    """Loads iceemdan.dll / libiceemdan.so from beside this file or from the build folder."""
    system = platform.system()
    if system == "Windows":
        names = ["iceemdan.dll", "libiceemdan.dll"]
    elif system == "Darwin":
        names = ["libiceemdan.dylib", "libiceemdan.so"]
    else:
        names = ["libiceemdan.so"]

    here = Path(__file__).parent
    root = here.parent
    folders = [here, here / "lib", root / "build" / "Release", root / "build", root, Path.cwd(),
               Path.cwd() / "build" / "Release", Path.cwd() / "build"]
    # Linux and macOS: RTLD_GLOBAL, so that libraries the library loads in turn see its symbols
    mode = 0 if system == "Windows" else ctypes.RTLD_GLOBAL

    errors = []
    for folder in folders:
        for name in names:
            path = folder / name
            if not path.exists():
                continue
            try:
                return ctypes.CDLL(str(path), mode=mode)
            except OSError as e:
                errors.append(f"{path}: {e}")
            if system == "Windows":
                # the library's folder first, then PATH, for the DLLs it needs in turn
                try:
                    return ctypes.CDLL(str(path), winmode=0x00000008)  # LOAD_WITH_ALTERED_SEARCH_PATH
                except (OSError, TypeError) as e:
                    errors.append(f"{path} (PATH searched): {e}")
    for name in names:
        try:
            return ctypes.CDLL(name, mode=mode)
        except OSError as e:
            errors.append(f"{name} (system): {e}")

    message = f"The iceemdan library was not found or did not load. Looked for {names} in {[str(f) for f in folders]}."
    if errors:
        message += "\n" + "\n".join(f"  - {e}" for e in errors)
        message += "\nOn Windows, libvfft.dll, libvfft_avx2.dll and libvfft_avx512.dll must be beside iceemdan.dll."
    raise ImportError(message)


_lib = _load_library()
_double_p = ctypes.POINTER(ctypes.c_double)


class _Config(ctypes.Structure):  # iceemdan_config_t
    _fields_ = [("ensemble_size", ctypes.c_int), ("noise", ctypes.c_double), ("max_imfs", ctypes.c_int),
                ("seed", ctypes.c_uint), ("nthreads", ctypes.c_int), ("flags", ctypes.c_uint)]


class _Info(ctypes.Structure):  # iceemdan_info_t
    _fields_ = [("version", ctypes.c_char_p), ("cpu_supported", ctypes.c_int), ("processors", ctypes.c_int),
                ("nthreads", ctypes.c_int), ("signals", ctypes.c_int), ("n", ctypes.c_int), ("imfs", ctypes.c_int),
                ("noise_kept", ctypes.c_int), ("repaired", ctypes.c_int)]


class _Spectrum(ctypes.Structure):  # iceemdan_spectrum_t
    _fields_ = [("frequency_bins", ctypes.c_int), ("fmin", ctypes.c_double), ("fmax", ctypes.c_double),
                ("time_step", ctypes.c_int), ("amplitude", ctypes.c_int)]


class _SignalStats(ctypes.Structure):  # iceemdan_signal_stats_t
    _fields_ = [("imfs", ctypes.c_int), ("orthogonality", ctypes.c_double), ("repaired", ctypes.c_int)]


class _ImfStats(ctypes.Structure):  # iceemdan_imf_stats_t
    _fields_ = [("copies", ctypes.c_int), ("iterations_max", ctypes.c_int), ("iterations_mean", ctypes.c_double),
                ("unconverged", ctypes.c_int)]


_FLAG_FAST = 1 << 0

_lib.iceemdan_create.argtypes = [ctypes.POINTER(_Config)]
_lib.iceemdan_create.restype = ctypes.c_void_p
_lib.iceemdan_decompose.argtypes = [ctypes.c_void_p, _double_p, ctypes.c_int, ctypes.c_int]
_lib.iceemdan_decompose.restype = ctypes.c_int
_lib.iceemdan_result.argtypes = [ctypes.c_void_p, _double_p, _double_p, ctypes.POINTER(ctypes.c_int)]
_lib.iceemdan_result.restype = ctypes.c_int
_lib.iceemdan_destroy.argtypes = [ctypes.c_void_p]
_lib.iceemdan_destroy.restype = None
_lib.iceemdan_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Info)]
_lib.iceemdan_info.restype = ctypes.c_int
_lib.iceemdan_error.argtypes = []
_lib.iceemdan_error.restype = ctypes.c_char_p
_lib.iceemdan_instantaneous.argtypes = [_double_p, ctypes.c_int, ctypes.c_double, _double_p, _double_p, _double_p]
_lib.iceemdan_instantaneous.restype = ctypes.c_int
_lib.iceemdan_spectrum.argtypes = [_double_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.POINTER(_Spectrum),
                                   _double_p, _double_p]
_lib.iceemdan_spectrum.restype = ctypes.c_int
_lib.iceemdan_signal_stats.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_SignalStats)]
_lib.iceemdan_signal_stats.restype = ctypes.c_int
_lib.iceemdan_imf_stats.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(_ImfStats)]
_lib.iceemdan_imf_stats.restype = ctypes.c_int


def _error() -> str:
    return _lib.iceemdan_error().decode("utf-8", "replace")


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_double_p)


def _library_info() -> _Info:
    facts = _Info()
    _lib.iceemdan_info(None, ctypes.byref(facts))
    return facts


if not _library_info().cpu_supported:
    raise ImportError("iceemdan needs a processor with AVX2 and FMA (Intel since 2013, AMD since 2015); "
                      "this one does not have them.")


# ============================================================================
# Decomposition
# ============================================================================

class BatchDecomposition(NamedTuple):
    """The decompositions of many signals of one length (decompose_batch())."""
    imfs: np.ndarray      # (count, rows, n): a signal with fewer IMFs than rows has zeros after its own
    residues: np.ndarray  # (count, n)
    num_imfs: np.ndarray  # (count,): the IMFs of each signal


class ICEEMDAN:
    """
    A decomposer. With no arguments: ICEEMDAN as published, on every processor
    of this machine.

    Parameters
    ----------
    ensemble_size : int
        The noisy copies averaged for each mode (default 100).
    noise : float
        The noise amplitude, the paper's epsilon, relative to the standard
        deviation of the signal (default 0.2).
    max_imfs : int
        The most IMFs to extract (default: as many as the signal has).
    seed : int
        The seed of the noise (default 42). The same seed gives the same result.
    threads : int
        The threads a decomposition runs on (default: every logical processor).
        The result does not depend on it.
    fast : bool
        8 to 12 times faster, no longer the published algorithm: a simpler
        rule for when sifting stops, one envelope mean for each noisy copy,
        and one shared noise decomposition.

    A decomposer keeps the noise it decomposed for one signal length, so
    further signals of that length take half the time or less, with the same
    result as they would get alone. NaN and Inf samples are filled in from
    their neighbours.

    Examples
    --------
    >>> d = iceemdan.ICEEMDAN()
    >>> imfs, residue = d.decompose(ecg)            # imfs[0] is the fastest oscillation
    >>> np.allclose(imfs.sum(axis=0) + residue, ecg)
    True
    >>> batch = d.decompose_batch(eeg)              # eeg: (channels, n)
    """

    def __init__(self, ensemble_size: int = 0, noise: float = 0.0, max_imfs: int = 0, seed: int = 0,
                 threads: int = 0, fast: bool = False):
        self._handle = None
        config = _Config(int(ensemble_size), float(noise), int(max_imfs), int(seed), int(threads),
                         _FLAG_FAST if fast else 0)
        handle = _lib.iceemdan_create(ctypes.byref(config))
        if not handle:
            raise ValueError(_error())
        self._handle = handle

    def close(self) -> None:
        """Frees the decomposer and the noise it keeps (also done when it is garbage collected)."""
        if getattr(self, "_handle", None):
            _lib.iceemdan_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _run(self, rows: np.ndarray) -> BatchDecomposition:
        if not self._handle:
            raise ValueError("the decomposer is closed")
        count, n = rows.shape
        most = _lib.iceemdan_decompose(self._handle, _ptr(rows), count, n)
        if most < 0:
            raise ValueError(_error())
        imfs = np.zeros((count, most, n))
        residues = np.zeros((count, n))
        num_imfs = np.zeros(count, dtype=np.intc)
        if _lib.iceemdan_result(self._handle, _ptr(imfs), _ptr(residues),
                                num_imfs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))) != 0:
            raise RuntimeError(_error())
        return BatchDecomposition(imfs, residues, num_imfs.astype(int))

    def decompose(self, signal) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose one signal.

        Parameters
        ----------
        signal : array of n >= 4 samples

        Returns
        -------
        imfs : np.ndarray, (k, n)
            The IMFs, from the fastest oscillation to the slowest.
        residue : np.ndarray, (n,)
            What is left: the trend. imfs.sum(axis=0) + residue is the signal.
        """
        row = np.ascontiguousarray(signal, dtype=np.float64)
        if row.ndim != 1:
            raise ValueError("signal must be 1D; for several signals, decompose_batch()")
        if row.size < 4:
            raise ValueError("signal must have at least 4 samples")
        batch = self._run(row[None, :])
        return batch.imfs[0], batch.residues[0]

    def decompose_batch(self, signals) -> BatchDecomposition:
        """
        Decompose many signals of one length in one call: the channels of a
        recording, or its epochs.

        Each signal gets the result decompose() gives it alone, bit for bit;
        the batch is faster, because the signals share one decomposed noise
        and are spread over the threads together.

        Parameters
        ----------
        signals : array (count, n), one signal per row (a list of equal-length arrays works)

        Returns
        -------
        BatchDecomposition
            imfs (count, rows, n), residues (count, n), num_imfs (count,).
            rows is the most IMFs of any signal; a signal with fewer has zeros
            after its own, so imfs[s].sum(axis=0) + residues[s] is signal s.

        Examples
        --------
        >>> eeg.shape                              # 64 channels, 10 s at 256 Hz
        (64, 2560)
        >>> imfs, residues, num_imfs = d.decompose_batch(eeg)
        >>> imfs.shape
        (64, 11, 2560)
        """
        rows = np.ascontiguousarray(signals, dtype=np.float64)
        if rows.ndim != 2:
            raise ValueError("signals must be 2D: (count, n), one signal per row")
        if rows.shape[0] < 1 or rows.shape[1] < 4:
            raise ValueError("need at least one signal of at least 4 samples")
        return self._run(rows)

    def info(self) -> Dict:
        """
        The decomposer's facts: threads, and of its last decomposition
        signals, n, imfs (the most of any signal), noise_kept (it used the
        noise of an earlier call) and repaired (NaN and Inf samples filled
        in).
        """
        if not self._handle:
            raise ValueError("the decomposer is closed")
        f = _Info()
        _lib.iceemdan_info(self._handle, ctypes.byref(f))
        return {"threads": f.nthreads, "signals": f.signals, "n": f.n, "imfs": f.imfs,
                "noise_kept": bool(f.noise_kept), "repaired": f.repaired}

    def diagnostics(self, signal: int = 0) -> Dict:
        """
        The audit trail of one signal of the last decomposition:
        orthogonality (the orthogonality index of its IMFs, lower is better),
        repaired (NaN and Inf samples filled in) and, per IMF, copies (the
        noisy copies averaged), iterations_max and iterations_mean (sifting
        iterations of the copies) and unconverged (copies that reached the
        iteration limit; 0 is the normal case).
        """
        if not self._handle:
            raise ValueError("the decomposer is closed")
        s = _SignalStats()
        if _lib.iceemdan_signal_stats(self._handle, int(signal), ctypes.byref(s)) != 0:
            raise ValueError(_error())
        per_imf = []
        for j in range(s.imfs):
            m = _ImfStats()
            _lib.iceemdan_imf_stats(self._handle, int(signal), j, ctypes.byref(m))
            per_imf.append(m)
        return {"orthogonality": s.orthogonality, "repaired": s.repaired,
                "copies": [m.copies for m in per_imf],
                "iterations_max": [m.iterations_max for m in per_imf],
                "iterations_mean": [m.iterations_mean for m in per_imf],
                "unconverged": [m.unconverged for m in per_imf]}


def decompose(signal, **config) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose one signal: imfs (k, n) and residue (n,). Keyword arguments as ICEEMDAN()."""
    with ICEEMDAN(**config) as d:
        return d.decompose(signal)


def decompose_batch(signals, **config) -> BatchDecomposition:
    """Decompose many signals of one length, (count, n). Keyword arguments as ICEEMDAN()."""
    with ICEEMDAN(**config) as d:
        return d.decompose_batch(signals)


def version() -> str:
    """The version of the loaded library."""
    return _library_info().version.decode("utf-8")


def info() -> Dict:
    """The library's facts: version, and processors (this machine's logical processors, the default thread count)."""
    f = _library_info()
    return {"version": f.version.decode("utf-8"), "processors": f.processors}


# ============================================================================
# Hilbert-Huang analysis of the IMFs
# ============================================================================

def _imf_rows(imfs) -> np.ndarray:
    """IMFs as a C-contiguous float64 array of shape (count, n)."""
    rows = np.ascontiguousarray(imfs, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError("imfs must be one IMF (n,) or several (count, n), with n >= 2")
    return rows


def instantaneous(imfs, sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The amplitude, frequency and phase of each IMF at every sample, from its
    analytic signal.

    Parameters
    ----------
    imfs : one IMF (n,) or several (count, n), as decompose() returns them
    sample_rate : samples per second; frequencies are in Hz (cycles per sample for 1)

    Returns
    -------
    amplitude, frequency, phase : np.ndarray
        Each shaped like imfs: the envelope, the frequency in Hz and the phase
        in radians (unwrapped). Within a period or two of each end of an IMF
        that does not fit the window a whole number of times, the values are
        distorted.
    """
    rows = _imf_rows(imfs)
    count, n = rows.shape
    amp, freq, phase = (np.empty_like(rows) for _ in range(3))
    for k in range(count):
        if _lib.iceemdan_instantaneous(_ptr(rows[k]), n, float(sample_rate), _ptr(amp[k]), _ptr(freq[k]),
                                       _ptr(phase[k])) != 0:
            raise ValueError(_error())
    if np.ndim(imfs) == 1:
        return amp[0], freq[0], phase[0]
    return amp, freq, phase


def mean_frequency(imfs, sample_rate: float = 1.0) -> Union[float, np.ndarray]:
    """The instantaneous frequency of each IMF averaged with its energy as the weight, in Hz:
    one value for one IMF, an array for several."""
    amp, freq, _ = instantaneous(_imf_rows(imfs), sample_rate)
    energy = amp * amp
    total = energy.sum(axis=1)
    values = np.divide((energy * freq).sum(axis=1), total, out=np.zeros_like(total), where=total > 0)
    return float(values[0]) if np.ndim(imfs) == 1 else values


class HilbertSpectrum(NamedTuple):
    """The Hilbert spectrum of a set of IMFs (hilbert_spectrum())."""
    spectrum: np.ndarray     # (frequency_bins, columns): energy or amplitude per frequency band and time column
    frequencies: np.ndarray  # (frequency_bins,): the centre of each band, Hz
    times: np.ndarray        # (columns,): the centre of each time column, s
    marginal: np.ndarray     # (frequency_bins,): the spectrum summed over time


def hilbert_spectrum(imfs, sample_rate: float = 1.0, frequency_bins: int = 128, fmin: float = 0.0,
                     fmax: Optional[float] = None, time_step: int = 1, amplitude: bool = False) -> HilbertSpectrum:
    """
    The Hilbert spectrum of the IMFs: their energy over time and frequency.

    At every sample each IMF adds its energy (amplitude squared; its amplitude
    with amplitude=True) to the frequency band of its instantaneous frequency.
    Summed over time, that is the marginal spectrum: how much energy the
    signal holds at each frequency.

    Parameters
    ----------
    imfs : one IMF (n,) or several (count, n); the residue is usually left out
    sample_rate : samples per second; frequencies and times are in Hz and seconds
    frequency_bins : bands of equal width from fmin to fmax
    fmin, fmax : the frequency range in Hz; fmax None: sample_rate / 2
    time_step : samples summed in each time column (1: a column per sample)
    amplitude : amplitude instead of energy

    Returns
    -------
    HilbertSpectrum
        spectrum (frequency_bins, columns), frequencies, times, marginal.

    Examples
    --------
    >>> imfs, residue = iceemdan.decompose(ecg)
    >>> hs = iceemdan.hilbert_spectrum(imfs, sample_rate=250.0, fmax=40.0)
    >>> plt.pcolormesh(hs.times, hs.frequencies, hs.spectrum)
    """
    rows = _imf_rows(imfs)
    count, n = rows.shape
    bins = int(frequency_bins)
    step = int(time_step)
    top = 0.5 * float(sample_rate) if fmax is None else float(fmax)
    if bins < 1 or step < 1:
        raise ValueError("frequency_bins and time_step must be 1 or more")
    if not float(fmin) < top:
        raise ValueError(f"fmin {fmin} is not below fmax {top}")
    cols = (n + step - 1) // step
    options = _Spectrum(bins, float(fmin), 0.0 if fmax is None else top, step, int(bool(amplitude)))
    spectrum = np.zeros((bins, cols))
    marginal = np.zeros(bins)
    if _lib.iceemdan_spectrum(_ptr(rows), count, n, float(sample_rate), ctypes.byref(options), _ptr(spectrum),
                              _ptr(marginal)) != 0:
        raise ValueError(_error())
    width = (top - float(fmin)) / bins
    frequencies = float(fmin) + (np.arange(bins) + 0.5) * width
    starts = np.arange(cols) * step
    ends = np.minimum(starts + step, n) - 1
    times = 0.5 * (starts + ends) / float(sample_rate)
    return HilbertSpectrum(spectrum, frequencies, times, marginal)


if __name__ == "__main__":
    t = np.linspace(0, 4, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 6 * t) + np.where((t > 1.5) & (t < 2.5), 0.5 * np.sin(2 * np.pi * 40 * t), 0.0)
    imfs, residue = decompose(x)
    print(f"VectorEMD {version()}: {len(imfs)} IMFs, mean frequencies {np.round(mean_frequency(imfs, 250.0), 1)} Hz")
    print(f"largest difference of IMFs + residue from the signal: {np.abs(imfs.sum(axis=0) + residue - x).max():.1e}")
