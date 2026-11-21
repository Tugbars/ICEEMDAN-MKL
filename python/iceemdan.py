"""
ICEEMDAN Python Wrapper (ctypes)

Usage:
    from iceemdan import ICEEMDAN, ProcessingMode
    
    # Create decomposer
    decomposer = ICEEMDAN(mode=ProcessingMode.FINANCE)
    
    # Configure (optional)
    decomposer.ensemble_size = 100
    decomposer.noise_std = 0.2
    
    # Decompose
    imfs, residue = decomposer.decompose(signal)
    
    # With diagnostics
    imfs, residue, diag = decomposer.decompose(signal, diagnostics=True)

    # Hilbert-Huang analysis of the IMFs (sample rate in Hz)
    amplitude, frequency, phase = instantaneous(imfs, sample_rate=250.0)
    hs = hilbert_spectrum(imfs, sample_rate=250.0, fmax=40.0)  # hs.spectrum, hs.frequencies, hs.times, hs.marginal
"""

import ctypes
import numpy as np
import sys
from pathlib import Path
from enum import IntEnum
from typing import Tuple, Optional, Union, Dict, NamedTuple
import platform

# ============================================================================
# Enums
# ============================================================================

class ProcessingMode(IntEnum):
    STANDARD = 0
    FINANCE = 1
    SCIENTIFIC = 2

class SplineMethod(IntEnum):
    CUBIC = 0
    AKIMA = 1
    LINEAR = 2

class VolatilityMethod(IntEnum):
    GLOBAL = 0
    SMA = 1
    EMA = 2

class BoundaryMethod(IntEnum):
    MIRROR = 0
    AR = 1
    LINEAR = 2

class LocalMeanMethod(IntEnum):
    SIFTED = 0           # Colominas et al. (2014): first mode sifted to convergence (default)
    SINGLE_ENVELOPE = 1  # one envelope mean per trial: faster, not the published algorithm

# ============================================================================
# Library Loading
# ============================================================================

_LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008  # Windows LoadLibraryEx flag


def _load_library() -> ctypes.CDLL:
    """Load the ICEEMDAN shared library."""
    
    # Determine library name based on platform
    system = platform.system()
    if system == "Windows":
        lib_names = ["iceemdan.dll", "libiceemdan.dll", "iceemdan_shared.dll"]
    elif system == "Darwin":
        lib_names = ["libiceemdan.dylib", "libiceemdan.so"]
    else:
        lib_names = ["libiceemdan.so"]
    
    # Search paths
    this_dir = Path(__file__).parent
    root_dir = this_dir.parent  # Parent of python/ folder
    
    search_paths = [
        this_dir,                                 # Same directory as this file
        this_dir / "lib",                         # ./lib/
        root_dir / "build" / "Release",           # ../build/Release/
        root_dir / "build" / "Debug",             # ../build/Debug/
        root_dir / "build",                       # ../build/
        root_dir,                                 # ../
        Path.cwd(),                               # Current working directory
        Path.cwd() / "build" / "Release",
        Path.cwd() / "build",
    ]
    
    # Linux and macOS: RTLD_GLOBAL, so that libraries the library loads in turn
    # see its symbols.
    load_mode = 0 if system == "Windows" else ctypes.RTLD_GLOBAL

    errors = []
    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                try:
                    return ctypes.CDLL(str(lib_path), mode=load_mode)
                except OSError as e:
                    errors.append(f"{lib_path}: {e}")
                if system == "Windows":
                    # Python 3.8+ leaves PATH out when it looks for the DLLs a
                    # library depends on (the compiler's runtime, for a GCC
                    # build). This search takes the library's folder first and
                    # PATH after it.
                    try:
                        return ctypes.CDLL(str(lib_path), winmode=_LOAD_WITH_ALTERED_SEARCH_PATH)
                    except (OSError, TypeError) as e:
                        errors.append(f"{lib_path} (PATH searched): {e}")
    
    # Try system paths
    for lib_name in lib_names:
        try:
            return ctypes.CDLL(lib_name, mode=load_mode)
        except OSError as e:
            errors.append(f"{lib_name} (system): {e}")
            continue
    
    error_msg = f"Could not find/load ICEEMDAN library.\n"
    error_msg += f"Searched for {lib_names} in {[str(p) for p in search_paths]}.\n"
    if errors:
        error_msg += "Load errors:\n" + "\n".join(f"  - {e}" for e in errors)
        error_msg += "\n\nOn Windows, libvfft.dll, libvfft_avx2.dll and libvfft_avx512.dll must be"
        error_msg += "\nbeside the ICEEMDAN DLL, and for a GCC build the compiler's bin folder"
        error_msg += "\nmust be in PATH (its C++ runtime)."
    raise RuntimeError(error_msg)

# Load library
_lib = _load_library()

# ============================================================================
# Function Signatures
# ============================================================================

# Opaque handle type
class _ICEEMDANHandle(ctypes.Structure):
    pass

_HandlePtr = ctypes.POINTER(_ICEEMDANHandle)

# Lifecycle
_lib.iceemdan_create.argtypes = [ctypes.c_int]
_lib.iceemdan_create.restype = _HandlePtr

_lib.iceemdan_destroy.argtypes = [_HandlePtr]
_lib.iceemdan_destroy.restype = None

# Configuration
_lib.iceemdan_set_ensemble_size.argtypes = [_HandlePtr, ctypes.c_int]
_lib.iceemdan_set_noise_std.argtypes = [_HandlePtr, ctypes.c_double]
_lib.iceemdan_set_max_imfs.argtypes = [_HandlePtr, ctypes.c_int]
_lib.iceemdan_set_max_sift_iters.argtypes = [_HandlePtr, ctypes.c_int]
_lib.iceemdan_set_sift_threshold.argtypes = [_HandlePtr, ctypes.c_double]
_lib.iceemdan_set_rng_seed.argtypes = [_HandlePtr, ctypes.c_uint]
_lib.iceemdan_set_s_number.argtypes = [_HandlePtr, ctypes.c_int]
_lib.iceemdan_set_spline_method.argtypes = [_HandlePtr, ctypes.c_int]
_lib.iceemdan_set_volatility_method.argtypes = [_HandlePtr, ctypes.c_int]
_lib.iceemdan_set_boundary_method.argtypes = [_HandlePtr, ctypes.c_int]
_lib.iceemdan_set_local_mean_method.argtypes = [_HandlePtr, ctypes.c_int]
_lib.iceemdan_set_keep_noise_bank.argtypes = [_HandlePtr, ctypes.c_int]
_lib.iceemdan_set_keep_noise_bank.restype = None
_lib.iceemdan_release_noise_bank.argtypes = [_HandlePtr]
_lib.iceemdan_release_noise_bank.restype = None

# Decomposition
_lib.iceemdan_decompose.argtypes = [_HandlePtr, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
_lib.iceemdan_decompose.restype = ctypes.c_int

_lib.iceemdan_decompose_with_diagnostics.argtypes = [_HandlePtr, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
_lib.iceemdan_decompose_with_diagnostics.restype = ctypes.c_int

# Results
_lib.iceemdan_get_num_imfs.argtypes = [_HandlePtr]
_lib.iceemdan_get_num_imfs.restype = ctypes.c_int

_lib.iceemdan_get_imf_length.argtypes = [_HandlePtr]
_lib.iceemdan_get_imf_length.restype = ctypes.c_int

_lib.iceemdan_get_imf.argtypes = [_HandlePtr, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
_lib.iceemdan_get_imf.restype = ctypes.c_int

_lib.iceemdan_get_residue.argtypes = [_HandlePtr, ctypes.POINTER(ctypes.c_double)]
_lib.iceemdan_get_residue.restype = ctypes.c_int

_lib.iceemdan_get_all_imfs.argtypes = [_HandlePtr, ctypes.POINTER(ctypes.c_double)]
_lib.iceemdan_get_all_imfs.restype = ctypes.c_int

# Diagnostics
_lib.iceemdan_get_orthogonality_index.argtypes = [_HandlePtr]
_lib.iceemdan_get_orthogonality_index.restype = ctypes.c_double

_lib.iceemdan_get_nan_count.argtypes = [_HandlePtr]
_lib.iceemdan_get_nan_count.restype = ctypes.c_int

_lib.iceemdan_get_rng_seed_used.argtypes = [_HandlePtr]
_lib.iceemdan_get_rng_seed_used.restype = ctypes.c_uint

_lib.iceemdan_get_valid.argtypes = [_HandlePtr]
_lib.iceemdan_get_valid.restype = ctypes.c_int

_lib.iceemdan_get_noise_bank_reused.argtypes = [_HandlePtr]
_lib.iceemdan_get_noise_bank_reused.restype = ctypes.c_int

# Batches
_lib.iceemdan_decompose_batch.argtypes = [_HandlePtr, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
_lib.iceemdan_decompose_batch.restype = ctypes.c_int
_lib.iceemdan_batch_num_imfs.argtypes = [_HandlePtr, ctypes.c_int]
_lib.iceemdan_batch_num_imfs.restype = ctypes.c_int
_lib.iceemdan_batch_orthogonality_index.argtypes = [_HandlePtr, ctypes.c_int]
_lib.iceemdan_batch_orthogonality_index.restype = ctypes.c_double
_lib.iceemdan_batch_get.argtypes = [_HandlePtr, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
_lib.iceemdan_batch_get.restype = ctypes.c_int

for _name, _restype in [('iceemdan_get_imf_sift_iterations', ctypes.c_int),
                        ('iceemdan_get_imf_mean_sift_iterations', ctypes.c_double),
                        ('iceemdan_get_imf_nonconverged_trials', ctypes.c_int),
                        ('iceemdan_get_imf_trials', ctypes.c_int),
                        ('iceemdan_get_imf_converged', ctypes.c_int)]:
    getattr(_lib, _name).argtypes = [_HandlePtr, ctypes.c_int]
    getattr(_lib, _name).restype = _restype

_lib.iceemdan_fft_backend.argtypes = [ctypes.c_int]
_lib.iceemdan_fft_backend.restype = ctypes.c_char_p

_lib.iceemdan_get_error.argtypes = [_HandlePtr]
_lib.iceemdan_get_error.restype = ctypes.c_char_p

# Utility
_lib.iceemdan_version.argtypes = []
_lib.iceemdan_version.restype = ctypes.c_char_p

_lib.iceemdan_set_num_threads.argtypes = [ctypes.c_int]
_lib.iceemdan_set_num_threads.restype = None
_lib.iceemdan_get_num_threads.argtypes = []
_lib.iceemdan_get_num_threads.restype = ctypes.c_int
_lib.iceemdan_shutdown_threads.argtypes = []
_lib.iceemdan_shutdown_threads.restype = None

_lib.iceemdan_compute_energy.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
_lib.iceemdan_compute_energy.restype = ctypes.c_double

# Hilbert-Huang analysis
_lib.iceemdan_instantaneous.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
_lib.iceemdan_instantaneous.restype = ctypes.c_int
_lib.iceemdan_mean_frequency.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double]
_lib.iceemdan_mean_frequency.restype = ctypes.c_double
_lib.iceemdan_spectrum_columns.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.iceemdan_spectrum_columns.restype = ctypes.c_int
_lib.iceemdan_hilbert_spectrum.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                           ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
_lib.iceemdan_hilbert_spectrum.restype = ctypes.c_int

# ============================================================================
# Python Wrapper Class
# ============================================================================

class BatchDecomposition(NamedTuple):
    """The decompositions of many signals of one length (see ICEEMDAN.decompose_batch())."""
    imfs: np.ndarray       # (count, max IMFs, n): a signal with fewer IMFs has rows of zeros after its own
    residues: np.ndarray   # (count, n)
    num_imfs: np.ndarray   # (count,): the IMFs of each signal


class ICEEMDAN:
    """
    High-performance ICEEMDAN decomposition.
    
    Parameters
    ----------
    mode : ProcessingMode, optional
        Processing mode (STANDARD, FINANCE, or SCIENTIFIC). Default: STANDARD
    
    Examples
    --------
    >>> from iceemdan import ICEEMDAN, ProcessingMode
    >>> import numpy as np
    >>> 
    >>> # Create decomposer
    >>> decomposer = ICEEMDAN(mode=ProcessingMode.FINANCE)
    >>> 
    >>> # Generate test signal
    >>> t = np.linspace(0, 1, 1024)
    >>> signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    >>> 
    >>> # Decompose
    >>> imfs, residue = decomposer.decompose(signal)
    >>> print(f"Extracted {len(imfs)} IMFs")
    """
    
    def __init__(self, mode: ProcessingMode = ProcessingMode.STANDARD):
        self._handle = _lib.iceemdan_create(int(mode))
        if not self._handle:
            raise RuntimeError("Failed to create ICEEMDAN instance")
        self._mode = mode
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.iceemdan_destroy(self._handle)
    
    # ========================================================================
    # Configuration Properties
    # ========================================================================
    
    @property
    def ensemble_size(self) -> int:
        """Number of noise realizations."""
        return self._ensemble_size if hasattr(self, '_ensemble_size') else 100
    
    @ensemble_size.setter
    def ensemble_size(self, value: int):
        _lib.iceemdan_set_ensemble_size(self._handle, int(value))
        self._ensemble_size = value
    
    @property
    def noise_std(self) -> float:
        """Noise standard deviation as fraction of signal std."""
        return self._noise_std if hasattr(self, '_noise_std') else 0.2
    
    @noise_std.setter
    def noise_std(self, value: float):
        _lib.iceemdan_set_noise_std(self._handle, float(value))
        self._noise_std = value
    
    @property
    def max_imfs(self) -> int:
        """Maximum number of IMFs to extract."""
        return self._max_imfs if hasattr(self, '_max_imfs') else 32
    
    @max_imfs.setter
    def max_imfs(self, value: int):
        _lib.iceemdan_set_max_imfs(self._handle, int(value))
        self._max_imfs = value
    
    @property
    def max_sift_iters(self) -> int:
        """Maximum sifting iterations per IMF."""
        return self._max_sift_iters if hasattr(self, '_max_sift_iters') else 100
    
    @max_sift_iters.setter
    def max_sift_iters(self, value: int):
        _lib.iceemdan_set_max_sift_iters(self._handle, int(value))
        self._max_sift_iters = value
    
    @property
    def sift_threshold(self) -> float:
        """Sifting convergence threshold (SD criterion)."""
        return self._sift_threshold if hasattr(self, '_sift_threshold') else 0.05
    
    @sift_threshold.setter
    def sift_threshold(self, value: float):
        _lib.iceemdan_set_sift_threshold(self._handle, float(value))
        self._sift_threshold = value
    
    @property
    def rng_seed(self) -> int:
        """Seed of the noise. The same seed gives the same decomposition."""
        return self._rng_seed if hasattr(self, '_rng_seed') else 42
    
    @rng_seed.setter
    def rng_seed(self, value: int):
        _lib.iceemdan_set_rng_seed(self._handle, int(value))
        self._rng_seed = value
    
    @property
    def s_number(self) -> int:
        """S-number stopping criterion (0 = disabled)."""
        return self._s_number if hasattr(self, '_s_number') else 6
    
    @s_number.setter
    def s_number(self, value: int):
        _lib.iceemdan_set_s_number(self._handle, int(value))
        self._s_number = value
    
    @property
    def spline_method(self) -> SplineMethod:
        """Spline interpolation method."""
        return self._spline_method if hasattr(self, '_spline_method') else SplineMethod.CUBIC
    
    @spline_method.setter
    def spline_method(self, value: SplineMethod):
        _lib.iceemdan_set_spline_method(self._handle, int(value))
        self._spline_method = value
    
    @property
    def volatility_method(self) -> VolatilityMethod:
        """Volatility scaling method."""
        return self._volatility_method if hasattr(self, '_volatility_method') else VolatilityMethod.GLOBAL
    
    @volatility_method.setter
    def volatility_method(self, value: VolatilityMethod):
        _lib.iceemdan_set_volatility_method(self._handle, int(value))
        self._volatility_method = value
    
    @property
    def boundary_method(self) -> BoundaryMethod:
        """Boundary extrapolation method."""
        return self._boundary_method if hasattr(self, '_boundary_method') else BoundaryMethod.MIRROR
    
    @boundary_method.setter
    def boundary_method(self, value: BoundaryMethod):
        _lib.iceemdan_set_boundary_method(self._handle, int(value))
        self._boundary_method = value

    @property
    def local_mean_method(self) -> LocalMeanMethod:
        """Local-mean operator: SIFTED (published algorithm, default) or SINGLE_ENVELOPE (faster)."""
        return self._local_mean_method if hasattr(self, '_local_mean_method') else LocalMeanMethod.SIFTED

    @local_mean_method.setter
    def local_mean_method(self, value: LocalMeanMethod):
        _lib.iceemdan_set_local_mean_method(self._handle, int(value))
        self._local_mean_method = value

    @property
    def keep_noise_bank(self) -> bool:
        """
        Keep the decomposed noise of the ensemble between calls (default: True).

        It depends on the signal length, the seed, the ensemble size and the
        sifting settings, not on the signal: with it kept, every further signal
        of the same length is decomposed faster and gets the same result as
        before. Set to False to free it at the end of each call.
        """
        return self._keep_noise_bank if hasattr(self, '_keep_noise_bank') else True

    @keep_noise_bank.setter
    def keep_noise_bank(self, value: bool):
        _lib.iceemdan_set_keep_noise_bank(self._handle, int(bool(value)))
        self._keep_noise_bank = bool(value)

    def release_noise_bank(self):
        """Free the noise kept from the last call. The next call builds it again."""
        _lib.iceemdan_release_noise_bank(self._handle)
    
    # ========================================================================
    # Decomposition
    # ========================================================================
    
    def decompose(
        self, 
        signal: np.ndarray,
        diagnostics: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Decompose signal into IMFs and residue.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D array of floats)
        diagnostics : bool, optional
            If True, return diagnostics dict. Default: False
        
        Returns
        -------
        imfs : np.ndarray
            IMFs as 2D array (num_imfs, signal_length). 
            imfs[0] = highest frequency, imfs[-1] = lowest frequency.
        residue : np.ndarray
            Monotonic residue (trend component)
        diag : dict, optional
            Diagnostics (if diagnostics=True):
            - orthogonality_index: Mode mixing metric (lower = better)
            - nan_count: Number of NaN/Inf values sanitized
            - rng_seed_used: Actual RNG seed used
            - valid: True if decomposition succeeded
            - noise_bank_reused: True if the noise kept from an earlier call was used
            - sift_iterations: per IMF, most sifting iterations any trial needed
            - mean_sift_iterations: per IMF, mean iterations per trial
            - nonconverged_trials: per IMF, trials that hit max_sift_iters
            - converged: per IMF, True if no trial hit max_sift_iters
            - trials_per_imf: per IMF, local means averaged
        
        Examples
        --------
        >>> imfs, residue = decomposer.decompose(signal)
        >>> imfs, residue, diag = decomposer.decompose(signal, diagnostics=True)
        """
        # Convert to contiguous double array
        signal = np.ascontiguousarray(signal, dtype=np.float64)
        n = len(signal)
        
        if n < 4:
            raise ValueError("Signal must have at least 4 samples")
        
        # Get pointer to data
        signal_ptr = signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Run decomposition
        if diagnostics:
            num_imfs = _lib.iceemdan_decompose_with_diagnostics(self._handle, signal_ptr, n)
        else:
            num_imfs = _lib.iceemdan_decompose(self._handle, signal_ptr, n)
        
        if num_imfs == 0:
            error = _lib.iceemdan_get_error(self._handle).decode('utf-8')
            raise RuntimeError(f"Decomposition failed: {error}")
        
        # Get IMFs
        imfs = np.zeros((num_imfs, n), dtype=np.float64)
        imfs_ptr = imfs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if not _lib.iceemdan_get_all_imfs(self._handle, imfs_ptr):
            raise RuntimeError("Failed to retrieve IMFs")
        
        # Get residue
        residue = np.zeros(n, dtype=np.float64)
        residue_ptr = residue.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if not _lib.iceemdan_get_residue(self._handle, residue_ptr):
            raise RuntimeError("Failed to retrieve residue")
        
        if diagnostics:
            diag = {
                'orthogonality_index': _lib.iceemdan_get_orthogonality_index(self._handle),
                'nan_count': _lib.iceemdan_get_nan_count(self._handle),
                'rng_seed_used': _lib.iceemdan_get_rng_seed_used(self._handle),
                'valid': bool(_lib.iceemdan_get_valid(self._handle)),
                'noise_bank_reused': bool(_lib.iceemdan_get_noise_bank_reused(self._handle)),
                'sift_iterations': [_lib.iceemdan_get_imf_sift_iterations(self._handle, k) for k in range(num_imfs)],
                'mean_sift_iterations': [_lib.iceemdan_get_imf_mean_sift_iterations(self._handle, k) for k in range(num_imfs)],
                'nonconverged_trials': [_lib.iceemdan_get_imf_nonconverged_trials(self._handle, k) for k in range(num_imfs)],
                'converged': [bool(_lib.iceemdan_get_imf_converged(self._handle, k)) for k in range(num_imfs)],
                'trials_per_imf': [_lib.iceemdan_get_imf_trials(self._handle, k) for k in range(num_imfs)],
            }
            return imfs, residue, diag
        
        return imfs, residue
    
    def decompose_batch(self, signals) -> BatchDecomposition:
        """
        Decompose many signals of one length in one call: the channels of a
        recording, or its epochs.

        Each result is the same, bit for bit, as decompose() gives for that
        signal alone. The signals share one noise bank, and their trials are
        interleaved on the library's threads: a thread that would wait for
        the end of one signal's stage takes trials of the next signal
        instead, which uses the threads better than decomposing the signals
        one after another.

        Parameters
        ----------
        signals : np.ndarray
            (count, n): one signal per row. A list of equal-length 1D arrays
            works as well.

        Returns
        -------
        BatchDecomposition
            imfs (count, max IMFs, n), residues (count, n), num_imfs (count,).
            A signal with fewer IMFs than the most has rows of zeros after its
            own, so imfs[s].sum(axis=0) + residues[s] is signal s.

        Examples
        --------
        >>> eeg.shape                  # 64 channels, 10 s at 256 Hz
        (64, 2560)
        >>> imfs, residues, num_imfs = decomposer.decompose_batch(eeg)
        >>> imfs.shape
        (64, 11, 2560)
        """
        rows = np.ascontiguousarray(signals, dtype=np.float64)
        if rows.ndim != 2:
            raise ValueError("signals must be 2D: (count, n), one signal per row")
        count, n = rows.shape
        if count < 1 or n < 4:
            raise ValueError("need at least one signal of at least 4 samples")
        most = _lib.iceemdan_decompose_batch(self._handle, rows.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                             count, n)
        if most == 0:
            error = _lib.iceemdan_get_error(self._handle).decode('utf-8')
            raise RuntimeError(f"Batch decomposition failed: {error}")
        imfs = np.zeros((count, most, n))
        residues = np.zeros((count, n))
        _lib.iceemdan_batch_get(self._handle, imfs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                residues.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        num_imfs = np.array([_lib.iceemdan_batch_num_imfs(self._handle, s) for s in range(count)])
        return BatchDecomposition(imfs, residues, num_imfs)

    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def reconstruct(self, imfs: np.ndarray, residue: np.ndarray) -> np.ndarray:
        """Reconstruct signal from IMFs and residue."""
        return imfs.sum(axis=0) + residue
    
    def detrend(self, signal: np.ndarray) -> np.ndarray:
        """Remove trend (residue) from signal."""
        imfs, residue = self.decompose(signal)
        return signal - residue
    
    def denoise(self, signal: np.ndarray, remove_imfs: int = 2) -> np.ndarray:
        """Remove high-frequency noise (first N IMFs)."""
        imfs, residue = self.decompose(signal)
        return imfs[remove_imfs:].sum(axis=0) + residue
    
    def extract_trend(self, signal: np.ndarray) -> np.ndarray:
        """Extract trend component (residue)."""
        imfs, residue = self.decompose(signal)
        return residue
    
    @staticmethod
    def compute_energy(signal: np.ndarray) -> float:
        """Compute signal energy (sum of squares)."""
        signal = np.ascontiguousarray(signal, dtype=np.float64)
        signal_ptr = signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return _lib.iceemdan_compute_energy(signal_ptr, len(signal))
    
    def imf_energies(self, imfs: np.ndarray, residue: np.ndarray) -> np.ndarray:
        """Compute energy of each IMF and residue."""
        energies = np.array([self.compute_energy(imf) for imf in imfs])
        energies = np.append(energies, self.compute_energy(residue))
        return energies
    
    def snr_analysis(self, signal: np.ndarray, noise_imfs: int = 2) -> Dict:
        """
        Analyze signal-to-noise ratio.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        noise_imfs : int
            Number of IMFs to consider as noise (default: 2)
        
        Returns
        -------
        dict with:
            - snr: Signal-to-noise ratio
            - noise_energy: Energy in noise IMFs
            - signal_energy: Energy in remaining IMFs + residue
            - noise_fraction: noise_energy / total_energy
        """
        imfs, residue = self.decompose(signal)
        
        noise_energy = sum(self.compute_energy(imfs[i]) for i in range(min(noise_imfs, len(imfs))))
        signal_energy = sum(self.compute_energy(imfs[i]) for i in range(noise_imfs, len(imfs)))
        signal_energy += self.compute_energy(residue)
        
        total_energy = noise_energy + signal_energy
        
        return {
            'snr': signal_energy / noise_energy if noise_energy > 0 else float('inf'),
            'noise_energy': noise_energy,
            'signal_energy': signal_energy,
            'noise_fraction': noise_energy / total_energy if total_energy > 0 else 0,
        }


# ============================================================================
# Module-level functions
# ============================================================================

def version() -> str:
    """Get ICEEMDAN library version."""
    return _lib.iceemdan_version().decode('utf-8')


def set_num_threads(n: int = 0) -> None:
    """Number of threads a decomposition uses, for the whole process.

    n < 1: every logical processor, which is also the default unless the
    environment variable EEMD_NUM_THREADS is set before the library loads.
    The results do not depend on the thread count.
    """
    _lib.iceemdan_set_num_threads(int(n))


def get_num_threads() -> int:
    """Number of threads a decomposition uses."""
    return _lib.iceemdan_get_num_threads()


def shutdown_threads() -> None:
    """Stop the library's worker threads; the next decomposition starts them again."""
    _lib.iceemdan_shutdown_threads()


def decompose(
    signal: np.ndarray,
    mode: ProcessingMode = ProcessingMode.STANDARD,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for one-shot decomposition.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    mode : ProcessingMode
        Processing mode
    **kwargs : dict
        Configuration options (ensemble_size, noise_std, etc.)
    
    Returns
    -------
    imfs, residue : Tuple[np.ndarray, np.ndarray]
    """
    decomposer = ICEEMDAN(mode=mode)
    
    for key, value in kwargs.items():
        if hasattr(decomposer, key):
            setattr(decomposer, key, value)
    
    return decomposer.decompose(signal)


def decompose_batch(
    signals: np.ndarray,
    mode: ProcessingMode = ProcessingMode.STANDARD,
    **kwargs
) -> BatchDecomposition:
    """
    Convenience function: many signals of one length in one call (see
    ICEEMDAN.decompose_batch()). Keyword arguments set configuration options
    (ensemble_size, noise_std, ...).
    """
    decomposer = ICEEMDAN(mode=mode)
    for key, value in kwargs.items():
        if hasattr(decomposer, key):
            setattr(decomposer, key, value)
    return decomposer.decompose_batch(signals)


# ============================================================================
# Quick test
# ============================================================================


def fft_backend(n: int = 1024) -> str:
    """Name of the FFT backend the library uses for length n: "VectorFFT", or "none"
    for n < 2 (no transform needed) and when VectorFFT could not be loaded."""
    return _lib.iceemdan_fft_backend(int(n)).decode('utf-8')


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


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def instantaneous(imfs: np.ndarray, sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Instantaneous amplitude, frequency and phase of each IMF, from its analytic signal.

    Parameters
    ----------
    imfs : np.ndarray
        One IMF (n,) or several (count, n), as decompose() returns them.
    sample_rate : float
        Samples per second. Frequencies are in Hz (cycles per sample for 1).

    Returns
    -------
    amplitude, frequency, phase : np.ndarray
        Each shaped like imfs: the envelope, the frequency in Hz and the
        phase in radians (unwrapped). Within a period or two of each end of a
        non-periodic IMF the values are distorted.
    """
    rows = _imf_rows(imfs)
    count, n = rows.shape
    amp, freq, phase = (np.empty_like(rows) for _ in range(3))
    for k in range(count):
        if not _lib.iceemdan_instantaneous(_ptr(rows[k]), n, float(sample_rate),
                                           _ptr(amp[k]), _ptr(freq[k]), _ptr(phase[k])):
            raise ValueError("instantaneous(): needs n >= 2 and sample_rate > 0")
    if np.ndim(imfs) == 1:
        return amp[0], freq[0], phase[0]
    return amp, freq, phase


def mean_frequency(imfs: np.ndarray, sample_rate: float = 1.0) -> Union[float, np.ndarray]:
    """Instantaneous frequency of each IMF averaged with its energy as the weight, in Hz:
    one value for one IMF, an array for several."""
    rows = _imf_rows(imfs)
    values = np.array([_lib.iceemdan_mean_frequency(_ptr(r), rows.shape[1], float(sample_rate)) for r in rows])
    return float(values[0]) if np.ndim(imfs) == 1 else values


class HilbertSpectrum(NamedTuple):
    """The Hilbert spectrum of a set of IMFs (see hilbert_spectrum())."""
    spectrum: np.ndarray     # (frequency_bins, columns): energy or amplitude per frequency bin and time column
    frequencies: np.ndarray  # (frequency_bins,): the centre of each frequency bin, Hz
    times: np.ndarray        # (columns,): the centre of each time column, s
    marginal: np.ndarray     # (frequency_bins,): the spectrum summed over time


def hilbert_spectrum(imfs: np.ndarray, sample_rate: float = 1.0, frequency_bins: int = 128,
                     fmin: float = 0.0, fmax: Optional[float] = None, time_step: int = 1,
                     amplitude: bool = False) -> HilbertSpectrum:
    """
    The Hilbert spectrum of the IMFs: their energy over time and frequency.

    At every sample, each IMF adds its instantaneous energy (amplitude
    squared; the amplitude with amplitude=True) to the frequency bin of its
    instantaneous frequency. Summed over time, that gives the marginal
    spectrum: how much energy the signal holds at each frequency.

    Parameters
    ----------
    imfs : np.ndarray
        One IMF (n,) or several (count, n), as decompose() returns them. The
        residue is usually left out.
    sample_rate : float
        Samples per second; frequencies and times are in Hz and seconds.
    frequency_bins : int
        Bins of equal width from fmin to fmax.
    fmin, fmax : float
        The frequency range in Hz; fmax None: sample_rate / 2. Frequencies
        outside it are left out.
    time_step : int
        Samples summed in each time column (1: a column per sample). For
        long recordings, a step keeps the map small.
    amplitude : bool
        Amplitude instead of energy.

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
    fmax_value = 0.0 if fmax is None else float(fmax)
    step = max(1, int(time_step))
    cols = _lib.iceemdan_spectrum_columns(n, step)
    spectrum = np.zeros((int(frequency_bins), cols))
    marginal = np.zeros(int(frequency_bins))
    if not _lib.iceemdan_hilbert_spectrum(_ptr(rows), count, n, float(sample_rate), int(frequency_bins),
                                          float(fmin), fmax_value, step, int(bool(amplitude)),
                                          _ptr(spectrum), _ptr(marginal)):
        raise ValueError("hilbert_spectrum(): needs n >= 2, sample_rate > 0, frequency_bins >= 1 "
                         "and 0 <= fmin < fmax")
    top = fmax_value if fmax_value > 0 else 0.5 * float(sample_rate)
    width = (top - float(fmin)) / int(frequency_bins)
    frequencies = float(fmin) + (np.arange(int(frequency_bins)) + 0.5) * width
    starts = np.arange(cols) * step
    ends = np.minimum(starts + step, n) - 1
    times = 0.5 * (starts + ends) / float(sample_rate)
    return HilbertSpectrum(spectrum, frequencies, times, marginal)


if __name__ == "__main__":
    print(f"ICEEMDAN version: {version()}")
    
    # Generate test signal
    t = np.linspace(0, 1, 1024)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t))
    
    # Test decomposition
    decomposer = ICEEMDAN(mode=ProcessingMode.STANDARD)
    decomposer.ensemble_size = 50  # Faster for testing
    
    imfs, residue, diag = decomposer.decompose(signal, diagnostics=True)
    
    print(f"Signal length: {len(signal)}")
    print(f"IMFs extracted: {len(imfs)}")
    print(f"Orthogonality index: {diag['orthogonality_index']:.6f}")
    print(f"Valid: {diag['valid']}")
    
    # Verify reconstruction
    reconstructed = decomposer.reconstruct(imfs, residue)
    max_error = np.max(np.abs(signal - reconstructed))
    print(f"Max reconstruction error: {max_error:.2e}")
    
    # SNR analysis
    snr = decomposer.snr_analysis(signal)
    print(f"SNR: {snr['snr']:.2f}")
    print(f"Noise fraction: {snr['noise_fraction']:.2%}")

