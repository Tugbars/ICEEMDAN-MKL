VectorEMD @VERSION@: ICEEMDAN for C and any language that calls C

Decomposes a signal into intrinsic mode functions (IMFs) with ICEEMDAN
(Colominas et al., 2014), as the authors' code computes it, and analyses them
(instantaneous frequency, Hilbert spectrum).

Needs Windows 10 or later on a processor with AVX2 and FMA (Intel since 2013,
AMD since 2015). Nothing else to install.

In this folder
  include/iceemdan.h     the header your program includes: eight functions, each described there
  include/iceemdan_diagnostics.h
                         extra facts for logs and reports; not needed to decompose a signal
  include/emd.h          plain EMD and EEMD, six functions of the same shape
  bin/iceemdan.dll       the library
  bin/libvfft.dll, bin/libvfft_avx2.dll, bin/libvfft_avx512.dll
                         the FFT library it uses (VectorFFT)
  lib/iceemdan.lib       what Visual Studio links
  examples/decompose.c   a signal decomposed in a few lines
  LICENSE.txt, LICENSE-VectorFFT.txt

1. Build the example, in this folder

   Visual Studio (in a "Developer Command Prompt"):
       cl /Iinclude examples\decompose.c lib\iceemdan.lib

   GCC (MinGW-w64):
       gcc -Iinclude examples/decompose.c bin/iceemdan.dll -o decompose.exe

   CMake: add include to the include directories and link lib/iceemdan.lib.

2. Run it. The four DLLs of bin must be beside the program, or bin on PATH:
       copy bin\*.dll .
       decompose.exe

3. In your program

       #include "iceemdan.h"

       iceemdan_t *d = iceemdan_create(NULL);        // NULL: the published algorithm
       int k = iceemdan_decompose(d, x, 1, n);       // 1 signal of n samples: k IMFs
       iceemdan_result(d, imfs, residue, NULL);      // k x n values, and n values
       iceemdan_destroy(d);

   The channels of a recording go through the same call: pass their number
   instead of 1. A call that cannot be served returns -1 (or NULL), and
   iceemdan_error() says why. The settings (ensemble size, noise, seed,
   threads) are one struct, iceemdan_config_t; zeros mean the defaults.

   iceemdan_instantaneous() gives the amplitude, frequency and phase of an
   IMF over time, iceemdan_spectrum() the Hilbert spectrum of the IMFs.

Other languages call the same DLL: Python (ctypes), MATLAB (loadlibrary),
R, C# (DllImport), LabVIEW (Call Library Function).

A decomposition comes out the same, bit for bit, on every processor that runs
the library and at any number of threads, for the same settings and seed.

Source, documentation and the Python package: https://github.com/Tugbars/iceemdan
