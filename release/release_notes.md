VectorEMD 0.9 beta

The library, ready built for Windows: a DLL, its headers and an example. There is nothing else to install.

**Start here**

1. Download `iceemdan-@VERSION@-windows-x64.zip` from the files below and unpack it.
2. In the unpacked folder, build the example:
   - Visual Studio (in a "Developer Command Prompt"): `cl /Iinclude examples\decompose.c lib\iceemdan.lib`
   - GCC (MinGW-w64): `gcc -Iinclude examples/decompose.c bin/iceemdan.dll -o decompose.exe`
3. Put the DLLs beside it and run it: `copy bin\*.dll .`, then `decompose.exe`.

`README.txt` in the package shows the few lines your own program needs.

**In the package**

- `bin/iceemdan.dll`, with the FFT libraries it uses (VectorFFT)
- `lib/iceemdan.lib` for Visual Studio
- `include/iceemdan.h` (ICEEMDAN and Hilbert-Huang analysis, eight functions), `include/iceemdan_diagnostics.h`, `include/emd.h` (EMD and EEMD)
- `examples/decompose.c`, `README.txt` and the licences

**What it computes**

- ICEEMDAN as published (Colominas et al., 2014). Given the same noise, it returns the modes of the authors' own code sample by sample; the README of the repository shows the comparison.
- EMD and EEMD.
- The amplitude, frequency and phase of an IMF over time, and the Hilbert spectrum.

**What it needs**

- Windows 10 or later, on a processor with AVX2 and FMA (Intel since 2013, AMD since 2015).
- Any language that can call a C library can use it: C, C++, Python (ctypes), MATLAB, R, C#, LabVIEW.

This is a beta. Please report problems in the issues of this repository.
