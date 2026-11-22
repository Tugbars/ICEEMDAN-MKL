cmake_minimum_required(VERSION 3.20)

# The version is ICEEMDAN_VERSION_STRING of include/iceemdan.h; the C++ header
# include/iceemdan.hpp carries the same string for programs that use it alone
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/iceemdan.h" _iceemdan_version_line
     REGEX "^#define ICEEMDAN_VERSION_STRING ")
string(REGEX MATCH "[0-9]+[.][0-9]+[.][0-9]+" ICEEMDAN_VERSION "${_iceemdan_version_line}")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/iceemdan.hpp" _iceemdan_hpp_version_line
     REGEX "^#define ICEEMDAN_VERSION_STRING ")
string(REGEX MATCH "[0-9]+[.][0-9]+[.][0-9]+" _iceemdan_hpp_version "${_iceemdan_hpp_version_line}")
if(NOT ICEEMDAN_VERSION STREQUAL _iceemdan_hpp_version)
    message(FATAL_ERROR "include/iceemdan.h says version ${ICEEMDAN_VERSION}, "
                        "include/iceemdan.hpp says ${_iceemdan_hpp_version}: they must be the same")
endif()

project(iceemdan 
    VERSION ${ICEEMDAN_VERSION}
    LANGUAGES C CXX
    DESCRIPTION "EEMD and ICEEMDAN (Improved Complete Ensemble EMD with Adaptive Noise)"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Threads: the C++ standard library's (std::thread). No OpenMP runtime.
# ============================================================================
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ============================================================================
# Compiler Detection & Flags
# ============================================================================

# For this machine (the default), or for every x86-64 processor with AVX2 and
# FMA (Intel since 2013, AMD since 2015): the binaries of a release, which
# then compute the same numbers on every such processor.
option(ICEEMDAN_PORTABLE "Build for any x86-64 CPU with AVX2 and FMA instead of this one (binaries to ship)" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(STATUS "Detected Intel LLVM compiler (ICX)")
    if(WIN32)
        if(ICEEMDAN_PORTABLE)
            set(CXX_FLAGS_RELEASE /O3 /QxCORE-AVX2 /Qipo /fp:fast /EHsc /DNDEBUG)
        else()
            set(CXX_FLAGS_RELEASE /O3 /QxHost /Qipo /fp:fast /EHsc /DNDEBUG)
        endif()
        set(CXX_FLAGS_SIMD /Qopenmp-simd)
    else()
        if(ICEEMDAN_PORTABLE)
            set(CXX_FLAGS_RELEASE -O3 -xCORE-AVX2 -ipo -fp-model fast -DNDEBUG)
        else()
            set(CXX_FLAGS_RELEASE -O3 -xHost -ipo -fp-model fast -DNDEBUG)
        endif()
        set(CXX_FLAGS_SIMD -qopenmp-simd)
    endif()
    set(USE_IPO TRUE)

elseif(MSVC)
    message(STATUS "Detected MSVC compiler")
    set(CXX_FLAGS_RELEASE /O2 /Ob2 /fp:fast /arch:AVX2 /GL /EHsc /DNDEBUG)
    set(USE_IPO TRUE)

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Detected GCC/Clang compiler")
    if(ICEEMDAN_PORTABLE)
        set(_iceemdan_arch -march=x86-64-v3) # AVX2, FMA, BMI1/2 (GCC 11, Clang 12 and later)
    else()
        set(_iceemdan_arch -march=native)
    endif()
    set(CXX_FLAGS_RELEASE -O3 ${_iceemdan_arch} -ffast-math -funroll-loops -flto -DNDEBUG)
    set(CXX_FLAGS_SIMD -fopenmp-simd)
    set(USE_IPO TRUE)
endif()

# CXX_FLAGS_SIMD: the loops marked "omp simd" are vectorized as marked. That is
# a compile-time hint only: nothing of OpenMP is linked or loaded.

# ============================================================================
# Main Interface Library (Header Only)
# ============================================================================

add_library(eemd INTERFACE)
add_library(iceemdan::eemd ALIAS eemd)

# The headers are grouped in folders under src/ and include one another by
# name, so each group folder is an include directory. An install puts the
# headers side by side in one folder.
target_include_directories(eemd INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/core>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/fft>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/threads>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/spline>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/rng>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/hht>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(eemd INTERFACE 
    $<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>
    ${CXX_FLAGS_SIMD}
)

# The library needs nothing else: its splines, random numbers and threads are
# its own, and the FFTs are VectorFFT's (below).
target_link_libraries(eemd INTERFACE
    Threads::Threads
)

target_compile_definitions(eemd INTERFACE
    _USE_MATH_DEFINES
)

# ============================================================================
# FFT: VectorFFT (https://github.com/Tugbars/VectorFFT), required
#
# The Hilbert transform and the instantaneous frequency run on
# VectorFFT's ready-built release library: libvfft.dll / libvfft.so, the two
# kernel libraries it loads from (AVX2, AVX-512) and vfft.h. It has a C
# interface and needs only the system's C runtime, so every compiler links it.
# CMake takes it from, in this order:
#   -DVECTORFFT_ROOT=<folder>   an unpacked release you already have
#   -DVectorFFT_DIR=<prefix>/lib/cmake/VectorFFT   a VectorFFT built from source
#   third_party/vectorfft-<version>/: the release pinned below (Windows x64,
#   Linux x64), downloaded on the first configure and checked against its
#   SHA-256. git ignores the folder.
# ============================================================================

set(ICEEMDAN_VECTORFFT_VERSION "0.8.1")
set(ICEEMDAN_VECTORFFT_SHA256_WINDOWS "321212b19d091c14ef648e1d1620e4d6e0508b484bbd16cbc48f5ac2c339a57d")
set(ICEEMDAN_VECTORFFT_SHA256_LINUX "08ea56906ec5ea409f7bf44c72391551f7adcde652ae9c247c30beb16e9cdf60")

set(VECTORFFT_ROOT "" CACHE PATH
    "Folder of an unpacked VectorFFT release (vfft.h, libvfft.dll or libvfft.so). Empty: download it")
set(ICEEMDAN_VECTORFFT_RUNTIME "")   # Windows: the DLLs a program needs beside it

if(NOT VECTORFFT_ROOT AND VectorFFT_DIR)
    find_package(VectorFFT CONFIG REQUIRED)
    set(ICEEMDAN_FFT_BACKEND "VectorFFT (installed: ${VectorFFT_DIR})")
else()
    set(_vfft_root "${VECTORFFT_ROOT}")
    if(NOT _vfft_root)
        if(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64)$")
            set(_vfft_file "vectorfft-${ICEEMDAN_VECTORFFT_VERSION}-windows-x64.zip")
            set(_vfft_sha256 "${ICEEMDAN_VECTORFFT_SHA256_WINDOWS}")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64)$")
            set(_vfft_file "vectorfft-${ICEEMDAN_VECTORFFT_VERSION}-linux-x64.tar.gz")
            set(_vfft_sha256 "${ICEEMDAN_VECTORFFT_SHA256_LINUX}")
        else()
            message(FATAL_ERROR
                "VectorFFT has no ready-built library for ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}. "
                "Build it from source (https://github.com/Tugbars/VectorFFT) and pass "
                "-DVectorFFT_DIR=<prefix>/lib/cmake/VectorFFT.")
        endif()

        set(_vfft_url
            "https://github.com/Tugbars/VectorFFT/releases/download/v${ICEEMDAN_VECTORFFT_VERSION}/${_vfft_file}")
        # Unpacked once into the source tree (ignored by git), for every build folder
        set(_vfft_third_party "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
        set(_vfft_root "${_vfft_third_party}/vectorfft-${ICEEMDAN_VECTORFFT_VERSION}")

        if(NOT EXISTS "${_vfft_root}/vfft.h")
            set(_vfft_archive "${_vfft_third_party}/${_vfft_file}")
            set(_vfft_unpack "${_vfft_third_party}/.vectorfft-unpack")
            set(_vfft_by_hand "Or download and unpack it yourself, then pass -DVECTORFFT_ROOT=<its vectorfft folder>.")
            message(STATUS "Downloading VectorFFT ${ICEEMDAN_VECTORFFT_VERSION}: ${_vfft_url}")
            file(DOWNLOAD "${_vfft_url}" "${_vfft_archive}" STATUS _vfft_status TLS_VERIFY ON)
            list(GET _vfft_status 0 _vfft_code)
            if(NOT _vfft_code EQUAL 0)
                list(GET _vfft_status 1 _vfft_why)
                file(REMOVE "${_vfft_archive}")
                message(FATAL_ERROR "Could not download ${_vfft_url}: ${_vfft_why}. ${_vfft_by_hand}")
            endif()
            file(SHA256 "${_vfft_archive}" _vfft_got)
            if(NOT _vfft_got STREQUAL _vfft_sha256)
                file(REMOVE "${_vfft_archive}")
                message(FATAL_ERROR
                    "${_vfft_url}: SHA-256 is ${_vfft_got}, expected ${_vfft_sha256}. "
                    "Configure again to download it again. ${_vfft_by_hand}")
            endif()
            # The archive holds one folder, vectorfft: unpack it aside, then
            # move it into place, so a half-unpacked folder is never used.
            file(REMOVE_RECURSE "${_vfft_unpack}" "${_vfft_root}")
            file(ARCHIVE_EXTRACT INPUT "${_vfft_archive}" DESTINATION "${_vfft_unpack}")
            file(RENAME "${_vfft_unpack}/vectorfft" "${_vfft_root}")
            file(REMOVE_RECURSE "${_vfft_unpack}")
            file(REMOVE "${_vfft_archive}")
        endif()
        set(ICEEMDAN_FFT_BACKEND "VectorFFT ${ICEEMDAN_VECTORFFT_VERSION} release library (${_vfft_root})")
    else()
        set(ICEEMDAN_FFT_BACKEND "VectorFFT release library (${_vfft_root})")
    endif()

    if(WIN32)
        set(_vfft_library "${_vfft_root}/libvfft.dll")
    else()
        set(_vfft_library "${_vfft_root}/libvfft.so")
    endif()
    if(NOT EXISTS "${_vfft_root}/vfft.h" OR NOT EXISTS "${_vfft_library}")
        message(FATAL_ERROR
            "${_vfft_root} is not an unpacked VectorFFT release: vfft.h or ${_vfft_library} is missing.")
    endif()

    # VFFT_SHARED: vfft.h declares the functions as imported from a DLL
    add_library(VectorFFT::vfft SHARED IMPORTED GLOBAL)
    set_target_properties(VectorFFT::vfft PROPERTIES
        IMPORTED_LOCATION "${_vfft_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${_vfft_root}"
        INTERFACE_COMPILE_DEFINITIONS "VFFT_SHARED"
    )
    if(WIN32)
        set_target_properties(VectorFFT::vfft PROPERTIES IMPORTED_IMPLIB "${_vfft_root}/libvfft.dll.a")
        # libvfft.dll loads one of the two kernel libraries from its own folder
        set(ICEEMDAN_VECTORFFT_RUNTIME
            "${_vfft_root}/libvfft.dll"
            "${_vfft_root}/libvfft_avx2.dll"
            "${_vfft_root}/libvfft_avx512.dll"
        )
    endif()
endif()

# An installed copy of these headers needs VectorFFT::vfft from whoever uses
# them: it is not installed with them.
target_link_libraries(eemd INTERFACE VectorFFT::vfft)

# Windows: put the VectorFFT DLLs beside the programs. On Linux the programs
# find libvfft.so through their build RPATH.
if(ICEEMDAN_VECTORFFT_RUNTIME)
    get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_multi_config)
        set(_programs_dir "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
    else()
        set(_programs_dir "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    add_custom_target(vectorfft_runtime
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_programs_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ICEEMDAN_VECTORFFT_RUNTIME} "${_programs_dir}"
        COMMENT "Copying the VectorFFT DLLs beside the programs"
        VERBATIM
    )
endif()

# ============================================================================
# ICEEMDAN Interface Library (extends EEMD)
# ============================================================================

add_library(iceemdan INTERFACE)
add_library(iceemdan::iceemdan ALIAS iceemdan)

target_include_directories(iceemdan INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/core>
    $<INSTALL_INTERFACE:include>
)

# ICEEMDAN depends on EEMD
target_link_libraries(iceemdan INTERFACE eemd)

# ============================================================================
# Executables
# ============================================================================

macro(add_eemd_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE eemd)
    if(USE_IPO)
        set_target_properties(${name} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()
    if(TARGET vectorfft_runtime)
        add_dependencies(${name} vectorfft_runtime)
    endif()
endmacro()

macro(add_iceemdan_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE iceemdan)
    if(USE_IPO)
        set_target_properties(${name} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()
    if(TARGET vectorfft_runtime)
        add_dependencies(${name} vectorfft_runtime)
    endif()
endmacro()

# ============================================================================
# Tests (tests/)
# ============================================================================

# --- Regression tests, run by ctest (exit code != 0 on failure) ---
add_iceemdan_executable(test_fft_hilbert tests/test_fft_hilbert.cpp)
add_iceemdan_executable(test_fixes tests/test_fixes.cpp)
add_iceemdan_executable(test_threads tests/test_threads.cpp)
add_iceemdan_executable(test_spline tests/test_spline.cpp)
add_iceemdan_executable(test_rng tests/test_rng.cpp)
add_iceemdan_executable(test_hht tests/test_hht.cpp)
add_iceemdan_executable(test_cpp_api tests/test_cpp_api.cpp)
add_iceemdan_executable(test_emd tests/test_emd.cpp)
add_iceemdan_executable(cpp_example examples/cpp/decompose.cpp)

enable_testing()
add_test(NAME fft_hilbert COMMAND test_fft_hilbert)
add_test(NAME fixes COMMAND test_fixes)
add_test(NAME threads COMMAND test_threads)
add_test(NAME spline COMMAND test_spline)
add_test(NAME rng COMMAND test_rng)
add_test(NAME hht COMMAND test_hht)
add_test(NAME cpp_api COMMAND test_cpp_api)
add_test(NAME emd COMMAND test_emd)
add_test(NAME cpp_example COMMAND cpp_example)

# --- Test programs run by hand ---
add_iceemdan_executable(accuracy_test tests/iceemdan_accuracy_test.cpp)

# ============================================================================
# Benchmarks (benchmarks/)
# ============================================================================

add_iceemdan_executable(emd_bench benchmarks/emd_bench.cpp)
add_iceemdan_executable(iceemdan_bench benchmarks/iceemdan_bench.cpp)
# batch_bench: many channels one by one against one decompose_batch() call
add_iceemdan_executable(batch_bench benchmarks/batch_bench.cpp)

# ============================================================================
# Intel MKL: optional, for two comparison programs only
#
# The library uses no MKL. benchmarks/spline_bench.cpp and
# benchmarks/rng_bench.cpp set the library's splines and random numbers
# against MKL's; they are built with -DEEMD_MKL_COMPARISONS=ON when MKL is
# found.
# ============================================================================

option(EEMD_MKL_COMPARISONS "Build the programs that compare the library with Intel MKL (needs MKL)" OFF)

if(EEMD_MKL_COMPARISONS)
    # 32-bit integers (LP64), and MKL's sequential layer: the programs call
    # MKL from one thread and need no OpenMP runtime
    set(MKL_INTERFACE "lp64" CACHE STRING "MKL interface (lp64 or ilp64)" FORCE)
    set(MKL_THREADING "sequential" CACHE STRING "MKL threading" FORCE)

    # MinGW cannot link MKL's layered libraries, import or static (they hold
    # objects built for the MSVC runtime): it links MKL's single dynamic library,
    # mkl_rt. FORCE: MKL's own configuration caches "dynamic" when it runs first.
    if(MINGW)
        set(MKL_LINK "sdl" CACHE STRING "MKL linking: static, dynamic or sdl" FORCE)
    endif()

    # Where MKL comes from, first hit:
    #   -DMKL_DIR=<folder of MKLConfig.cmake>
    #   MKLROOT in the environment (oneAPI's setvars sets it)
    #   oneAPI's standard install folder
    #   CMake's own search (CMAKE_PREFIX_PATH, ...)
    if(NOT MKL_DIR)
        set(_mkl_roots "")
        if(DEFINED ENV{MKLROOT})
            list(APPEND _mkl_roots "$ENV{MKLROOT}")
        endif()
        if(DEFINED ENV{ONEAPI_ROOT})
            list(APPEND _mkl_roots "$ENV{ONEAPI_ROOT}/mkl/latest")
        endif()
        list(APPEND _mkl_roots "C:/Program Files (x86)/Intel/oneAPI/mkl/latest" "/opt/intel/oneapi/mkl/latest")
        foreach(_mkl_root IN LISTS _mkl_roots)
            if(EXISTS "${_mkl_root}/lib/cmake/mkl/MKLConfig.cmake")
                file(TO_CMAKE_PATH "${_mkl_root}/lib/cmake/mkl" _mkl_dir)
                set(MKL_DIR "${_mkl_dir}" CACHE PATH "Folder of MKLConfig.cmake")
                break()
            endif()
        endforeach()
    endif()

    # MKL's configuration caches the folder of mkl.h and keeps it when MKL_DIR
    # changes: a build folder first configured against another MKL would go on
    # compiling with that one's headers (and whatever else its include folder
    # holds). Look it up again on every configure.
    unset(MKL_INCLUDE CACHE)

    find_package(MKL CONFIG QUIET)
    if(MKL_FOUND)
        message(STATUS "MKL found, for the comparison programs: ${MKL_ROOT}")
    else()
        message(WARNING "EEMD_MKL_COMPARISONS is ON but MKL was not found: the comparison programs are skipped")
    endif()

    if(MKL_FOUND)
        # spline_bench: the library's envelope splines against MKL Data Fitting
        # rng_bench: the library's random numbers against MKL's, which it used before
        foreach(_cmp spline_bench rng_bench)
            add_eemd_executable(${_cmp} benchmarks/${_cmp}.cpp)
            target_link_libraries(${_cmp} PRIVATE MKL::MKL)
            target_compile_definitions(${_cmp} PRIVATE MKL_LP64)
            if(MKL_LINK STREQUAL "sdl")
                # mkl_rt picks its threading layer at run time: the programs ask for the sequential one
                target_compile_definitions(${_cmp} PRIVATE EEMD_MKL_SINGLE_DYNAMIC_LIBRARY=1)
            endif()
        endforeach()
    endif()
endif()


# ============================================================================
# Validation against the reference code (validation/README.md)
# ============================================================================

add_iceemdan_executable(decompose_csv validation/decompose_csv.cpp)

# Data for the README figures and timings (validation/figures/README.md)
add_iceemdan_executable(figure_tool validation/figures/figure_tool.cpp)

# The library and the authors' code on the same noise (validation/same_noise/README.md)
add_iceemdan_executable(same_noise validation/same_noise/same_noise.cpp)

# ============================================================================
# C interface (src/c_api), built as the shared library the Python binding loads
# ============================================================================

add_library(iceemdan_shared SHARED src/c_api/iceemdan_c.cpp)
target_link_libraries(iceemdan_shared PRIVATE iceemdan)
# include/iceemdan.h: the interface, for the library itself and its users
target_include_directories(iceemdan_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(iceemdan_shared PROPERTIES
    OUTPUT_NAME "iceemdan"
    POSITION_INDEPENDENT_CODE ON
)
if(WIN32)
    # iceemdan.dll (not libiceemdan.dll), whatever the compiler
    set_target_properties(iceemdan_shared PROPERTIES PREFIX "" IMPORT_PREFIX "")
endif()
if(MINGW)
    # libstdc++, libgcc and winpthreads linked in: the DLL needs only what
    # Windows has (and VectorFFT's DLLs), not the MinGW runtime DLLs
    target_link_options(iceemdan_shared PRIVATE -static)
endif()
if(USE_IPO)
    set_target_properties(iceemdan_shared PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
endif()
if(TARGET vectorfft_runtime)
    add_dependencies(iceemdan_shared vectorfft_runtime)
endif()
if(WIN32)
    # Copy the DLL, and the VectorFFT DLLs it loads, to the python folder for easy testing
    add_custom_command(TARGET iceemdan_shared POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:iceemdan_shared>
        ${ICEEMDAN_VECTORFFT_RUNTIME}
        ${CMAKE_CURRENT_SOURCE_DIR}/python/
    )
endif()

# The C interface from C: an example for users, and a test of include/iceemdan.h
# compiled as C against the shared library
add_executable(c_example examples/c/decompose.c)
add_executable(test_c_api tests/test_c_api.c)
foreach(c_program c_example test_c_api)
    target_link_libraries(${c_program} PRIVATE iceemdan_shared)
    if(UNIX)
        target_link_libraries(${c_program} PRIVATE m)
    endif()
    if(TARGET vectorfft_runtime)
        add_dependencies(${c_program} vectorfft_runtime)
    endif()
endforeach()
add_test(NAME c_api COMMAND test_c_api)
add_test(NAME c_example COMMAND c_example)

# The Python module against the shared library, where Python and NumPy are there
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                    RESULT_VARIABLE _numpy_missing OUTPUT_QUIET ERROR_QUIET)
    if(NOT _numpy_missing)
        add_test(NAME python COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python.py)
    endif()
endif()

# ============================================================================
# Installation (optional)
# ============================================================================

include(GNUInstallDirs)

install(TARGETS eemd iceemdan
    EXPORT iceemdan_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# The C++ interface (iceemdan.hpp) and the implementation headers it stands on
install(FILES
    include/iceemdan.hpp
    include/emd.hpp
    src/core/iceemdan_engine.hpp
    src/core/emd_engine.hpp
    src/core/eemd_signal.hpp
    src/core/eemd_base.hpp
    src/core/eemd_extrema.hpp
    src/core/eemd_sift.hpp
    src/core/eemd_local_mean.hpp
    src/core/eemd_noise.hpp
    src/core/eemd_system.hpp
    src/fft/eemd_fft.hpp
    src/threads/eemd_threads.hpp
    src/spline/cubic_spline.hpp
    src/rng/eemd_rng.hpp
    src/hht/eemd_hht.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# The C interface: include/iceemdan.h and the shared library (on Windows
# iceemdan.dll in bin, its import library in lib)
install(TARGETS iceemdan_shared
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES include/iceemdan.h include/iceemdan_diagnostics.h include/emd.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT iceemdan_targets
    FILE iceemdan-targets.cmake
    NAMESPACE iceemdan::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/iceemdan
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== ICEEMDAN Configuration ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Portable (AVX2 + FMA, for release binaries): ${ICEEMDAN_PORTABLE}")
message(STATUS "  MKL comparison programs: ${EEMD_MKL_COMPARISONS}")
message(STATUS "  C++ interfaces: include/iceemdan.hpp, include/emd.hpp")
message(STATUS "  C interfaces: include/iceemdan.h, include/emd.h (the shared library iceemdan)")
message(STATUS "  FFT backend: ${ICEEMDAN_FFT_BACKEND}")
message(STATUS "")