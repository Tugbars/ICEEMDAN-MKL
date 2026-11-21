cmake_minimum_required(VERSION 3.20)

project(eemd_mkl 
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "EEMD/ICEEMDAN with Intel MKL"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# MKL Configuration
# ============================================================================

# CRITICAL: Force LP64 (32-bit integers)
set(MKL_INTERFACE "lp64" CACHE STRING "MKL interface (lp64 or ilp64)" FORCE)
set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading" FORCE)

find_package(MKL CONFIG REQUIRED)

message(STATUS "MKL found: ${MKL_ROOT}")
message(STATUS "MKL interface: ${MKL_INTERFACE}")

# ============================================================================
# OpenMP Support
# ============================================================================
find_package(OpenMP REQUIRED)

# ============================================================================
# Compiler Detection & Flags
# ============================================================================

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(STATUS "Detected Intel LLVM compiler (ICX)")
    if(WIN32)
        set(CXX_FLAGS_RELEASE /O3 /QxHost /Qipo /fp:fast /Qopenmp /EHsc /DNDEBUG)
    else()
        set(CXX_FLAGS_RELEASE -O3 -xHost -ipo -fp-model fast -qopenmp -DNDEBUG)
    endif()
    set(USE_IPO TRUE)

elseif(MSVC)
    message(STATUS "Detected MSVC compiler")
    set(CXX_FLAGS_RELEASE /O2 /Ob2 /fp:fast /arch:AVX2 /GL /EHsc /DNDEBUG /openmp)
    set(USE_IPO TRUE)

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Detected GCC/Clang compiler")
    set(CXX_FLAGS_RELEASE -O3 -march=native -ffast-math -funroll-loops -flto -fopenmp -DNDEBUG)
    set(USE_IPO TRUE)
endif()

# ============================================================================
# Main Interface Library (Header Only)
# ============================================================================

add_library(eemd_mkl INTERFACE)
add_library(eemd::eemd_mkl ALIAS eemd_mkl)

target_include_directories(eemd_mkl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(eemd_mkl INTERFACE 
    $<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>
)

target_link_libraries(eemd_mkl INTERFACE 
    MKL::MKL 
    OpenMP::OpenMP_CXX
)

target_compile_definitions(eemd_mkl INTERFACE 
    MKL_LP64 
    _USE_MATH_DEFINES
)

# ============================================================================
# FFT backend: VectorFFT (third_party/vectorfft, a git submodule)
#
# The Hilbert transform, instantaneous frequency and spectral entropy run on
# VectorFFT when it is available, on MKL DFTI otherwise (and for any length
# VectorFFT refuses). Fetch it with:
#   git submodule update --init --depth 1 third_party/vectorfft
# or point CMake at an installed copy: -DVectorFFT_DIR=<prefix>/lib/cmake/VectorFFT
# VectorFFT builds with GCC or Clang (MinGW on Windows), not with MSVC.
# ============================================================================

option(ICEEMDAN_USE_VECTORFFT "Use VectorFFT for the FFTs (Hilbert transform, spectral entropy)" ON)
set(ICEEMDAN_FFT_BACKEND "MKL DFTI")

if(ICEEMDAN_USE_VECTORFFT)
    find_package(VectorFFT CONFIG QUIET)
    if(TARGET VectorFFT::vfft)
        set(ICEEMDAN_FFT_BACKEND "VectorFFT (installed: ${VectorFFT_DIR})")
    elseif(MSVC)
        message(WARNING "VectorFFT is built with GCC or Clang, not MSVC: using MKL DFTI. "
                        "To use VectorFFT here, install a MinGW/Clang build and set VectorFFT_DIR.")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vectorfft/CMakeLists.txt")
        # The Python binding is a shared library: build libvfft.a as PIC
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        add_subdirectory(third_party/vectorfft EXCLUDE_FROM_ALL)
        set(ICEEMDAN_FFT_BACKEND "VectorFFT (third_party/vectorfft)")
    else()
        message(WARNING "third_party/vectorfft is missing: using MKL DFTI. Run "
                        "'git submodule update --init --depth 1 third_party/vectorfft' to use VectorFFT.")
    endif()

    if(TARGET VectorFFT::vfft)
        # BUILD_INTERFACE: an installed copy of these headers is configured for
        # MKL DFTI (it does not export the bundled VectorFFT).
        target_link_libraries(eemd_mkl INTERFACE $<BUILD_INTERFACE:VectorFFT::vfft>)
        target_compile_definitions(eemd_mkl INTERFACE $<BUILD_INTERFACE:EEMD_USE_VECTORFFT=1>)
    endif()
endif()

# ============================================================================
# ICEEMDAN Interface Library (extends EEMD)
# ============================================================================

add_library(iceemdan_mkl INTERFACE)
add_library(eemd::iceemdan_mkl ALIAS iceemdan_mkl)

target_include_directories(iceemdan_mkl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# ICEEMDAN depends on EEMD
target_link_libraries(iceemdan_mkl INTERFACE eemd_mkl)

# ============================================================================
# Executables
# ============================================================================

macro(add_eemd_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE eemd_mkl)
    if(USE_IPO)
        set_target_properties(${name} PROPERTIES 
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()
endmacro()

macro(add_iceemdan_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE iceemdan_mkl)
    if(USE_IPO)
        set_target_properties(${name} PROPERTIES 
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()
endmacro()

# ============================================================================
# Examples (in root)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/example_simple.cpp")
    add_iceemdan_executable(example_simple example_simple.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/example_finance_csv.cpp")
    add_iceemdan_executable(example_finance_csv example_finance_csv.cpp)
endif()

# ============================================================================
# Tests and Benchmarks (in test/ subdirectory)
# ============================================================================

set(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")

# --- Core EEMD tests ---
if(EXISTS "${TEST_DIR}/eemd_bench.cpp")
    add_eemd_executable(eemd_bench ${TEST_DIR}/eemd_bench.cpp)
endif()

if(EXISTS "${TEST_DIR}/eemd_test.cpp")
    add_eemd_executable(eemd_test ${TEST_DIR}/eemd_test.cpp)
endif()

if(EXISTS "${TEST_DIR}/eemd_debug.cpp")
    add_eemd_executable(eemd_debug ${TEST_DIR}/eemd_debug.cpp)
endif()

# --- ICEEMDAN tests ---
if(EXISTS "${TEST_DIR}/iceemdan_bench.cpp")
    add_iceemdan_executable(iceemdan_bench ${TEST_DIR}/iceemdan_bench.cpp)
endif()

if(EXISTS "${TEST_DIR}/iceemdan_test.cpp")
    add_iceemdan_executable(iceemdan_test ${TEST_DIR}/iceemdan_test.cpp)
endif()

if(EXISTS "${TEST_DIR}/iceemdan_accuracy_test.cpp")
    add_iceemdan_executable(accuracy_test ${TEST_DIR}/iceemdan_accuracy_test.cpp)
endif()

if(EXISTS "${TEST_DIR}/iceemdan_finance_bench.cpp")
    add_iceemdan_executable(iceemdan_finance_bench ${TEST_DIR}/iceemdan_finance_bench.cpp)
endif()

# --- Optimization benchmarks ---
if(EXISTS "${TEST_DIR}/bench_adaptive.cpp")
    add_iceemdan_executable(bench_adaptive ${TEST_DIR}/bench_adaptive.cpp)
endif()

if(EXISTS "${TEST_DIR}/bench_snumber.cpp")
    add_iceemdan_executable(bench_snumber ${TEST_DIR}/bench_snumber.cpp)
endif()

if(EXISTS "${TEST_DIR}/bench_simd.cpp")
    add_iceemdan_executable(bench_simd ${TEST_DIR}/bench_simd.cpp)
endif()

if(EXISTS "${TEST_DIR}/test_akima.cpp")
    add_iceemdan_executable(test_akima ${TEST_DIR}/test_akima.cpp)
endif()

if(EXISTS "${TEST_DIR}/spline_bench.cpp")
    add_eemd_executable(spline_bench ${TEST_DIR}/spline_bench.cpp)
endif()

# --- Validation against the reference code (validation/README.md) ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/validation/decompose_csv.cpp")
    add_iceemdan_executable(decompose_csv ${CMAKE_CURRENT_SOURCE_DIR}/validation/decompose_csv.cpp)
endif()

# --- Regression tests (exit code != 0 on failure) ---
if(EXISTS "${TEST_DIR}/test_fft_hilbert.cpp")
    add_iceemdan_executable(test_fft_hilbert ${TEST_DIR}/test_fft_hilbert.cpp)
endif()

if(EXISTS "${TEST_DIR}/test_fixes.cpp")
    add_iceemdan_executable(test_fixes ${TEST_DIR}/test_fixes.cpp)
endif()

enable_testing()
if(TARGET test_fft_hilbert)
    add_test(NAME fft_hilbert COMMAND test_fft_hilbert)
endif()
if(TARGET test_fixes)
    add_test(NAME fixes COMMAND test_fixes)
endif()

# ============================================================================
# Python Bindings (Shared Library)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/iceemdan_c.cpp")
    add_library(iceemdan_shared SHARED iceemdan_c.cpp)
    target_link_libraries(iceemdan_shared PRIVATE iceemdan_mkl)
    set_target_properties(iceemdan_shared PROPERTIES
        OUTPUT_NAME "iceemdan"
        POSITION_INDEPENDENT_CODE ON
    )
    if(USE_IPO)
        set_target_properties(iceemdan_shared PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()
    if(WIN32)
        # Copy DLL to python folder for easy testing
        add_custom_command(TARGET iceemdan_shared POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:iceemdan_shared>
            ${CMAKE_CURRENT_SOURCE_DIR}/python/
        )
    endif()
endif()

# ============================================================================
# Installation (optional)
# ============================================================================

include(GNUInstallDirs)

install(TARGETS eemd_mkl iceemdan_mkl
    EXPORT eemd_mkl_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES 
    eemd_mkl.hpp 
    eemd_fft.hpp
    iceemdan_mkl.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT eemd_mkl_targets
    FILE eemd_mkl-targets.cmake
    NAMESPACE eemd::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/eemd_mkl
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== EEMD/ICEEMDAN-MKL Configuration ===")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  MKL interface: ${MKL_INTERFACE}")
message(STATUS "  MKL threading: ${MKL_THREADING}")
message(STATUS "  EEMD header: eemd_mkl.hpp")
message(STATUS "  ICEEMDAN header: iceemdan_mkl.hpp")
message(STATUS "  FFT backend: ${ICEEMDAN_FFT_BACKEND}")
message(STATUS "")