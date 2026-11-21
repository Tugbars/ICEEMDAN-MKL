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
# Sequential MKL: the library runs its own threads (eemd_threads.hpp) and calls
# MKL from each of them, so MKL needs no threads of its own and no OpenMP runtime.
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

find_package(MKL CONFIG REQUIRED)

message(STATUS "MKL found: ${MKL_ROOT}")
message(STATUS "MKL interface: ${MKL_INTERFACE}")

# ============================================================================
# Threads: the C++ standard library's (std::thread). No OpenMP runtime.
# ============================================================================
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ============================================================================
# Compiler Detection & Flags
# ============================================================================

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(STATUS "Detected Intel LLVM compiler (ICX)")
    if(WIN32)
        set(CXX_FLAGS_RELEASE /O3 /QxHost /Qipo /fp:fast /EHsc /DNDEBUG)
        set(CXX_FLAGS_SIMD /Qopenmp-simd)
    else()
        set(CXX_FLAGS_RELEASE -O3 -xHost -ipo -fp-model fast -DNDEBUG)
        set(CXX_FLAGS_SIMD -qopenmp-simd)
    endif()
    set(USE_IPO TRUE)

elseif(MSVC)
    message(STATUS "Detected MSVC compiler")
    set(CXX_FLAGS_RELEASE /O2 /Ob2 /fp:fast /arch:AVX2 /GL /EHsc /DNDEBUG)
    set(USE_IPO TRUE)

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Detected GCC/Clang compiler")
    set(CXX_FLAGS_RELEASE -O3 -march=native -ffast-math -funroll-loops -flto -DNDEBUG)
    set(CXX_FLAGS_SIMD -fopenmp-simd)
    set(USE_IPO TRUE)
endif()

# CXX_FLAGS_SIMD: the loops marked "omp simd" are vectorized as marked. That is
# a compile-time hint only: nothing of OpenMP is linked or loaded.

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
    ${CXX_FLAGS_SIMD}
)

target_link_libraries(eemd_mkl INTERFACE 
    MKL::MKL 
    Threads::Threads
)

target_compile_definitions(eemd_mkl INTERFACE 
    MKL_LP64 
    _USE_MATH_DEFINES
)

# mkl_rt picks its threading layer at run time: the headers ask for the
# sequential one (with the layered libraries, MKL_THREADING above links it)
if(MKL_LINK STREQUAL "sdl")
    target_compile_definitions(eemd_mkl INTERFACE EEMD_MKL_SINGLE_DYNAMIC_LIBRARY=1)
endif()

# ============================================================================
# FFT: VectorFFT (https://github.com/Tugbars/VectorFFT), required
#
# The Hilbert transform, instantaneous frequency and spectral entropy run on
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

# An installed copy of these headers needs VectorFFT::vfft (and MKL::MKL) from
# whoever uses them: neither is installed with them.
target_link_libraries(eemd_mkl INTERFACE VectorFFT::vfft)

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
    if(TARGET vectorfft_runtime)
        add_dependencies(${name} vectorfft_runtime)
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
    if(TARGET vectorfft_runtime)
        add_dependencies(${name} vectorfft_runtime)
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

if(EXISTS "${TEST_DIR}/test_threads.cpp")
    add_iceemdan_executable(test_threads ${TEST_DIR}/test_threads.cpp)
endif()

enable_testing()
if(TARGET test_fft_hilbert)
    add_test(NAME fft_hilbert COMMAND test_fft_hilbert)
endif()
if(TARGET test_fixes)
    add_test(NAME fixes COMMAND test_fixes)
endif()
if(TARGET test_threads)
    add_test(NAME threads COMMAND test_threads)
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
    eemd_threads.hpp
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