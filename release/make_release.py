"""
Builds a binary release of VectorEMD's C interface.

    python release/make_release.py [--generator NAME]

The library is built for every x86-64 processor with AVX2 and FMA
(ICEEMDAN_PORTABLE=ON), tested, and packed with its header, VectorFFT's
libraries, an example, a short guide and the licences:

    dist/iceemdan-<version>-windows-x64.zip        (or -linux-x64.tar.gz)
      include/iceemdan.h, include/iceemdan_diagnostics.h, include/emd.h
      bin/iceemdan.dll, bin/libvfft*.dll           (Linux: lib/libiceemdan.so, lib/libvfft.so)
      lib/iceemdan.lib                             the import library for Visual Studio
      examples/decompose.c
      README.txt, LICENSE.txt, LICENSE-VectorFFT.txt

Before packing it checks that the library exports every function of the
header and needs no compiler runtime of its own (Windows), and after packing
it compiles the example against the package alone, as a user would, with
every compiler it finds (GCC, and Visual Studio's cl), and runs it.

Needs CMake and a C/C++ compiler on PATH. The version is the one in
include/iceemdan.h.
"""
import argparse
import glob
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = os.name == "nt"


def run(cmd, cwd=None, quiet=False):
    print("  $ " + " ".join(cmd))
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"failed ({out.returncode}):\n{out.stdout}\n{out.stderr}")
    if not quiet and out.stdout.strip():
        print("    " + out.stdout.strip().replace("\n", "\n    "))
    return out.stdout


HEADERS = ("iceemdan.h", "iceemdan_diagnostics.h", "emd.h")


def header_version_and_functions():
    text = "".join(open(os.path.join(ROOT, "include", h), encoding="utf-8").read() for h in HEADERS)
    version = re.search(r'#define ICEEMDAN_VERSION_STRING "([^"]+)"', text).group(1)
    functions = re.findall(r"(?:ICEEMDAN|EMD)_API\s+[^;(]*?\b((?:iceemdan|emd)_\w+)\s*\(", text)
    return version, sorted(set(functions))


def pe_exports_and_imports(path):
    """The names a Windows DLL exports and the DLLs it imports (PE32+)."""
    data = open(path, "rb").read()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    n_sections = struct.unpack_from("<H", data, pe + 6)[0]
    opt = pe + 24
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    dirs = opt + (112 if struct.unpack_from("<H", data, opt)[0] == 0x20B else 96)
    sections = []
    for i in range(n_sections):
        s = opt + opt_size + 40 * i
        vsize, va, rsize, raw = struct.unpack_from("<IIII", data, s + 8)
        sections.append((va, max(vsize, rsize), raw))

    def off(rva):
        for va, size, raw in sections:
            if va <= rva < va + size:
                return rva - va + raw
        raise ValueError(rva)

    def cstr(rva):
        o = off(rva)
        return data[o:data.index(b"\0", o)].decode()

    exports = []
    exp_rva = struct.unpack_from("<I", data, dirs)[0]
    if exp_rva:
        e = off(exp_rva)
        n_names = struct.unpack_from("<I", data, e + 24)[0]  # NumberOfNames
        names = off(struct.unpack_from("<I", data, e + 32)[0])  # AddressOfNames
        exports = [cstr(struct.unpack_from("<I", data, names + 4 * i)[0]) for i in range(n_names)]
    imports = []
    imp_rva = struct.unpack_from("<I", data, dirs + 8)[0]
    if imp_rva:
        d = off(imp_rva)
        while True:
            name_rva = struct.unpack_from("<I", data, d + 12)[0]
            if name_rva == 0:
                break
            imports.append(cstr(name_rva))
            d += 20
    return exports, imports


def visual_studio():
    """(lib.exe, vcvars64.bat) of the newest Visual Studio with C++, or (None, None)."""
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        return None, None
    out = subprocess.run([vswhere, "-latest", "-products", "*", "-requires",
                          "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        return None, None
    vcvars = os.path.join(out, "VC", "Auxiliary", "Build", "vcvars64.bat")
    libs = sorted(glob.glob(os.path.join(out, "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "lib.exe")))
    return (libs[-1] if libs else None), (vcvars if os.path.exists(vcvars) else None)


def import_library(dll, functions, out_lib, work):
    """iceemdan.lib for Visual Studio: from a .def of the header's functions."""
    def_file = os.path.join(work, "iceemdan.def")
    with open(def_file, "w") as f:
        f.write("LIBRARY iceemdan.dll\nEXPORTS\n" + "".join(f"    {name}\n" for name in functions))
    lib_exe, _ = visual_studio()
    if lib_exe:
        run([lib_exe, "/nologo", f"/def:{def_file}", "/machine:x64", f"/out:{out_lib}"])
        for extra in (out_lib[:-4] + ".exp",):
            if os.path.exists(extra):
                os.remove(extra)
        return "lib.exe (Visual Studio)"
    dlltool = shutil.which("dlltool")
    if dlltool:
        run([dlltool, "-d", def_file, "-D", "iceemdan.dll", "-l", out_lib])
        return "dlltool"
    sys.exit("no lib.exe (Visual Studio) or dlltool (MinGW) to make the import library with")


def try_package(stage, compilers):
    """Compiles examples/decompose.c against the package alone and runs it."""
    results = []
    for name in compilers:
        work = tempfile.mkdtemp(prefix="iceemdan_check_")
        try:
            for item in ("include", "lib", "examples"):
                shutil.copytree(os.path.join(stage, item), os.path.join(work, item))
            if WINDOWS:
                for dll in glob.glob(os.path.join(stage, "bin", "*.dll")):
                    shutil.copy(dll, work)
            if name == "gcc":
                exe = os.path.join(work, "decompose" + (".exe" if WINDOWS else ""))
                if WINDOWS:  # MinGW links the DLL itself
                    shutil.copytree(os.path.join(stage, "bin"), os.path.join(work, "bin"))
                    cmd = ["gcc", "-O2", "-Iinclude", "examples/decompose.c", "bin/iceemdan.dll", "-o", exe]
                else:
                    cmd = ["gcc", "-O2", "-Iinclude", "examples/decompose.c", "-Llib", "-liceemdan", "-lm",
                           "-Wl,-rpath,$ORIGIN/lib", "-o", exe]
                run(cmd, cwd=work, quiet=True)
            else:  # Visual Studio's cl, in its own environment
                _, vcvars = visual_studio()
                bat = os.path.join(work, "build.bat")
                with open(bat, "w") as f:
                    f.write(f'@call "{vcvars}" >nul\r\n'
                            "cl /nologo /O2 /Iinclude examples\\decompose.c lib\\iceemdan.lib /Fe:decompose.exe\r\n")
                run(["cmd", "/c", bat], cwd=work, quiet=True)
                exe = os.path.join(work, "decompose.exe")
            out = subprocess.run([exe], cwd=work, capture_output=True, text=True)
            ok = out.returncode == 0 and "IMFs" in out.stdout
            results.append((name, ok))
            print(f"  {name}: {'works' if ok else 'FAILED'}")
            if not ok:
                print(out.stdout, out.stderr)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--generator", help="CMake generator (default: CMake's choice)")
    args = ap.parse_args()

    version, functions = header_version_and_functions()
    system = "windows" if WINDOWS else platform.system().lower()
    arch = {"amd64": "x64", "x86_64": "x64"}.get(platform.machine().lower(), platform.machine().lower())
    name = f"iceemdan-{version}-{system}-{arch}"
    dist = os.path.join(ROOT, "dist")
    build = os.path.join(dist, "build")
    stage = os.path.join(dist, name)
    print(f"VectorEMD {version}: {len(functions)} functions in the headers -> dist/{name}")

    print("1. Build (portable: AVX2 + FMA)")
    shutil.rmtree(build, ignore_errors=True)
    shutil.rmtree(stage, ignore_errors=True)
    cmd = ["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DICEEMDAN_PORTABLE=ON"]
    if args.generator:
        cmd += ["-G", args.generator]
    run(cmd, quiet=True)
    run(["cmake", "--build", build, "--config", "Release", "--parallel",
         "--target", "iceemdan_shared", "test_c_api", "c_example"], quiet=True)
    cache = open(os.path.join(build, "CMakeCache.txt"), encoding="utf-8", errors="replace").read()
    msvc_build = "CMAKE_CXX_COMPILER_ID:INTERNAL=MSVC" in cache or os.path.isdir(os.path.join(build, "Release"))
    out_dir = os.path.join(build, "Release") if os.path.isdir(os.path.join(build, "Release")) else build

    print("2. Test")
    for program in ("test_c_api", "c_example"):
        exe = os.path.join(out_dir, program + (".exe" if WINDOWS else ""))
        out = subprocess.run([exe], cwd=out_dir, capture_output=True, text=True)
        print("    " + out.stdout.strip().splitlines()[-1])
        if out.returncode != 0:
            sys.exit(out.stdout + out.stderr)

    print("3. Stage")
    for d in ("include", "lib", "examples") + (("bin",) if WINDOWS else ()):
        os.makedirs(os.path.join(stage, d))
    for h in HEADERS:
        shutil.copy(os.path.join(ROOT, "include", h), os.path.join(stage, "include"))
    shutil.copy(os.path.join(ROOT, "examples", "c", "decompose.c"), os.path.join(stage, "examples"))
    shutil.copy(os.path.join(ROOT, "LICENSE"), os.path.join(stage, "LICENSE.txt"))
    vfft = sorted(glob.glob(os.path.join(ROOT, "third_party", "vectorfft-*")))
    if vfft and os.path.exists(os.path.join(vfft[-1], "LICENSE")):
        shutil.copy(os.path.join(vfft[-1], "LICENSE"), os.path.join(stage, "LICENSE-VectorFFT.txt"))
    readme = open(os.path.join(ROOT, "release", "README.txt"), encoding="utf-8").read()
    with open(os.path.join(stage, "README.txt"), "w", encoding="utf-8", newline="\r\n" if WINDOWS else "\n") as f:
        f.write(readme.replace("@VERSION@", version))

    if WINDOWS:
        dll = os.path.join(out_dir, "iceemdan.dll")
        exports, imports = pe_exports_and_imports(dll)
        missing = [f for f in functions if f not in exports]
        if missing:
            sys.exit(f"iceemdan.dll lacks functions of the header: {missing}")
        allowed = re.compile(r"^(kernel32|libvfft|api-ms-win-crt-.*|vcruntime140(_1)?|msvcp140)\.dll$", re.I)
        foreign = [i for i in imports if not allowed.match(i)]
        if foreign:
            sys.exit(f"iceemdan.dll needs DLLs a user would not have: {foreign}")
        print(f"    iceemdan.dll exports all {len(functions)} functions; it needs only {', '.join(imports[:2])}, ...")
        shutil.copy(dll, os.path.join(stage, "bin"))
        for v in sorted(glob.glob(os.path.join(out_dir, "libvfft*.dll"))):
            shutil.copy(v, os.path.join(stage, "bin"))
        lib = os.path.join(stage, "lib", "iceemdan.lib")
        if msvc_build and os.path.exists(os.path.join(out_dir, "iceemdan.lib")):
            shutil.copy(os.path.join(out_dir, "iceemdan.lib"), lib)
            print("    import library: the build's")
        else:
            print("    import library: " + import_library(dll, functions, lib, build))
    else:
        for so in glob.glob(os.path.join(build, "libiceemdan.so*")):
            shutil.copy(so, os.path.join(stage, "lib"), follow_symlinks=True)
        if vfft:
            for so in glob.glob(os.path.join(vfft[-1], "**", "libvfft*.so*"), recursive=True):
                shutil.copy(so, os.path.join(stage, "lib"), follow_symlinks=True)

    print("4. The package alone, as a user compiles against it")
    compilers = ["gcc"] if shutil.which("gcc") else []
    if WINDOWS and visual_studio()[1]:
        compilers.append("cl")
    results = try_package(stage, compilers)
    if not all(ok for _, ok in results):
        sys.exit("the package does not work")

    print("5. Pack")
    if WINDOWS:
        archive = shutil.make_archive(stage, "zip", dist, name)
    else:
        archive = stage + ".tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(stage, arcname=name)
    size = os.path.getsize(archive) / 1e6
    print(f"    {os.path.relpath(archive, ROOT)}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
