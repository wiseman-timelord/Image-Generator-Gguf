#!/usr/bin/env python3
"""
installer.py - Standalone setup script for Image Generator GGUF.

Detects hardware (CPU cores/threads, Vulkan GPUs), creates a venv,
installs Python dependencies, then installs llama-cpp-python and
stable-diffusion-cpp-python via pip wheels (pre-built where available,
compiled from source with CPU/Vulkan flags where not).

Writes:
    ./data/constants.ini   - hardware constants, thread counts, GPU info
    ./data/persistent.json - default user config (only if absent)

No imports from scripts.* — this is self-contained.
"""

from __future__ import annotations

import argparse
import configparser
import ctypes
import json
import math
import os
import platform
import shutil
import stat
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
_DATA_DIR    = _ROOT / "data"
_VENV_DIR    = _ROOT / "venv"
_CONST_PATH  = _DATA_DIR / "constants.ini"
_PERSIST_PATH = _DATA_DIR / "persistent.json"
_MODELS_DIR  = _ROOT / "models"
_OUTPUT_DIR  = _ROOT / "output"

REQUIREMENTS = [
    "gradio>=5.0",
    "Pillow>=10.0",
    "numpy>=1.26",
]

# ---------------------------------------------------------------------------
# Backend wheel constants
# ---------------------------------------------------------------------------

# llama-cpp-python: pre-built Vulkan wheel index (abetlen)
LLAMA_CPP_VULKAN_INDEX  = "https://abetlen.github.io/llama-cpp-python/whl/vulkan"
# llama-cpp-python: pre-built CPU wheel (eswarthammana, pinned stable version)
LLAMA_CPP_CPU_VERSION   = "0.3.16"
LLAMA_CPP_CPU_WHEEL_BASE = (
    "https://github.com/eswarthammana/llama-cpp-wheels/releases/download/"
    "v{ver}/llama_cpp_python-{ver}-{pytag}-{pytag}-win_amd64.whl"
)
# stable-diffusion-cpp-python: source build only (no Vulkan binary wheel exists)
SD_CPP_PACKAGE          = "stable-diffusion-cpp-python"
# Retry settings for pip builds
BUILD_MAX_RETRIES       = 3
BUILD_RETRY_DELAY       = 10
# Inactivity timeout for long pip builds (seconds)
BUILD_INACTIVITY_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str = "") -> None:
    print(f"  {msg}" if msg else "")


def header(title: str) -> None:
    os.system("cls" if platform.system() == "Windows" else "clear")
    print()
    print("  " + "=" * 78)
    print(f"      {title}")
    print("  " + "=" * 78)
    print()


def section(title: str) -> None:
    """Inline section label — no cls, scrolls with the install log."""
    print()
    print(f"  {title}")
    print("  " + "-" * len(title))


def _safe_rmtree(path: Path) -> bool:
    """Remove a directory tree on Windows, clearing read-only flags first.
    Git pack files and index files are often marked read-only, which causes
    a plain shutil.rmtree to raise PermissionError on Windows.
    Returns True if the directory is gone afterwards.
    """
    def _on_error(func, fpath, exc_info):
        # Clear read-only bit and retry
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception:
            pass

    if not path.exists():
        return True
    shutil.rmtree(path, onerror=_on_error)
    return not path.exists()


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in (_DATA_DIR, _MODELS_DIR, _OUTPUT_DIR, _ROOT / "scripts"):
        d.mkdir(parents=True, exist_ok=True)
    init = _ROOT / "scripts" / "__init__.py"
    if not init.exists():
        init.write_text("# scripts package\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------

def detect_cpu() -> Dict[str, Any]:
    logical = os.cpu_count() or 4
    default_threads = max(1, math.ceil(logical * 0.85))
    arch = _cpu_arch()
    brand = platform.processor() or "unknown"
    vendor = "unknown"

    info: Dict[str, Any] = {
        "arch": arch,
        "brand": brand,
        "vendor": vendor,
        "cores_logical": logical,
        "default_threads": default_threads,
        "has_avx": False,
        "has_avx2": False,
        "has_f16c": False,
        "has_fma": False,
        "has_avx512": False,
        "has_sse4_2": False,
        "has_aocl": False,
    }

    if platform.system() == "Windows":
        try:
            r = subprocess.run(["wmic", "cpu", "get", "Name", "/value"],
                               capture_output=True, text=True, timeout=10)
            for line in r.stdout.strip().splitlines():
                if line.startswith("Name="):
                    info["brand"] = line.split("=", 1)[1].strip()
                    break
        except Exception:
            pass

    # Try py-cpuinfo for exact flags
    try:
        import cpuinfo  # type: ignore
        ci = cpuinfo.get_cpu_info()
        fl = [x.lower() for x in ci.get("flags", [])]
        info.update(
            has_avx="avx" in fl,
            has_avx2="avx2" in fl,
            has_f16c="f16c" in fl,
            has_fma="fma" in fl,
            has_sse4_2="sse4_2" in fl,
            has_avx512=any("avx512" in x for x in fl),
        )
        if ci.get("brand_raw"):
            info["brand"] = ci["brand_raw"]
    except ImportError:
        # Fallback: infer from CPU name
        n = info["brand"].lower()
        is_amd = any(k in n for k in ("amd", "ryzen", "epyc", "threadripper"))
        is_intel = any(k in n for k in ("intel", "core", "xeon"))
        if is_amd:
            info["vendor"] = "AMD"
            info.update(has_avx=True, has_avx2=True, has_f16c=True,
                        has_fma=True, has_sse4_2=True)
        elif is_intel:
            info["vendor"] = "Intel"
            info.update(has_avx=True, has_sse4_2=True)
            if any(k in n for k in ("haswell", "broadwell", "skylake",
                                    "kaby", "coffee", "comet", "ice",
                                    "tiger", "alder", "raptor", "arrow",
                                    "meteor", "ultra")):
                info.update(has_avx2=True, has_f16c=True, has_fma=True)
        elif arch == "x86_64":
            info.update(has_avx=True, has_avx2=True, has_f16c=True,
                        has_fma=True, has_sse4_2=True)

    # AOCL
    for p in (os.environ.get("AOCL_ROOT", ""), os.environ.get("AOCL_PATH", ""),
              r"C:\Program Files\AMD\AOCL", r"C:\AOCL"):
        if p and Path(p).exists():
            info["has_aocl"] = True
            break

    # Vendor from brand if not set
    if info["vendor"] == "unknown":
        n = info["brand"].lower()
        if any(k in n for k in ("amd", "ryzen", "epyc")):
            info["vendor"] = "AMD"
        elif "intel" in n:
            info["vendor"] = "Intel"

    # cmake flags
    flags = []
    for feat, flag in (("has_avx", "GGML_AVX=ON"), ("has_avx2", "GGML_AVX2=ON"),
                       ("has_f16c", "GGML_F16C=ON"), ("has_fma", "GGML_FMA=ON"),
                       ("has_avx512", "GGML_AVX512=ON")):
        if info[feat]:
            flags.append(flag)
    info["cmake_flags"] = flags

    return info


def _cpu_arch() -> str:
    m = platform.machine().lower()
    if m in ("amd64", "x86_64"):
        return "x86_64"
    if m in ("i386", "i686", "x86"):
        return "x86"
    if "aarch64" in m:
        return "aarch64"
    return m


# ---------------------------------------------------------------------------
# Vulkan / GPU detection
# ---------------------------------------------------------------------------
def _parse_vk_devices_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse GPU devices from full vulkaninfo text output.
    Lines look like: "GPU id = 0 (NVIDIA GeForce GTX 1060 3GB)"
    Captures full name even when it contains parentheses.
    """
    import re
    devices = []
    # Pattern: GPU id = digits, space, (, any characters, ) at end of line
    pattern = re.compile(r"GPU id = (\d+)\s*\((.*)\)$")
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            idx = int(m.group(1))
            name = m.group(2).strip()
            if not any(d["index"] == idx for d in devices):
                devices.append({"index": idx, "name": name, "type": ""})
    # If the above fails, fallback to block parsing (GPU0: ... deviceName = ...)
    if not devices:
        devices = _parse_vk_devices_from_blocks(text)
    return devices

def _parse_vk_devices_from_blocks(text: str) -> List[Dict[str, Any]]:
    """Fallback: parse from GPU0: / GPU1: blocks."""
    import re
    devices = []
    blocks = re.split(r'\nGPU(\d+):\n', text)
    for i in range(1, len(blocks), 2):
        idx = int(blocks[i])
        block = blocks[i+1]
        name_match = re.search(r'deviceName\s*=\s*([^\n]+)', block)
        name = name_match.group(1).strip() if name_match else f"GPU{idx}"
        type_match = re.search(r'deviceType\s*=\s*([^\n]+)', block)
        dev_type = type_match.group(1).strip() if type_match else ""
        devices.append({"index": idx, "name": name, "type": dev_type})
    return devices

def detect_vulkan() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "version": "unknown",
        "sdk": os.environ.get("VULKAN_SDK", ""),
        "devices": [],        # list of {"index": int, "name": str, "type": str}
    }

    vi = shutil.which("vulkaninfo")
    if not vi:
        return result

    # Try JSON output first (most reliable)
    try:
        proc = subprocess.run([vi, "--json"], capture_output=True, text=True, timeout=30)
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            devices = []
            if "physicalDevices" in data:
                dev_list = data["physicalDevices"]
            elif isinstance(data, list):
                dev_list = data
            else:
                dev_list = []
                for v in data.values():
                    if isinstance(v, list) and v and "deviceName" in v[0]:
                        dev_list = v
                        break
            for i, d in enumerate(dev_list):
                if isinstance(d, dict):
                    idx = d.get("deviceID", d.get("deviceId", i))
                    name = d.get("deviceName", f"GPU{idx}")
                    dev_type = d.get("deviceType", "")
                    devices.append({"index": idx, "name": name, "type": dev_type})
            if devices:
                result["available"] = True
                result["devices"] = devices
                result["version"] = _parse_vk_version("")
                return result
    except Exception:
        pass

    # Fallback: parse full text output
    try:
        proc = subprocess.run([vi], capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            result["available"] = True
            result["devices"] = _parse_vk_devices_from_text(proc.stdout)
            result["version"] = _parse_vk_version(proc.stdout)
    except Exception:
        pass

    # Last resort: check if DLL exists (Windows)
    if not result["available"] and platform.system() == "Windows":
        try:
            ctypes.windll.LoadLibrary("vulkan-1.dll")
            result["available"] = True
            result["version"] = "1.x"
        except Exception:
            pass

    return result


def _parse_vk_version(stdout: str) -> str:
    for line in stdout.splitlines():
        for tok in line.split():
            if tok.startswith("1.") and len(tok) >= 3:
                return tok
    return "detected"

def _parse_vk_devices(stdout: str) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("GPU") and "=" in s:
            if current:
                devices.append(current)
            idx_str = s.split("=")[0].replace("GPU", "").strip()
            try:
                idx = int(idx_str)
            except ValueError:
                idx = len(devices)
            current = {"index": idx, "name": s.split("=", 1)[1].strip(), "type": ""}
        elif current:
            sl = s.lower().replace(" ", "")
            if sl.startswith("devicetype"):
                current["type"] = s.split("=", 1)[-1].strip()
    if current:
        devices.append(current)
    return devices


# ---------------------------------------------------------------------------
# Write constants.ini
# ---------------------------------------------------------------------------

def write_constants(cpu: Dict[str, Any], vk: Dict[str, Any]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = configparser.ConfigParser()

    # Read existing so we don't wipe user edits in other sections
    if _CONST_PATH.exists():
        cfg.read(_CONST_PATH, encoding="utf-8")

    # [cpu]
    if not cfg.has_section("cpu"):
        cfg.add_section("cpu")
    cfg["cpu"]["brand"]           = cpu["brand"]
    cfg["cpu"]["vendor"]          = cpu["vendor"]
    cfg["cpu"]["arch"]            = cpu["arch"]
    cfg["cpu"]["cores_logical"]   = str(cpu["cores_logical"])
    cfg["cpu"]["default_threads"] = str(cpu["default_threads"])
    cfg["cpu"]["has_avx"]         = str(cpu["has_avx"])
    cfg["cpu"]["has_avx2"]        = str(cpu["has_avx2"])
    cfg["cpu"]["has_f16c"]        = str(cpu["has_f16c"])
    cfg["cpu"]["has_fma"]         = str(cpu["has_fma"])
    cfg["cpu"]["has_avx512"]      = str(cpu["has_avx512"])
    cfg["cpu"]["has_sse4_2"]      = str(cpu["has_sse4_2"])
    cfg["cpu"]["has_aocl"]        = str(cpu["has_aocl"])
    cfg["cpu"]["cmake_flags"]     = " ".join(cpu["cmake_flags"])

    # [vulkan]
    if not cfg.has_section("vulkan"):
        cfg.add_section("vulkan")
    cfg["vulkan"]["available"]    = str(vk["available"])
    cfg["vulkan"]["version"]      = vk["version"]
    cfg["vulkan"]["sdk"]          = vk["sdk"]
    # gpu_count: number of discrete GPUs found
    gpu_indices = [str(d["index"]) for d in vk["devices"]]
    gpu_names   = [d["name"] for d in vk["devices"]]
    cfg["vulkan"]["gpu_count"]    = str(len(vk["devices"]))
    cfg["vulkan"]["gpu_numbers"]  = ",".join(gpu_indices)   # e.g. "0,1"
    cfg["vulkan"]["gpu_names"]    = ",".join(gpu_names)     # e.g. "RX 580,RX 470"
    # Per-GPU entries for easy lookup
    for d in vk["devices"]:
        cfg["vulkan"][f"gpu{d['index']}_name"] = d["name"]
        cfg["vulkan"][f"gpu{d['index']}_type"] = d.get("type", "")

    with open(_CONST_PATH, "w", encoding="utf-8") as f:
        cfg.write(f)
    log(f"constants.ini written → {_CONST_PATH}")


# ---------------------------------------------------------------------------
# Write default persistent.json (only if missing)
# ---------------------------------------------------------------------------

def write_default_persistent(cpu: Dict[str, Any]) -> None:
    if _PERSIST_PATH.exists():
        return  # never overwrite user config
    dt = cpu["default_threads"]
    defaults: Dict[str, Any] = {
        "encoder_model_path": "", "encoder_model_name": "",
        "imagegen_model_path": "", "imagegen_model_name": "",
        "vae_model_path": "", "vae_model_name": "",
        "backend_encoder": "CPU",
        "backend_imagegen": "CPU",
        "encoder_threads": dt,
        "encoder_batch_size": 512,
        "encoder_ctx_size": 4096,
        "encoder_flash_attn": True,
        "encoder_gpu_layers": -1,
        "imagegen_threads": dt,
        "imagegen_width": 512,
        "imagegen_height": 512,
        "imagegen_steps": 4,
        "imagegen_cfg_scale": 1.0,
        "imagegen_seed": -1,
        "imagegen_sampling": "euler_a",
        "imagegen_batch_count": 1,
        "imagegen_clip_skip": 2,
        "vulkan_device": 0,
        "output_format": "png",
        "auto_save": True,
        "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "negative_prompt": "",
        "ui_theme": "Default",
        "first_run": True,
    }
    tmp = _PERSIST_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=4, ensure_ascii=False)
    tmp.replace(_PERSIST_PATH)
    log(f"persistent.json written → {_PERSIST_PATH}")


# ---------------------------------------------------------------------------
# venv helpers
# ---------------------------------------------------------------------------

def _venv_python() -> Path:
    if platform.system() == "Windows":
        return _VENV_DIR / "Scripts" / "python.exe"
    return _VENV_DIR / "bin" / "python"


def _venv_pip() -> Path:
    if platform.system() == "Windows":
        return _VENV_DIR / "Scripts" / "pip.exe"
    return _VENV_DIR / "bin" / "pip"


def create_venv() -> bool:
    if _venv_python().exists():
        log(f"venv already exists at {_VENV_DIR}")
        return True
    log(f"Creating venv at {_VENV_DIR} ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(_VENV_DIR)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        log("venv created OK")
        return True
    except Exception as e:
        log(f"ERROR creating venv: {e}")
        return False


def install_deps() -> bool:
    vpy = _venv_python()
    if not vpy.exists():
        log("ERROR: venv python not found — run venv creation first")
        return False
    log("Upgrading pip inside venv...")
    try:
        subprocess.run(
            [str(vpy), "-m", "pip", "install", "--upgrade", "pip"],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        log(f"pip upgrade warning: {e}")

    all_ok = True
    for req in REQUIREMENTS:
        log(f"  Installing {req} ...")
        try:
            subprocess.run(
                [str(vpy), "-m", "pip", "install", req],
                check=True, capture_output=True, text=True, timeout=300,
            )
            log(f"  {req} OK")
        except subprocess.CalledProcessError as e:
            log(f"  FAILED: {req}")
            log(f"    {e.stderr[-300:] if e.stderr else ''}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Build tools detection (used only for banner display and cmake wheel check)
# ---------------------------------------------------------------------------

def _find_cmake_in_vs_installations() -> Optional[Path]:
    """Return the cmake.exe bin directory from a VS / Build Tools install, or None."""
    prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    prog_files     = os.environ.get("ProgramFiles",       r"C:\Program Files")
    install_roots: List[str] = []

    vswhere_exe = Path(prog_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere_exe.exists():
        try:
            result = subprocess.run(
                [str(vswhere_exe), "-all", "-prerelease", "-property", "installationPath"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                install_roots = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        except Exception:
            pass

    for base in (prog_files_x86, prog_files):
        for year in ("2022", "2019"):
            for edition in ("BuildTools", "Enterprise", "Professional", "Community", "Preview"):
                candidate = os.path.join(base, "Microsoft Visual Studio", year, edition)
                if os.path.isdir(candidate) and candidate not in install_roots:
                    install_roots.append(candidate)

    for root in install_roots:
        cmake_bin = os.path.join(root, "Common7", "IDE", "CommonExtensions",
                                 "Microsoft", "CMake", "CMake", "bin")
        cmake_exe = os.path.join(cmake_bin, "cmake.exe")
        if os.path.isfile(cmake_exe):
            return Path(cmake_bin)
    return None


def find_cmake() -> Optional[Path]:
    c = shutil.which("cmake")
    if c:
        return Path(c)
    cmake_bin_dir = _find_cmake_in_vs_installations()
    if cmake_bin_dir:
        os.environ["PATH"] = str(cmake_bin_dir) + os.pathsep + os.environ.get("PATH", "")
        return cmake_bin_dir / "cmake.exe"
    for p in (r"C:\Program Files\CMake\bin\cmake.exe",
              r"C:\Program Files (x86)\CMake\bin\cmake.exe"):
        if Path(p).exists():
            return Path(p)
    return None


def find_git() -> Optional[Path]:
    g = shutil.which("git")
    return Path(g) if g else None


# ---------------------------------------------------------------------------
# pip install with live output + inactivity watchdog
# (adapted from reference installer; used for long C++ source builds)
# ---------------------------------------------------------------------------

def _pip_install_watched(pip_exe: Path, args: List[str],
                         max_retries: int = BUILD_MAX_RETRIES,
                         initial_delay: float = BUILD_RETRY_DELAY) -> bool:
    """Run pip install <args> with streaming output and an inactivity watchdog.
    Retries up to max_retries times with exponential backoff.
    Returns True on success.
    """
    _PROGRESS_KW = ("downloading", "installing", "collected", "building",
                    "running", "error", "warning", "failed", "%", "->")
    _SUPPRESS_KW = ("pip's dependency resolver",)

    delay = float(initial_delay)
    cmd   = [str(pip_exe), "install"] + args

    for attempt in range(1, max_retries + 1):
        all_output: List[str] = []
        last_activity         = [time.time()]
        reader_done           = [False]
        stall_reason: List[Optional[str]] = [None]

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)

            def _read():
                try:
                    for raw in proc.stdout:
                        line = raw.rstrip()
                        if not line:
                            continue
                        if any(kw in line.lower() for kw in _SUPPRESS_KW):
                            continue
                        last_activity[0] = time.time()
                        all_output.append(line)
                        if any(kw in line.lower() for kw in _PROGRESS_KW):
                            log(f"  {line}")
                finally:
                    reader_done[0] = True

            t = threading.Thread(target=_read, daemon=True)
            t.start()

            while not reader_done[0]:
                time.sleep(2)
                idle = time.time() - last_activity[0]
                if idle >= BUILD_INACTIVITY_TIMEOUT:
                    stall_reason[0] = f"No output for {idle:.0f}s — stalled"
                    proc.kill()
                    break

            t.join(timeout=5)
            proc.wait()

            combined = "\n".join(all_output).lower()
            if proc.returncode == 0 or "already satisfied" in combined:
                return True

            reason = stall_reason[0]
            if not reason:
                errs = [l for l in all_output if "error" in l.lower()]
                reason = errs[-1][:120] if errs else f"exit code {proc.returncode}"

            if attempt < max_retries:
                log(f"  Attempt {attempt}/{max_retries} failed: {reason}")
                log(f"  Retrying in {delay:.0f}s ...")
                time.sleep(delay)
                delay = min(delay * 2, 300)

        except Exception as e:
            if attempt < max_retries:
                log(f"  Unexpected error: {e}  — retrying in {delay:.0f}s ...")
                time.sleep(delay)
                delay = min(delay * 2, 300)

    return False


# ---------------------------------------------------------------------------
# Backend installation — pip wheels
# ---------------------------------------------------------------------------

def _py_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def install_llama_cpp_python(cpu: Dict[str, Any], use_vulkan: bool) -> str:
    """Install llama-cpp-python.

    use_vulkan=True  → pre-built Vulkan wheel from abetlen index
    use_vulkan=False → pre-built CPU wheel (eswarthammana v0.3.16);
                       falls back to source compile with CPU flags only
                       if the pre-built wheel is unavailable for this Python version.

    Returns "success (...)" or an error string.
    """
    pip = _venv_pip()
    if not pip.exists():
        return "error: venv pip not found"

    if use_vulkan:
        log("  llama-cpp-python: installing Vulkan pre-built wheel ...")
        ok = _pip_install_watched(pip, [
            "llama-cpp-python",
            "--prefer-binary",
            "--extra-index-url", LLAMA_CPP_VULKAN_INDEX,
            "--force-reinstall",
            "--no-cache-dir",
        ])
        if ok:
            return "success (Vulkan wheel)"
        return "error: Vulkan wheel install failed"

    # CPU path — try pre-built wheel first
    ver    = LLAMA_CPP_CPU_VERSION
    pytag  = _py_tag()
    whl_url = LLAMA_CPP_CPU_WHEEL_BASE.format(ver=ver, pytag=pytag)
    log(f"  llama-cpp-python: installing CPU pre-built wheel v{ver} ...")
    ok = _pip_install_watched(pip, [
        whl_url,
        "--force-reinstall",
        "--no-cache-dir",
    ])
    if ok:
        return "success (CPU wheel)"

    # Pre-built wheel unavailable for this Python version — compile from source
    log("  Pre-built wheel unavailable — compiling from source with CPU flags ...")
    cmake_args_str = " ".join(f"-D{f}" for f in cpu["cmake_flags"])
    env_patch: Dict[str, str] = {"FORCE_CMAKE": "1"}
    if cmake_args_str:
        env_patch["CMAKE_ARGS"] = cmake_args_str
    old_env = os.environ.copy()
    os.environ.update(env_patch)
    try:
        ok = _pip_install_watched(pip, [
            "llama-cpp-python",
            "--no-binary", "llama-cpp-python",
            "--no-cache-dir",
            "--force-reinstall",
        ])
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    return "success (compiled CPU)" if ok else "error: all strategies failed"


def install_sd_cpp_python(cpu: Dict[str, Any], use_vulkan: bool) -> str:
    """Install stable-diffusion-cpp-python from source.

    No pre-built binary wheel exists on PyPI — this package always builds
    from source. CMAKE_ARGS controls Vulkan and CPU optimisation flags.

    use_vulkan=True  → SD_VULKAN=ON + CPU flags
    use_vulkan=False → CPU flags only

    Returns "success (...)" or an error string.
    """
    pip = _venv_pip()
    if not pip.exists():
        return "error: venv pip not found"

    # Pre-install cmake binary wheel so the build backend can find cmake
    log("  Pre-installing cmake wheel for build toolchain ...")
    _pip_install_watched(pip, ["cmake", "--only-binary=cmake", "--upgrade",
                               "--no-cache-dir"])

    # Build CMAKE_ARGS from cpu flags + optional Vulkan flag
    flags: List[str] = list(cpu["cmake_flags"])
    if use_vulkan:
        flags.append("SD_VULKAN=ON")
    cmake_args_str = " ".join(f"-D{f}" for f in flags)

    env_patch: Dict[str, str] = {"FORCE_CMAKE": "1"}
    if cmake_args_str:
        env_patch["CMAKE_ARGS"] = cmake_args_str
        log(f"  CMAKE_ARGS: {cmake_args_str}")

    mode = "Vulkan" if use_vulkan else "CPU"
    log(f"  stable-diffusion-cpp-python: compiling from source ({mode}) ...")

    old_env = os.environ.copy()
    os.environ.update(env_patch)
    try:
        ok = _pip_install_watched(pip, [
            SD_CPP_PACKAGE,
            "--no-binary", "stable-diffusion-cpp-python",
            "--no-cache-dir",
            "--force-reinstall",
        ], max_retries=BUILD_MAX_RETRIES, initial_delay=BUILD_RETRY_DELAY)
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    return f"success (compiled {mode})" if ok else f"error: compile failed ({mode})"


# ---------------------------------------------------------------------------
# Install menu helpers
# ---------------------------------------------------------------------------

def _detect_build_tools() -> Tuple[Optional[Path], Optional[Path]]:
    """Return (git, cmake) for banner display. cmake needed for source builds."""
    return find_git(), find_cmake()


def _print_install_banner(cpu: Dict[str, Any], vk: Dict[str, Any]) -> None:
    git, cmake = _detect_build_tools()
    header("Image-Generator-Gguf — Install Method")
    print()
    print()
    print("  System Detections...")
    print(f"     Platform: Windows {platform.version().split('.')[0] if platform.system() == 'Windows' else platform.system()};"
          f" Python {platform.python_version()}")
    print(f"     Build Tools: Git {'OK' if git else 'NOT FOUND'};"
          f" CMake {'OK' if cmake else 'NOT FOUND'}")
    print(f"     Architecture: AVX {cpu['has_avx']};"
          f" AVX2 {cpu['has_avx2']};"
          f" F16C {cpu['has_f16c']};"
          f" FMA {cpu['has_fma']}")
    gpu_str = ", ".join(str(d["index"]) for d in vk["devices"]) if vk["devices"] else "none"
    print(f"     Hardware: CPUs {cpu['cores_logical']};"
          f" GPUs {gpu_str};"
          f" Vulkan {vk['version']}")
    print()
    print()
    print("  " + "-" * 79)
    print()
    print()
    print()
    print("     1. Clean Install (Purge First)")
    print()
    print("     2. Check/Install (Fix Missing Packages/Libraries)")
    print()
    print("     3. Refresh Configs (Only Remake Ini/Json)")
    print()
    print()
    print()
    print("  " + "=" * 79)


def _purge_for_clean_install() -> None:
    """Remove venv and build dirs so everything is rebuilt from scratch."""
    section("Purging previous installation...")
    for target, label in ((_VENV_DIR, "venv"),):
        if target.exists():
            log(f"Removing {label} at {target} ...")
            if _safe_rmtree(target):
                log(f"{label} removed.")
            else:
                log(f"WARNING: could not fully remove {label} at {target}.")
                log(f"  Close any programs using it (antivirus, explorer) and retry.")
        else:
            log(f"{label} not present, skipping.")
    if _PERSIST_PATH.exists():
        _PERSIST_PATH.unlink()
        log("persistent.json removed (will be regenerated).")


def _run_deps(cpu: Dict[str, Any]) -> None:
    """Create venv + install Python deps."""
    section("Python virtual environment...")
    if not create_venv():
        log("FATAL: could not create venv.")
        return

    section("Python dependencies...")
    if not install_deps():
        log("WARNING: some packages failed — the app may not work correctly.")
    else:
        log("All packages installed OK.")


def _print_backend_banner(vk: Dict[str, Any]) -> None:
    vk_label = "detected" if vk["available"] else "not detected"
    header("Image-Generator-Gguf — Backend Selection")
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print(f"     1. Compile for CPU")
    print()
    print(f"     2. Compile for Vulkan  ({vk_label})")
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print("  " + "=" * 79)


def _choose_backend(vk: Dict[str, Any]) -> bool:
    """Show backend menu, return True for Vulkan, False for CPU."""
    while True:
        _print_backend_banner(vk)
        choice = input("  Selection; Menu Options = 1-2, Abandon Install = A: ").strip().upper()
        if choice == "A":
            print()
            print("  Abandoning install — returning to batch menu.")
            print()
            raise SystemExit(0)
        if choice == "1":
            return False
        if choice == "2":
            return True
        print()
        print("  Invalid selection, please try again.")
        print()


def _run_build(cpu: Dict[str, Any], use_vulkan: bool) -> None:
    """Install llama-cpp-python and stable-diffusion-cpp-python via pip."""
    mode = "Vulkan" if use_vulkan else "CPU"
    section(f"Backend install  ({mode})  —  llama-cpp-python + stable-diffusion-cpp-python...")

    log("llama-cpp-python ...")
    llama_status = install_llama_cpp_python(cpu, use_vulkan)
    log(f"  llama-cpp-python  →  {llama_status}")

    log()
    log("stable-diffusion-cpp-python ...")
    sd_status = install_sd_cpp_python(cpu, use_vulkan)
    log(f"  stable-diffusion-cpp-python  →  {sd_status}")


def _run_summary(t0: float) -> None:
    elapsed = round(time.time() - t0, 1)
    section("Installation summary")
    log(f"Time elapsed : {elapsed}s")
    log(f"constants.ini: {_CONST_PATH}")
    log(f"persistent   : {_PERSIST_PATH}")
    log(f"venv         : {_VENV_DIR}")
    log()
    log("Press Enter to return to the batch menu...")
    input()


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_detection() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    section("Hardware detection...")
    cpu = detect_cpu()
    vk  = detect_vulkan()

    log(f"CPU  : {cpu['brand']}")
    log(f"Arch : {cpu['arch']}  Vendor: {cpu['vendor']}")
    log(f"Cores: {cpu['cores_logical']} logical  →  {cpu['default_threads']} threads (85%)")
    log(f"AVX  : {cpu['has_avx']}  AVX2: {cpu['has_avx2']}  F16C: {cpu['has_f16c']}  FMA: {cpu['has_fma']}")
    log()
    log(f"Vulkan : {vk['available']}  ver={vk['version']}")
    log(f"SDK    : {vk['sdk'] or 'not set'}")
    if vk["devices"]:
        log("GPUs :")
        for d in vk["devices"]:
            log(f"  GPU{d['index']}: {d['name']}  ({d.get('type','')})")
    else:
        log("GPUs   : none detected via vulkaninfo")
    return cpu, vk


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Generator GGUF Installer")
    parser.add_argument("--detect-only",  action="store_true",
                        help="Detect hardware and write constants.ini only")
    parser.add_argument("--deps-only",    action="store_true",
                        help="Create venv and install Python packages only")
    parser.add_argument("--build-only",   action="store_true",
                        help="Build llama.cpp and sd.cpp only")
    args = parser.parse_args()

    ensure_dirs()
    header("Image-Generator-Gguf — Initialize Install")
    if args.detect_only:
        cpu, vk = run_detection()
        write_constants(cpu, vk)
        write_default_persistent(cpu)
        log("Detection complete.")
        return

    if args.deps_only:
        cpu, vk = run_detection()
        write_constants(cpu, vk)
        write_default_persistent(cpu)
        t0 = time.time()
        _run_deps(cpu)
        _run_summary(t0)
        return

    if args.build_only:
        cpu, vk = run_detection()
        write_constants(cpu, vk)
        t0 = time.time()
        use_vulkan = _choose_backend(vk)
        _run_build(cpu, use_vulkan)
        _run_summary(t0)
        return

    # Interactive menu
    cpu, vk = run_detection()

    while True:
        _print_install_banner(cpu, vk)
        choice = input("  Selection; Menu Options = 1-3, Abandon Install = A: ").strip().upper()

        if choice == "A":
            print()
            print("  Abandoning install — returning to batch menu.")
            print()
            return

        if choice == "1":
            t0 = time.time()
            use_vulkan = _choose_backend(vk)
            header("Image-Generator-Gguf — Installation")
            _purge_for_clean_install()
            write_constants(cpu, vk)
            write_default_persistent(cpu)
            _run_deps(cpu)
            _run_build(cpu, use_vulkan)
            _run_summary(t0)
            return

        if choice == "2":
            t0 = time.time()
            use_vulkan = _choose_backend(vk)
            header("Image-Generator-Gguf — Installation")
            write_constants(cpu, vk)
            write_default_persistent(cpu)
            _run_deps(cpu)
            _run_build(cpu, use_vulkan)
            _run_summary(t0)
            return

        if choice == "3":
            t0 = time.time()
            section("Refreshing configs...")
            write_constants(cpu, vk)
            if _PERSIST_PATH.exists():
                _PERSIST_PATH.unlink()
                log("persistent.json removed (will be regenerated with defaults).")
            write_default_persistent(cpu)
            _run_summary(t0)
            return

        # Invalid input — redisplay menu
        print()
        print("  Invalid selection, please try again.")
        print()


if __name__ == "__main__":
    main()