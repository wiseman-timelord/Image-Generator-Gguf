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
    "gradio>=6.19",
    "Pillow>=10.0",
    "numpy>=1.26",
]

# ---------------------------------------------------------------------------
# Centralized CPU Features Map
# Maps internal key, display name, and ggml CMake flag.
# Used globally for detection, logging, config writing, and building.
# ---------------------------------------------------------------------------
CPU_FEATURES = [
    {"key": "has_sse",      "name": "SSE",    "cmake": "GGML_SSE=ON"},
    {"key": "has_sse2",     "name": "SSE2",   "cmake": "GGML_SSE2=ON"},
    {"key": "has_sse3",     "name": "SSE3",   "cmake": "GGML_SSE3=ON"},
    {"key": "has_ssse3",    "name": "SSSE3",  "cmake": "GGML_SSSE3=ON"},
    {"key": "has_sse4_1",   "name": "SSE4.1", "cmake": "GGML_SSE4_1=ON"},
    {"key": "has_sse4_2",   "name": "SSE4.2", "cmake": "GGML_SSE4_2=ON"},
    {"key": "has_avx",      "name": "AVX",    "cmake": "GGML_AVX=ON"},
    {"key": "has_avx2",     "name": "AVX2",   "cmake": "GGML_AVX2=ON"},
    {"key": "has_f16c",     "name": "F16C",   "cmake": "GGML_F16C=ON"},
    {"key": "has_fma",      "name": "FMA",    "cmake": "GGML_FMA=ON"},
    {"key": "has_avx512",   "name": "AVX512", "cmake": "GGML_AVX512=ON"},
]

# ---------------------------------------------------------------------------
# Backend source build constants
# ---------------------------------------------------------------------------
GIT_CLONE_RETRIES = 3
GIT_CLONE_TIMEOUT = 900
GIT_RETRY_DELAYS = [5, 10, 15]  # seconds between attempts

LLAMA_CPP_SOURCE_URL = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_SOURCE_DIR = "lc_src"   # Short name keeps paths under MAX_PATH for MSBuild FileTracker

SD_CPP_SOURCE_URL = "https://github.com/leejet/stable-diffusion.cpp.git"
SD_CPP_SOURCE_DIR = "sd_src"      # Short name keeps paths under MAX_PATH for MSBuild FileTracker

LLAMA_BIN_DIR = "data/llama_cpp_binaries"
SD_BIN_DIR    = "data/stable_diffusion_binaries"

BUILD_MAX_RETRIES        = 2
BUILD_RETRY_DELAY        = 15
BUILD_INACTIVITY_TIMEOUT = 900

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
    print()
    print(f"  {title}")
    print("  " + "-" * len(title))

def _safe_rmtree(path: Path) -> bool:
    def _on_error(func, fpath, exc_info):
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
    brand = platform.processor() or "unknown"
    vendor = "unknown"
    
    info: Dict[str, Any] = {
        "arch": "x86_64",
        "brand": brand,
        "vendor": vendor,
        "cores_logical": logical,
        "default_threads": default_threads,
        "has_aocl": False,
    }
    
    for feat in CPU_FEATURES:
        info[feat["key"]] = False

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

    try:
        import cpuinfo  # type: ignore
        ci = cpuinfo.get_cpu_info()
        fl = [x.lower() for x in ci.get("flags", [])]
        
        for feat in CPU_FEATURES:
            flag_name = feat["key"].replace("has_", "")
            if flag_name == "avx512":
                info[feat["key"]] = "avx512f" in fl or "avx512" in fl
            else:
                info[feat["key"]] = flag_name in fl
                
        if ci.get("brand_raw"):
            info["brand"] = ci["brand_raw"]
    except ImportError:
        n = info["brand"].lower()
        is_amd = any(k in n for k in ("amd", "ryzen", "epyc", "threadripper"))
        is_intel = any(k in n for k in ("intel", "core", "xeon", "pentium", "celeron"))
        
        if is_amd:
            info["vendor"] = "AMD"
            info.update(has_sse=True, has_sse2=True, has_sse3=True, has_ssse3=True,
                        has_sse4_1=True, has_sse4_2=True, has_avx=True, has_avx2=True,
                        has_f16c=True, has_fma=True)
            if any(k in n for k in ("ryzen 7", "ryzen 9", "ryzen threadripper 7", "epyc 9", "epyc 8", "zen 4", "zen 5")):
                info["has_avx512"] = True
        elif is_intel:
            info["vendor"] = "Intel"
            info.update(has_sse=True, has_sse2=True, has_sse3=True, has_ssse3=True,
                        has_sse4_1=True, has_sse4_2=True, has_avx=True)
            if any(k in n for k in ("haswell", "broadwell", "skylake", "kaby", "coffee",
                                    "comet", "ice", "tiger", "alder", "raptor", "meteor",
                                    "arrow", "lunar", "ultra")):
                info.update(has_avx2=True, has_f16c=True, has_fma=True)
            if any(k in n for k in ("skylake-x", "cascade lake", "ice lake", "tiger lake",
                                    "sapphire rapids", "emerald rapids", "granite rapids")):
                info["has_avx512"] = True
        else:
            info.update(has_sse=True, has_sse2=True, has_sse3=True, has_ssse3=True,
                        has_sse4_1=True, has_sse4_2=True, has_avx=True, has_avx2=True,
                        has_f16c=True, has_fma=True)

    for p in (os.environ.get("AOCL_ROOT", ""), os.environ.get("AOCL_PATH", ""),
              r"C:\Program Files\AMD\AOCL", r"C:\AOCL"):
        if p and Path(p).exists():
            info["has_aocl"] = True
            break
            
    if info["vendor"] == "unknown":
        n = info["brand"].lower()
        if any(k in n for k in ("amd", "ryzen", "epyc")):
            info["vendor"] = "AMD"
        elif "intel" in n:
            info["vendor"] = "Intel"

    info["cmake_flags"] = [feat["cmake"] for feat in CPU_FEATURES if info.get(feat["key"])]
    return info

# ---------------------------------------------------------------------------
# Vulkan / GPU detection
# ---------------------------------------------------------------------------
def _parse_vk_devices_from_text(text: str) -> List[Dict[str, Any]]:
    import re
    devices = []
    pattern = re.compile(r"GPU id = (\d+)\s*\((.*)\)$")
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            idx = int(m.group(1))
            name = m.group(2).strip()
            if not any(d["index"] == idx for d in devices):
                devices.append({"index": idx, "name": name, "type": ""})
    if not devices:
        devices = _parse_vk_devices_from_blocks(text)
    return devices

def _parse_vk_devices_from_blocks(text: str) -> List[Dict[str, Any]]:
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
        "devices": [],
    }
    vi = shutil.which("vulkaninfo")
    if not vi:
        return result
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
    try:
        proc = subprocess.run([vi], capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            result["available"] = True
            result["devices"] = _parse_vk_devices_from_text(proc.stdout)
            result["version"] = _parse_vk_version(proc.stdout)
    except Exception:
        pass
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

# ---------------------------------------------------------------------------
# Write constants.ini
# ---------------------------------------------------------------------------
def write_constants(cpu: Dict[str, Any], vk: Dict[str, Any],
                    use_vulkan: Optional[bool] = None) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = configparser.ConfigParser()
    if _CONST_PATH.exists():
        cfg.read(_CONST_PATH, encoding="utf-8")

    # [general] — only written when an install route is explicitly chosen.
    # Detection-only calls pass use_vulkan=None and leave any existing value.
    if use_vulkan is not None:
        if not cfg.has_section("general"):
            cfg.add_section("general")
        cfg["general"]["install_type"] = "vulkan" if use_vulkan else "cpu_only"

    if not cfg.has_section("cpu"):
        cfg.add_section("cpu")
    cfg["cpu"]["brand"]           = cpu["brand"]
    cfg["cpu"]["vendor"]          = cpu["vendor"]
    cfg["cpu"]["arch"]            = cpu["arch"]
    cfg["cpu"]["cores_logical"]   = str(cpu["cores_logical"])
    cfg["cpu"]["default_threads"] = str(cpu["default_threads"])
    cfg["cpu"]["has_aocl"]        = str(cpu["has_aocl"])
    cfg["cpu"]["cmake_flags"]     = " ".join(cpu["cmake_flags"])
    
    for feat in CPU_FEATURES:
        cfg["cpu"][feat["key"]] = str(cpu.get(feat["key"], False))
    
    if not cfg.has_section("vulkan"):
        cfg.add_section("vulkan")
    cfg["vulkan"]["available"]    = str(vk["available"])
    cfg["vulkan"]["version"]      = vk["version"]
    cfg["vulkan"]["sdk"]          = vk["sdk"]
    
    gpu_indices = [str(d["index"]) for d in vk["devices"]]
    gpu_names   = [d["name"] for d in vk["devices"]]
    cfg["vulkan"]["gpu_count"]    = str(len(vk["devices"]))
    cfg["vulkan"]["gpu_numbers"]  = ",".join(gpu_indices)
    cfg["vulkan"]["gpu_names"]    = ",".join(gpu_names)
    
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
        return
    dt = cpu["default_threads"]
    defaults: Dict[str, Any] = {
        "encoder_model_path": "", "encoder_model_name": "",
        "imagegen_model_path": "", "imagegen_model_name": "",
        "vae_model_path": "", "vae_model_name": "",
        "last_model_browse_dir": ".\\models",  # <--- NEW KEY ADDED
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

def create_venv() -> bool:
    if _venv_python().exists():
        log(f"venv already exists at {_VENV_DIR}")
        return True
    log(f"Creating venv at {_VENV_DIR}...")
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
        log(f"  Installing {req}...")
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
# Build tools detection
# ---------------------------------------------------------------------------
def _find_cmake_in_vs_installations() -> Optional[Path]:
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
# Build root — short temp path if project root is too deep for MSBuild
# ---------------------------------------------------------------------------
# MSBuild's FileTracker (tracker.exe) cannot create .tlog files when the full
# path exceeds ~260 chars (Windows MAX_PATH). The ggml vulkan-shaders-gen
# ExternalProject adds ~100 chars of its own nesting on top of the build root,
# so any project root longer than ~150 chars risks FTK1011 errors at link time.
#
# Strategy: if the project root is short enough, build inside data/build as
# normal. If it is too long, redirect the entire build tree to C:\build_temp\igg\
# (short and fixed). After binaries are copied out, the temp tree is cleaned up.
#
# No registry modification is made. The system is not permanently altered.

_BUILD_TEMP_ROOT = Path(r"C:\build_temp\igg")
_PATH_SAFETY_THRESHOLD = 150   # chars; project root longer than this -> use temp


def _get_build_root() -> Tuple[Path, bool]:
    """
    Return (build_root_path, using_temp).
    using_temp=True means the caller must delete build_root after copying binaries.
    """
    root_len = len(str(_ROOT))
    if root_len > _PATH_SAFETY_THRESHOLD:
        log(f"  Project root is {root_len} chars — redirecting build to {_BUILD_TEMP_ROOT}")
        log(f"  (temp build tree will be deleted after binaries are copied)")
        return _BUILD_TEMP_ROOT, True
    return _ROOT / "data" / "build", False


def _cleanup_build_temp(build_root: Path, using_temp: bool) -> None:
    """Remove the temporary build tree if one was used."""
    if not using_temp:
        return
    log(f"  Cleaning up temporary build tree at {build_root} ...")
    if _safe_rmtree(build_root):
        log("  Temp build tree removed.")
    else:
        log(f"  WARNING: could not fully remove temp build tree at {build_root}.")
        log("  You may delete it manually.")

# ---------------------------------------------------------------------------
# Git clone helper with resume capability
# ---------------------------------------------------------------------------
def _git_clone(repo_url: str, dest_dir: Path, ref: str, retries: int = GIT_CLONE_RETRIES) -> bool:
    """Clone or resume a git repository with retries."""
    git_exe = find_git()
    if git_exe is None:
        log("  ERROR: git not found. Cannot clone repository.")
        return False

    # Set global postBuffer to 500 MB to avoid early EOF errors
    try:
        subprocess.run(
            [str(git_exe), "config", "--global", "http.postBuffer", "524288000"],
            capture_output=True, check=False, timeout=10
        )
    except Exception:
        pass

    repo_name = repo_url.split('/')[-1].replace('.git', '')

    for attempt in range(1, retries + 1):
        log(f"  Cloning {repo_name} (ref: {ref}) — attempt {attempt}/{retries}")

        # Check if we have an existing repo that we can resume
        if dest_dir.exists() and (dest_dir / ".git").is_dir():
            log(f"    Found existing repository at {dest_dir}, attempting to resume...")
            try:
                # Fetch all updates (including the ref we need)
                fetch_cmd = [str(git_exe), "-C", str(dest_dir), "fetch", "--all", "--prune"]
                subprocess.run(fetch_cmd, check=True, timeout=GIT_CLONE_TIMEOUT, capture_output=False)
                
                # Checkout the requested ref (branch or tag)
                checkout_cmd = [str(git_exe), "-C", str(dest_dir), "checkout", ref]
                # If ref is a tag, we might need to fetch it explicitly; but fetch --all should get it.
                # Force checkout to discard local changes
                subprocess.run(checkout_cmd, check=True, timeout=60, capture_output=False)
                
                # Verify we have the correct commit
                verify_cmd = [str(git_exe), "-C", str(dest_dir), "rev-parse", "HEAD"]
                verify_proc = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=10)
                if verify_proc.returncode == 0:
                    log("  Repository updated and verified.")
                    return True
                else:
                    log("  Repository verification failed after resume.")
                    # Fall through to fresh clone
            except subprocess.CalledProcessError as e:
                log(f"  Resume failed (exit {e.returncode}), will try fresh clone.")
            except Exception as e:
                log(f"  Resume failed: {e}, will try fresh clone.")
            # If resume fails, we will proceed to fresh clone below.
            # Remove the broken directory before fresh clone.
            log(f"    Removing existing directory for fresh clone: {dest_dir}")
            _safe_rmtree(dest_dir)

        # Fresh clone
        log(f"    Cloning fresh from {repo_url}...")
        cmd = [
            str(git_exe), "clone",
            "--depth", "1",
            "--single-branch",
            "--branch", ref,
            "--recurse-submodules",
            "--shallow-submodules",
            repo_url,
            str(dest_dir)
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                timeout=GIT_CLONE_TIMEOUT
            )
            if proc.returncode == 0:
                # Verify
                verify_cmd = [str(git_exe), "-C", str(dest_dir), "rev-parse", "HEAD"]
                verify_proc = subprocess.run(
                    verify_cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if verify_proc.returncode == 0:
                    log("  Clone complete and verified.")
                    return True
                else:
                    log("  Clone succeeded but repository verification failed.")
                    # Treat as failure and retry
            else:
                log(f"  ERROR cloning (exit code {proc.returncode}).")
        except subprocess.TimeoutExpired:
            log("  ERROR: git clone timed out.")
        except Exception as e:
            log(f"  ERROR cloning: {e}")

        # If this was the last attempt, give up
        if attempt == retries:
            log(f"  All {retries} attempts failed.")
            return False

        # Wait before next retry
        delay = GIT_RETRY_DELAYS[min(attempt - 1, len(GIT_RETRY_DELAYS) - 1)]
        log(f"  Retrying in {delay} seconds...")
        time.sleep(delay)

    return False

def _run_cmake_build(source_dir: Path, build_dir: Path,
                     cmake_defs: List[str], jobs: int) -> bool:
    cmake_exe = find_cmake()
    if not cmake_exe:
        log("  ERROR: cmake not found. Install CMake or Visual Studio Build Tools.")
        return False
        
    build_dir.mkdir(parents=True, exist_ok=True)
    d_args = [f"-D{d}" for d in cmake_defs]
    configure_cmd = [
        str(cmake_exe),
        str(source_dir),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ] + d_args
    
    log(f"  cmake configure: {' '.join(d_args)}")
    try:
        proc = subprocess.run(
            configure_cmd, capture_output=False,
            timeout=300, cwd=str(source_dir),
        )
        if proc.returncode != 0:
            log(f"  cmake configure failed (exit {proc.returncode})")
            return False
    except Exception as e:
        log(f"  cmake configure error: {e}")
        return False
        
    build_cmd = [
        str(cmake_exe), "--build", str(build_dir),
        "--config", "Release",
        "--parallel", str(jobs),
    ]
    log(f"  cmake build (--parallel {jobs})...")
    try:
        proc = subprocess.run(
            build_cmd, capture_output=False,
            timeout=BUILD_INACTIVITY_TIMEOUT,
        )
        if proc.returncode != 0:
            log(f"  cmake build failed (exit {proc.returncode})")
            return False
    except Exception as e:
        log(f"  cmake build error: {e}")
        return False
        
    return True

# ---------------------------------------------------------------------------
# Backend installation — compile from source
# ---------------------------------------------------------------------------
def compile_llama_cpp(cpu: Dict[str, Any], use_vulkan: bool) -> str:
    bin_dir = _ROOT / LLAMA_BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    # If the binary already exists, skip cloning and building entirely.
    if (bin_dir / "llama-cli.exe").exists():
        log("  llama-cli.exe already present, skipping clone and build.")
        return "success (binary already present)"

    # Binary missing – log and proceed with clone/build.
    log("  llama-cli.exe not present, cloning and building...")

    build_root, using_temp = _get_build_root()
    build_root.mkdir(parents=True, exist_ok=True)
    
    src_dir = build_root / "llama_cpp_src" / LLAMA_CPP_SOURCE_DIR
    build_dir = build_root / "lc_bld"
    
    if not _git_clone(LLAMA_CPP_SOURCE_URL, src_dir, "b9670"):
        _cleanup_build_temp(build_root, using_temp)
        return "error: git clone failed"
        
    cmake_defs: List[str] = ["BUILD_SHARED_LIBS=OFF"]
    
    # Add CPU optimizations from central map (applies to BOTH CPU and Vulkan builds)
    for feat in CPU_FEATURES:
        if cpu.get(feat["key"]):
            cmake_defs.append(feat["cmake"])
            
    if use_vulkan:
        cmake_defs.append("GGML_VULKAN=ON")
        
    mode = "Vulkan" if use_vulkan else "CPU"
    log(f"  Compiling llama.cpp ({mode}) — this will take several minutes...")
    jobs = max(1, cpu.get("default_threads", 4))
    if not _run_cmake_build(src_dir, build_dir, cmake_defs, jobs):
        _cleanup_build_temp(build_root, using_temp)
        return f"error: cmake build failed ({mode})"
        
    # llama.cpp outputs llama-cli.exe — search fixed candidates then fall back
    # to a recursive glob in case the layout shifts between releases.
    candidates = [
        build_dir / "bin" / "Release" / "llama-cli.exe",
        build_dir / "bin" / "llama-cli.exe",
        build_dir / "Release" / "llama-cli.exe",
        build_dir / "llama-cli.exe",
    ]
    src_exe: Optional[Path] = next((c for c in candidates if c.exists()), None)
    if src_exe is None:
        found = list(build_dir.rglob("llama-cli.exe"))
        src_exe = found[0] if found else None

    if not src_exe:
        log("  WARNING: llama-cli.exe not found in expected locations after build.")
        log(f"  Searched under: {build_dir}")
        _cleanup_build_temp(build_root, using_temp)
        return "error: binary not found after build"
        
    dest_exe = bin_dir / "llama-cli.exe"
    shutil.copy2(str(src_exe), str(dest_exe))
    
    for dll in src_exe.parent.glob("*.dll"):
        shutil.copy2(str(dll), str(bin_dir / dll.name))
        
    log(f"  llama-cli.exe → {dest_exe}")
    _cleanup_build_temp(build_root, using_temp)
    return f"success (compiled {mode})"

def compile_sd_cpp(cpu: Dict[str, Any], use_vulkan: bool) -> str:
    bin_dir = _ROOT / SD_BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    # If the binary already exists, skip cloning and building entirely.
    # Check for either sd-cli.exe or legacy sd.exe.
    if (bin_dir / "sd-cli.exe").exists() or (bin_dir / "sd.exe").exists():
        log("  sd-cli.exe (or sd.exe) already present, skipping clone and build.")
        return "success (binary already present)"

    # Binary missing – log and proceed with clone/build.
    log("  sd.exe not present, cloning and building...")

    build_root, using_temp = _get_build_root()
    build_root.mkdir(parents=True, exist_ok=True)
    
    src_dir = build_root / "sd_cpp_src" / SD_CPP_SOURCE_DIR
    build_dir = build_root / "sd_bld"
    
    if not _git_clone(SD_CPP_SOURCE_URL, src_dir, "master"):
        _cleanup_build_temp(build_root, using_temp)
        return "error: git clone failed"
        
    cmake_defs: List[str] = ["BUILD_SHARED_LIBS=OFF", "SD_BUILD_EXAMPLES=ON"]
    
    # Add CPU optimizations from central map (applies to BOTH CPU and Vulkan builds)
    for feat in CPU_FEATURES:
        if cpu.get(feat["key"]):
            cmake_defs.append(feat["cmake"])
            
    if use_vulkan:
        cmake_defs.append("SD_VULKAN=ON")
        
    sdk = os.environ.get("VULKAN_SDK", "")
    if sdk:
        cmake_defs.append(f"Vulkan_INCLUDE_DIR={sdk}/Include")
        cmake_defs.append(f"Vulkan_LIBRARY={sdk}/Lib/vulkan-1.lib")
        
    mode = "Vulkan" if use_vulkan else "CPU"
    log(f"  Compiling stable-diffusion.cpp ({mode}) — this will take several minutes...")
    jobs = max(1, cpu.get("default_threads", 4))
    if not _run_cmake_build(src_dir, build_dir, cmake_defs, jobs):
        _cleanup_build_temp(build_root, using_temp)
        return f"error: cmake build failed ({mode})"

    # stable-diffusion.cpp (post-refactor) outputs sd-cli.exe and sd-server.exe
    # under bin/Release/. Search fixed candidates then fall back to rglob so the
    # installer remains robust if the layout shifts between releases.
    cli_candidates = [
        build_dir / "bin" / "Release" / "sd-cli.exe",
        build_dir / "bin" / "sd-cli.exe",
        build_dir / "Release" / "sd-cli.exe",
        build_dir / "examples" / "cli" / "Release" / "sd-cli.exe",
        # legacy name kept as final fallback
        build_dir / "bin" / "Release" / "sd.exe",
        build_dir / "bin" / "sd.exe",
        build_dir / "Release" / "sd.exe",
        build_dir / "examples" / "cli" / "Release" / "sd.exe",
        build_dir / "sd.exe",
    ]
    src_exe: Optional[Path] = next((c for c in cli_candidates if c.exists()), None)
    if src_exe is None:
        # Recursive search: prefer sd-cli.exe, fall back to sd.exe
        for name in ("sd-cli.exe", "sd.exe"):
            found = list(build_dir.rglob(name))
            if found:
                src_exe = found[0]
                break

    if not src_exe:
        log("  WARNING: sd-cli.exe (or sd.exe) not found after build.")
        log(f"  Searched under: {build_dir}")
        _cleanup_build_temp(build_root, using_temp)
        return "error: binary not found after build"

    # Copy the CLI binary using its real name (preserve sd-cli.exe or sd.exe)
    dest_exe = bin_dir / src_exe.name
    shutil.copy2(str(src_exe), str(dest_exe))
    log(f"  {src_exe.name} → {dest_exe}")

    # Copy sd-server.exe if present alongside the CLI binary
    srv_src = src_exe.parent / "sd-server.exe"
    if srv_src.exists():
        srv_dst = bin_dir / "sd-server.exe"
        shutil.copy2(str(srv_src), str(srv_dst))
        log(f"  sd-server.exe → {srv_dst}")

    for dll in src_exe.parent.glob("*.dll"):
        shutil.copy2(str(dll), str(bin_dir / dll.name))

    _cleanup_build_temp(build_root, using_temp)
    return f"success (compiled {mode})"

# ---------------------------------------------------------------------------
# Install menu helpers
# ---------------------------------------------------------------------------
def _detect_build_tools() -> Tuple[Optional[Path], Optional[Path]]:
    return find_git(), find_cmake()

def _print_install_banner(cpu: Dict[str, Any], vk: Dict[str, Any]) -> None:
    git, cmake = _detect_build_tools()
    header("Image-Gradio-Gguf — Install Method")
    print()
    print()
    print("  System Detections...")
    print(f"     Platform: Windows {platform.version().split('.')[0] if platform.system() == 'Windows' else platform.system()};"
          f" Python {platform.python_version()}")
    print(f"     Build Tools: Git {'OK' if git else 'NOT FOUND'};"
          f" CMake {'OK' if cmake else 'NOT FOUND'}")
    
    # Clean comma-separated list of detected architecture features from central map
    arch_features = [feat["name"] for feat in CPU_FEATURES if cpu.get(feat["key"])]
    arch_str = ", ".join(arch_features) if arch_features else "Baseline x86_64"
    print(f"     Architecture: {arch_str}")
    
    gpu_str = ", ".join(str(d["index"]) for d in vk["devices"]) if vk["devices"] else "none"
    print(f"     Hardware: CPUs {cpu['cores_logical']};"
          f" GPUs {gpu_str};"
          f" Vulkan {vk['version']}")
    print()
    print()
    print("  " + "-" * 79)
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
    section("Purging previous installation...")
    targets = [
        (_VENV_DIR,                        "venv"),
        (_ROOT / LLAMA_BIN_DIR,            "llama_cpp_binaries"),
        (_ROOT / SD_BIN_DIR,               "stable_diffusion_binaries"),
        (_ROOT / "data" / "build",         "data/build"),
        (_BUILD_TEMP_ROOT,                 "build_temp (C:\\build_temp\\igg)"),
    ]
    for target, label in targets:
        if target.exists():
            log(f"Removing {label} at {target}...")
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
    header("Image-Gradio-Gguf — Backend Selection")
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
    mode = "Vulkan" if use_vulkan else "CPU"
    section(f"Backend compile  ({mode})  —  llama.cpp + stable-diffusion.cpp...")
    log("llama.cpp...")
    llama_status = compile_llama_cpp(cpu, use_vulkan)
    log(f"  llama.cpp  →  {llama_status}")
    log()
    log("stable-diffusion.cpp...")
    sd_status = compile_sd_cpp(cpu, use_vulkan)
    log(f"  stable-diffusion.cpp  →  {sd_status}")

def _run_summary(t0: float) -> None:
    elapsed = round(time.time() - t0, 1)
    section("Installation summary")
    log(f"Time elapsed : {elapsed}s")
    log(f"constants.ini: {_CONST_PATH}")
    log(f"persistent   : {_PERSIST_PATH}")
    log(f"venv         : {_VENV_DIR}")
    log(f"llama bins   : {_ROOT / LLAMA_BIN_DIR}")
    log(f"sd bins      : {_ROOT / SD_BIN_DIR}")
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
    
    # Clean comma-separated list for detection log from central map
    arch_features = [feat["name"] for feat in CPU_FEATURES if cpu.get(feat["key"])]
    arch_str = ", ".join(arch_features) if arch_features else "Baseline x86_64"
    log(f"Features: {arch_str}")
    
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
    parser.add_argument("--detect-only",  action="store_true", help="Detect hardware and write constants.ini only")
    parser.add_argument("--deps-only",    action="store_true", help="Create venv and install Python packages only")
    parser.add_argument("--build-only",   action="store_true", help="Build llama.cpp and sd.cpp only")
    args = parser.parse_args()
    
    ensure_dirs()
    header("Image-Gradio-Gguf — Initialize Install")
    
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
        t0 = time.time()
        use_vulkan = _choose_backend(vk)
        write_constants(cpu, vk, use_vulkan=use_vulkan)
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
            header("Image-Gradio-Gguf — Installation")
            _purge_for_clean_install()
            write_constants(cpu, vk, use_vulkan=use_vulkan)
            write_default_persistent(cpu)
            _run_deps(cpu)
            _run_build(cpu, use_vulkan)
            _run_summary(t0)
            return
        if choice == "2":
            t0 = time.time()
            use_vulkan = _choose_backend(vk)
            header("Image-Gradio-Gguf — Installation")
            write_constants(cpu, vk, use_vulkan=use_vulkan)
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
            
        print()
        print("  Invalid selection, please try again.")
        print()

if __name__ == "__main__":
    main()