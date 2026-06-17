"""
configure.py - Configuration, global variables, constants, maps, and lists.
All shared constants and config file I/O live here.
Reads hardware info from data/constants.ini (written by installer.py).
"""

from __future__ import annotations

import configparser
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Shared constants / maps / lists
# ---------------------------------------------------------------------------

SAMPLER_MAP: Dict[str, str] = {
    "euler_a": "euler_a", "euler": "euler", "heun": "heun",
    "dpm2": "dpm2", "dpm++2s_a": "dpm++2s_a", "dpm++2m": "dpm++2m",
    "dpm++2mv2": "dpm++2mv2", "lcm": "lcm", "ddim": "ddim", "plms": "plms",
}

IMAGE_SIZES       = [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 1024]
STEP_CHOICES      = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
BATCH_SIZE_CHOICES = [128, 256, 512, 1024, 2048]
CTX_SIZE_CHOICES  = [2048, 4096, 8192, 16384, 32768]
GPU_LAYER_CHOICES = [-1, 0, 10, 20, 30, 40, 50, 99]
CLIP_SKIP_CHOICES = [1, 2, 3]
BATCH_COUNT_CHOICES = [1, 2, 3, 4]
OUTPUT_FORMATS    = ["png", "jpg", "bmp"]

# Key used by the unified bottom status bar shared across all pages.
STATUS_BAR_KEY: str = "status"


# ---------------------------------------------------------------------------
# Runtime state  (transient — not persisted)
# ---------------------------------------------------------------------------

APP_STATE: Dict[str, Any] = {
    "last_image_path": "",
    "last_prompt": "",
    "is_building": False,
    "cancel_requested": False,
}


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Optional[Path] = None


def _get_project_root() -> Path:
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT
    current = Path(__file__).resolve().parent
    _PROJECT_ROOT = current.parent if current.name == "scripts" else current
    return _PROJECT_ROOT


def get_constants_path() -> Path:
    return _get_project_root() / "data" / "constants.ini"


def get_persistent_path() -> Path:
    return _get_project_root() / "data" / "persistent.json"


def get_output_dir() -> Path:
    out = _get_project_root() / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_models_dir() -> Path:
    return _get_project_root() / "models"


def get_build_dir() -> Path:
    return _get_project_root() / "data" / "build"


def get_llama_bin_dir() -> Path:
    return _get_project_root() / "data" / "llama_cpp_binaries"


def get_sd_bin_dir() -> Path:
    return _get_project_root() / "data" / "stable_diffusion_binaries"


def ensure_data_dirs() -> None:
    root = _get_project_root()
    for d in ("data", "output", "models"):
        (root / d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Hardware constants — read from constants.ini (written by installer)
# ---------------------------------------------------------------------------

def _read_constants() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    p = get_constants_path()
    if p.exists():
        cfg.read(p, encoding="utf-8")
    return cfg


def get_default_threads() -> int:
    """Return 85% of logical cores, sourced from constants.ini if available."""
    cfg = _read_constants()
    try:
        return int(cfg["cpu"]["default_threads"])
    except (KeyError, ValueError):
        return max(1, math.ceil((os.cpu_count() or 4) * 0.85))


def get_cpu_info() -> Dict[str, Any]:
    """Return CPU details dict sourced from constants.ini."""
    cfg = _read_constants()
    sec = cfg["cpu"] if cfg.has_section("cpu") else {}
    cores = int(sec.get("cores_logical", str(os.cpu_count() or 4)))
    dt    = int(sec.get("default_threads", str(max(1, math.ceil(cores * 0.85)))))
    return {
        "brand":          sec.get("brand", "unknown"),
        "vendor":         sec.get("vendor", "unknown"),
        "arch":           sec.get("arch", "x86_64"),
        "cores_logical":  cores,
        "default_threads": dt,
        "has_avx":        sec.get("has_avx", "False") == "True",
        "has_avx2":       sec.get("has_avx2", "False") == "True",
        "has_f16c":       sec.get("has_f16c", "False") == "True",
        "has_fma":        sec.get("has_fma", "False") == "True",
        "has_avx512":     sec.get("has_avx512", "False") == "True",
        "has_sse4_2":     sec.get("has_sse4_2", "False") == "True",
        "has_aocl":       sec.get("has_aocl", "False") == "True",
        "cmake_flags":    sec.get("cmake_flags", "").split(),
    }


def get_vulkan_info() -> Dict[str, Any]:
    """Return Vulkan/GPU details sourced from constants.ini."""
    cfg = _read_constants()
    sec = cfg["vulkan"] if cfg.has_section("vulkan") else {}

    available   = sec.get("available", "False") == "True"
    gpu_numbers = sec.get("gpu_numbers", "")      # e.g. "0,1"
    gpu_names   = sec.get("gpu_names", "")        # e.g. "RX 580,RX 470"

    indices = [int(x.strip()) for x in gpu_numbers.split(",") if x.strip().lstrip("-").isdigit()]
    names   = [x.strip() for x in gpu_names.split(",") if x.strip()]

    devices = []
    for i, idx in enumerate(indices):
        devices.append({
            "index": idx,
            "name":  names[i] if i < len(names) else f"GPU{idx}",
        })

    return {
        "available":    available,
        "version":      sec.get("version", "unknown"),
        "sdk":          sec.get("sdk", ""),
        "gpu_count":    int(sec.get("gpu_count", "0")),
        "gpu_numbers":  gpu_numbers,
        "gpu_names":    gpu_names,
        "devices":      devices,
    }


def get_install_type() -> str:
    """Return 'vulkan' or 'cpu_only' from constants.ini [general] section.
    Falls back to 'vulkan' when the key is absent (pre-existing installs that
    have Vulkan available should keep full GPU options).
    """
    cfg = _read_constants()
    if cfg.has_section("general"):
        return cfg["general"].get("install_type", "vulkan").strip().lower()
    # Legacy installs without [general]: infer from vulkan availability.
    vk = cfg["vulkan"] if cfg.has_section("vulkan") else {}
    return "vulkan" if vk.get("available", "False") == "True" else "cpu_only"


def get_backend_choices() -> Dict[str, List[str]]:
    """
    Build the dropdown choices for encoder/imagegen backends.

    Returns {"cpu_choices": [...], "gpu_choices": [...], "all_choices": [...]}
    where each choice is a string the UI shows and inference.py uses.

    CPU entry  : "<CPU brand name>"  (e.g. "AMD Ryzen 9 3900X 12-Core Processor")
    GPU entries: "Vulkan GPU 0 — RX 580", "Vulkan GPU 1 — RX 470", ...
    GPU entries are omitted entirely for a cpu_only install.
    """
    cpu_info = get_cpu_info()
    cpu_label = cpu_info.get("brand", "CPU") or "CPU"
    cpu_choices = [cpu_label]
    gpu_choices: List[str] = []

    if get_install_type() != "cpu_only":
        vk = get_vulkan_info()
        if vk["available"]:
            for d in vk["devices"]:
                label = f"Vulkan GPU {d['index']} — {d['name']}"
                gpu_choices.append(label)

    all_choices = cpu_choices + gpu_choices
    return {
        "cpu_choices": cpu_choices,
        "gpu_choices": gpu_choices,
        "all_choices": all_choices,
    }


def get_thread_choices() -> List[int]:
    """Thread count dropdown choices anchored to the detected default."""
    dt = get_default_threads()
    base = [4, 8, 12, 16, 20, 24, 28, 32, 48, 64]
    choices = sorted(set(base + [dt]))
    return choices


def parse_backend_choice(choice: str) -> Dict[str, Any]:
    """
    Convert a backend dropdown string back to the values inference needs.

    Returns {"use_vulkan": bool, "vulkan_device": int}
    e.g. "Vulkan GPU 1 — RX 470" → {"use_vulkan": True, "vulkan_device": 1}
         "AMD Ryzen 9 3900X ..."  → {"use_vulkan": False, "vulkan_device": -1}
    """
    if choice.startswith("Vulkan GPU"):
        # "Vulkan GPU 1 — name"
        parts = choice.split()
        try:
            idx = int(parts[2])
        except (IndexError, ValueError):
            idx = 0
        return {"use_vulkan": True, "vulkan_device": idx}
    return {"use_vulkan": False, "vulkan_device": -1}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def _default_persistent() -> Dict[str, Any]:
    dt = get_default_threads()
    cpu_label = get_cpu_info().get("brand", "CPU") or "CPU"
    return {
        "encoder_model_path": "", "encoder_model_name": "",
        "imagegen_model_path": "", "imagegen_model_name": "",
        "vae_model_path": "", "vae_model_name": "",
        "backend_encoder": cpu_label,
        "backend_imagegen": cpu_label,
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


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def load_constants() -> configparser.ConfigParser:
    return _read_constants()


def save_constants(config: configparser.ConfigParser) -> None:
    path = get_constants_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        config.write(f)


def load_persistent() -> Dict[str, Any]:
    path = get_persistent_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, IOError):
            pass
    return _default_persistent()


def save_persistent(data: Dict[str, Any]) -> None:
    path = get_persistent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    tmp.replace(path)


def update_persistent(updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_persistent()
    data.update(updates)
    save_persistent(data)
    return data


def resolve_model_path(path_str: str,
                       fallback_dir: Optional[Path] = None) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if p.is_absolute() and p.exists():
        return p
    fb = fallback_dir or get_models_dir()
    if (fb / p).exists():
        return fb / p
    root_rel = _get_project_root() / p
    if root_rel.exists():
        return root_rel
    return None


# ---------------------------------------------------------------------------
# Generation presets
# ---------------------------------------------------------------------------

def get_generation_presets() -> Dict[str, Dict[str, Any]]:
    return {
        "Fast (Turbo)": {
            "imagegen_steps": 4, "imagegen_cfg_scale": 1.0,
            "imagegen_sampling": "euler_a",
            "imagegen_width": 512, "imagegen_height": 512,
        },
        "Balanced": {
            "imagegen_steps": 8, "imagegen_cfg_scale": 1.5,
            "imagegen_sampling": "euler_a",
            "imagegen_width": 512, "imagegen_height": 512,
        },
        "Quality": {
            "imagegen_steps": 20, "imagegen_cfg_scale": 7.0,
            "imagegen_sampling": "dpm++2m",
            "imagegen_width": 768, "imagegen_height": 768,
        },
        "Portrait": {
            "imagegen_steps": 8, "imagegen_cfg_scale": 1.5,
            "imagegen_sampling": "euler_a",
            "imagegen_width": 512, "imagegen_height": 768,
        },
        "Widescreen": {
            "imagegen_steps": 8, "imagegen_cfg_scale": 1.5,
            "imagegen_sampling": "euler_a",
            "imagegen_width": 768, "imagegen_height": 432,
        },
    }