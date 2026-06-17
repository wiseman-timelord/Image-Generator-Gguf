"""
inference.py - Image generation, model handling, and text generation.
Model discovery, probing, prompt enhancement, and image generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import struct
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import scripts.configure as configure
import scripts.utilities as utilities


# ---------------------------------------------------------------------------
# GGUF constants
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
GGUF_TYPES = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16",
    4: "UINT32", 5: "INT32", 6: "FLOAT32", 7: "BOOL",
    8: "STRING", 9: "ARRAY", 10: "UINT64", 11: "INT64",
    12: "FLOAT64",
}


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------

def find_sd_cpp() -> Optional[Path]:
    """
    Find the stable-diffusion.cpp CLI binary.  Search order:
      1. ./data/stable_diffusion_binaries/sd-cli.exe  (current installer output)
      2. ./data/stable_diffusion_binaries/sd.exe       (legacy fallback)
      3. PATH
    """
    bin_dir = configure.get_sd_bin_dir()
    for name in ("sd-cli.exe", "sd-cli", "sd.exe", "sd"):
        p = bin_dir / name
        if p.exists():
            return p
    for name in ("sd-cli", "sd"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def find_llama_cli() -> Optional[Path]:
    """
    Find llama-cli.exe.  Search order:
      1. ./data/llama_cpp_binaries/llama-cli.exe  (compiled by installer)
      2. PATH
    """
    bin_dir = configure.get_llama_bin_dir()
    for name in ("llama-cli.exe", "llama-cli", "main.exe", "main"):
        p = bin_dir / name
        if p.exists():
            return p
    for name in ("llama-cli", "main"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


# ---------------------------------------------------------------------------
# Model probing
# ---------------------------------------------------------------------------

def scan_directory(dir_path: str,
                   extensions: Tuple[str, ...] = (".gguf", ".safetensors")
                   ) -> List[Dict[str, Any]]:
    path = Path(dir_path).expanduser()
    if not path.is_dir():
        return []
    results: List[Dict[str, Any]] = []
    for ext in extensions:
        for fp in path.rglob(f"*{ext}"):
            meta = probe_model(str(fp))
            if meta:
                results.append(meta)
    return results


def probe_model(file_path: str) -> Optional[Dict[str, Any]]:
    p = Path(file_path).expanduser()
    if not p.exists():
        return None
    ext = p.suffix.lower()
    if ext == ".gguf":
        return _probe_gguf(p)
    if ext == ".safetensors":
        return _probe_safetensors(p)
    return None


def find_model_by_name(substring: str, search_dirs: List[str]) -> Optional[str]:
    for d in search_dirs:
        dp = Path(d).expanduser()
        if not dp.is_dir():
            continue
        for fp in dp.rglob("*"):
            if (substring.lower() in fp.name.lower()
                    and fp.suffix.lower() in (".gguf", ".safetensors")):
                return str(fp)
    return None


def get_quantization_label(filename: str) -> str:
    n = filename.lower()
    for q in ("q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1",
              "q4_k_s", "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m",
              "q6_k", "q8_0", "f16", "f32", "iq2_xxs", "iq2_xs",
              "iq3_xxs", "iq3_s", "iq4_xs", "iq4_nl"):
        if q in n:
            return q.upper()
    return "Unknown"


def categorize_models(models: List[Dict[str, Any]]
                      ) -> Dict[str, List[Dict[str, Any]]]:
    cats: Dict[str, List[Dict[str, Any]]] = {
        "encoder": [], "diffusion": [], "vae": [], "unknown": [],
    }
    for m in models:
        name = m.get("filename", "").lower()
        fmt = m.get("format", "")
        if fmt.startswith("Safetensors"):
            if any(k in name for k in ("ae.", "vae", "autoencoder")):
                cats["vae"].append(m)
            else:
                cats["unknown"].append(m)
        elif any(k in name for k in
                 ("qwen", "engineer", "encoder", "vision", "vl", "llava")):
            cats["encoder"].append(m)
        elif any(k in name for k in
                 ("turbo", "diffusion", "unet", "sdxl", "flux", "z_image")):
            cats["diffusion"].append(m)
        else:
            cats["unknown"].append(m)
    return cats


# ---------------------------------------------------------------------------
# GGUF parser
# ---------------------------------------------------------------------------

def _probe_gguf(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return None
            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            meta: Dict[str, Any] = {
                "format": "GGUF", "file": str(p), "filename": p.name,
                "version": version, "tensor_count": tensor_count,
                "kv_count": kv_count,
                "size_bytes": p.stat().st_size,
                "size_human": utilities.human_size(p.stat().st_size),
                "quantization": get_quantization_label(p.name),
                "hash_prefix": utilities.quick_hash(str(p)),
            }
            for _ in range(min(kv_count, 128)):
                try:
                    kv = _read_gguf_kv(f)
                    if kv is None:
                        break
                    key, value = kv
                    short = key.split(".")[-1]
                    if short in ("architecture", "name", "description"):
                        meta[short] = value
                except Exception:
                    break
            return meta
    except Exception:
        return {
            "format": "GGUF (unreadable)", "file": str(p),
            "filename": p.name,
            "size_bytes": p.stat().st_size,
            "size_human": utilities.human_size(p.stat().st_size),
            "quantization": get_quantization_label(p.name),
        }


def _read_gguf_kv(f) -> Optional[Tuple[str, Any]]:
    try:
        key_len = struct.unpack("<Q", f.read(8))[0]
        if key_len > 65536:
            return None
        key = f.read(key_len).decode("utf-8", errors="replace")
        type_idx = struct.unpack("<I", f.read(4))[0]
        value = _read_gguf_value(f, type_idx)
        return key, value
    except Exception:
        return None


def _read_gguf_value(f, type_idx: int) -> Any:
    gt = GGUF_TYPES.get(type_idx, "UNKNOWN")
    if gt == "UINT8":
        return struct.unpack("<B", f.read(1))[0]
    if gt == "INT8":
        return struct.unpack("<b", f.read(1))[0]
    if gt == "UINT16":
        return struct.unpack("<H", f.read(2))[0]
    if gt == "INT16":
        return struct.unpack("<h", f.read(2))[0]
    if gt == "UINT32":
        return struct.unpack("<I", f.read(4))[0]
    if gt == "INT32":
        return struct.unpack("<i", f.read(4))[0]
    if gt == "FLOAT32":
        return struct.unpack("<f", f.read(4))[0]
    if gt == "UINT64":
        return struct.unpack("<Q", f.read(8))[0]
    if gt == "INT64":
        return struct.unpack("<q", f.read(8))[0]
    if gt == "FLOAT64":
        return struct.unpack("<d", f.read(8))[0]
    if gt == "BOOL":
        return struct.unpack("<B", f.read(1))[0] != 0
    if gt == "STRING":
        slen = struct.unpack("<Q", f.read(8))[0]
        if slen > 1_000_000:
            return ""
        return f.read(slen).decode("utf-8", errors="replace")
    if gt == "ARRAY":
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        if arr_len > 10000:
            return []
        vals = []
        for _ in range(arr_len):
            vals.append(_read_gguf_value(f, arr_type))
        return vals
    return None


# ---------------------------------------------------------------------------
# Safetensors parser
# ---------------------------------------------------------------------------

def _probe_safetensors(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len > 100_000_000:
                return None
            header = json.loads(f.read(header_len))
        dtype_counts: Dict[str, int] = {}
        total_params = 0
        for info in header.values():
            if isinstance(info, dict) and "dtype" in info:
                dt = info["dtype"]
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
                for x in info.get("shape", []):
                    total_params = total_params * x if total_params else x
        return {
            "format": "Safetensors", "file": str(p), "filename": p.name,
            "size_bytes": p.stat().st_size,
            "size_human": utilities.human_size(p.stat().st_size),
            "tensor_count": len(header), "dtypes": dtype_counts,
            "total_params": total_params,
            "hash_prefix": utilities.quick_hash(str(p)),
        }
    except Exception:
        return {
            "format": "Safetensors (unreadable)", "file": str(p),
            "filename": p.name,
            "size_bytes": p.stat().st_size,
            "size_human": utilities.human_size(p.stat().st_size),
        }


# ---------------------------------------------------------------------------
# Prompt enhancement
# ---------------------------------------------------------------------------

def enhance_prompt(prompt: str, cfg: Dict[str, Any],
                   progress_callback: Optional[Callable] = None) -> str:
    model_path = cfg.get("encoder_model_path", "")
    if not model_path or not Path(model_path).exists():
        return prompt
    cli = find_llama_cli()
    if not cli:
        return prompt

    template = cfg.get("prompt_template", "{prompt}")
    system_msg = (
        "You are an expert image prompt engineer. Convert the user's "
        "request into a highly detailed, vivid image generation prompt. "
        "Be descriptive about style, lighting, composition, colors, and "
        "atmosphere. Respond ONLY with the prompt text."
    )
    full_prompt = template.replace(
        "{prompt}", f"{system_msg}\n\nUser request: {prompt}")

    args = [
        str(cli), "-m", model_path, "-p", full_prompt,
        "-c", str(cfg.get("encoder_ctx_size", 4096)),
        "-t", str(cfg.get("encoder_threads", configure.get_default_threads())),
        "-b", str(cfg.get("encoder_batch_size", 512)),
        "-n", "256", "--temp", "0.7", "--log-disable",
    ]

    backend = cfg.get("backend_encoder", "Vulkan GPU 1")
    env = os.environ.copy()
    if "Vulkan" in backend:
        args.extend(["-ngl", str(cfg.get("encoder_gpu_layers", 99))])
        env["GGML_VULKAN_DEVICE"] = str(cfg.get("vulkan_device", 1))
    else:
        args.extend(["-ngl", "0"])

    if cfg.get("encoder_flash_attn", True):
        args.append("--flash-attn")

    try:
        if progress_callback:
            progress_callback("Enhancing prompt with LLM...", 0.1)
        result = subprocess.run(args, capture_output=True, text=True,
                                timeout=120, encoding="utf-8",
                                errors="replace", env=env)
        output = result.stdout.strip()
        if output:
            for marker in ("<|im_start|>assistant", "<|start_header_id|>assistant"):
                if marker in output:
                    output = output.split(marker)[-1]
                    break
            for marker in ("<|im_end|>", "<|im_start|>", "<|eot_id|>"):
                output = output.replace(marker, "")
            output = output.strip()
        if output and len(output) > 10:
            if progress_callback:
                progress_callback("Prompt enhanced.", 0.2)
            return output[:2000]
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    return prompt


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_image(prompt: str, cfg: Dict[str, Any],
                   progress_callback: Optional[Callable] = None
                   ) -> Dict[str, Any]:
    t0 = time.time()
    result: Dict[str, Any] = {
        "success": False, "output_path": "", "message": "",
        "seed_used": -1, "elapsed_seconds": 0.0,
    }

    diff_path = cfg.get("imagegen_model_path", "")
    if not diff_path or not Path(diff_path).exists():
        result["message"] = "Image generation model not found. Configure in Configuration tab."
        return result

    vae_path = cfg.get("vae_model_path", "")
    if not vae_path or not Path(vae_path).exists():
        result["message"] = "VAE model (ae.safetensors) not found. Configure in Configuration tab."
        return result

    sd_cli = find_sd_cpp()
    if not sd_cli:
        result["message"] = ("stable-diffusion.cpp not found. "
                             "Run installer or build from Configuration tab.")
        return result

    # Enhance prompt
    enhanced = prompt
    if cfg.get("encoder_model_path") and Path(cfg["encoder_model_path"]).exists():
        try:
            enhanced = enhance_prompt(prompt, cfg, progress_callback)
        except Exception:
            enhanced = prompt

    # Seed
    seed = cfg.get("imagegen_seed", -1)
    if seed is None or int(seed) < 0:
        seed = random.randint(1, 2147483647)
    else:
        seed = int(seed)

    # Output path
    output_dir = configure.get_output_dir()
    fmt = cfg.get("output_format", "png")
    output_name = f"img_{int(time.time())}_s{seed}.{fmt}"
    output_path = output_dir / output_name

    # Build sd command.
    # --diffusion-model for standalone gguf diffusion weights (not -m).
    # Z-Image-Turbo is S3-DiT architecture: it requires --llm pointing at
    # the Qwen3 encoder gguf. Without it sd.cpp cannot find the conditioner
    # tensors (text_encoders.llm.*) and fails at model metadata validation.
    enc_path = cfg.get("encoder_model_path", "")
    args = [
        str(sd_cli),
        "--diffusion-model", str(diff_path),
        "--vae", str(vae_path),
        "-p", enhanced, "-o", str(output_path),
        "-H", str(int(cfg.get("imagegen_height", 512))),
        "-W", str(int(cfg.get("imagegen_width", 512))),
        "--steps", str(int(cfg.get("imagegen_steps", 4))),
        "--cfg-scale", str(float(cfg.get("imagegen_cfg_scale", 1.0))),
        "--seed", str(seed),
        "--sampling-method",
        configure.SAMPLER_MAP.get(cfg.get("imagegen_sampling", "euler_a"),
                                  "euler_a"),
        "-t", str(int(cfg.get("imagegen_threads", configure.get_default_threads()))),
    ]

    # Pass the Qwen3 encoder as the LLM text encoder for Z-Image-Turbo.
    # This is mandatory — the diffusion gguf has no bundled text encoder.
    if enc_path and Path(enc_path).exists():
        args.extend(["--llm", enc_path])

    neg = cfg.get("negative_prompt", "")
    if neg:
        args.extend(["-n", neg])

    backend = cfg.get("backend_imagegen", "CPU")
    env = os.environ.copy()
    if "Vulkan" in backend:
        vk_dev = cfg.get("vulkan_device", 1)
        # --backend replaces --vulkan in current sd.cpp builds.
        # Format: --backend vulkan{N}  (e.g. --backend vulkan1)
        args.extend(["--backend", f"vulkan{vk_dev}"])
        sdk = utilities.detect_vulkan().get("vulkan_sdk", "")
        if sdk:
            env["PATH"] = str(Path(sdk) / "Bin") + ";" + env.get("PATH", "")

    batch = int(cfg.get("imagegen_batch_count", 1))
    if batch > 1:
        args.extend(["-b", str(batch)])

    # --clip-skip is only valid for SD1.x / SD2.x architectures.
    # Flux-based models (z_image_turbo) do not use CLIP skip — omit it.
    diff_name = Path(str(diff_path)).name.lower()
    is_sd_classic = not any(k in diff_name for k in
                            ("flux", "z_image", "sd3", "wan", "ltx"))
    clip_skip = int(cfg.get("imagegen_clip_skip", 2))
    if clip_skip > 1 and is_sd_classic:
        args.extend(["--clip-skip", str(clip_skip)])

    if progress_callback:
        progress_callback("Generating image...", 0.3)

    try:
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        output_lines: List[str] = []
        for line in process.stdout:
            output_lines.append(line)
            if progress_callback and ("step" in line.lower() or "%" in line):
                try:
                    progress_callback(line.strip()[:100], 0.3)
                except Exception:
                    pass
            if configure.APP_STATE.get("cancel_requested"):
                process.kill()
                result["message"] = "Generation cancelled."
                result["elapsed_seconds"] = round(time.time() - t0, 2)
                configure.APP_STATE["cancel_requested"] = False
                return result

        process.wait(timeout=600)
        elapsed = time.time() - t0

        if process.returncode == 0 and output_path.exists():
            result.update(
                success=True, output_path=str(output_path),
                message=f"Saved {output_name} ({elapsed:.1f}s)",
                seed_used=seed,
                elapsed_seconds=round(elapsed, 2),
            )
            if progress_callback:
                progress_callback(f"Done: {output_name}", 1.0)
        else:
            # Full subprocess output -> terminal (Windows console) only.
            full_output = "".join(output_lines)
            print(
                f"\n--- sd.cpp output (exit {process.returncode}) ---\n"
                f"{full_output}"
                f"--- end sd.cpp output ---\n", flush=True)
            result["message"] = (
                f"Generation failed (exit {process.returncode}). "
                f"See terminal for details."
            )
            result["elapsed_seconds"] = round(time.time() - t0, 2)

    except subprocess.TimeoutExpired:
        process.kill()
        print("\n--- sd.cpp timed out after 600 s ---\n", flush=True)
        result["message"] = "Generation timed out (600 s). See terminal."
        result["elapsed_seconds"] = round(time.time() - t0, 2)
    except Exception as e:
        print(f"\n--- generate_image exception ---\n{traceback.format_exc()}"
              f"--- end traceback ---\n", flush=True)
        result["message"] = f"Error: {e}. See terminal for traceback."
        result["elapsed_seconds"] = round(time.time() - t0, 2)

    return result