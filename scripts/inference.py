"""
inference.py - Image generation, model handling, and text generation.
Model discovery, probing, prompt enhancement, and image generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
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

# Used to resolve the -1 sentinel ("all layers") into a concrete count 
# before retry dampening arithmetic. Sourced from configure.py to keep 
# model constraints centralized.
_QWEN3_4B_LAYERS: int = configure.ENCODER_MAX_LAYERS

# OOM fingerprints that appear in llama-cli stdout/stderr output.
_OOM_MARKERS: Tuple[str, ...] = (
    "out of memory",
    "erroroutofdevicememory",
    "alloc.*failed",
    "failed to allocate",
    "ggml_vulkan: device memory allocation",
    "cudamalloc failed",
)


def _is_oom_output(text: str) -> bool:
    """Return True if the combined process output contains an OOM indicator."""
    lower = text.lower().replace(" ", "")
    for marker in _OOM_MARKERS:
        # markers without regex wildcards: simple substring match
        if "*" not in marker:
            if marker.replace(" ", "") in lower:
                return True
        else:
            if re.search(marker, text, re.IGNORECASE):
                return True
    return False


def _resolve_gpu_layers(raw: int, model_path: str) -> int:
    """
    Resolve the -1 sentinel to a concrete layer count.

    -1 means "offload all layers".  We use the known layer count for the
    Qwen3-4B family; for any other model we fall back to 99 (llama.cpp treats
    values larger than the actual layer count as "all layers").
    """
    if raw >= 0:
        return raw
    # -1 sentinel: resolve to model-specific maximum
    name = Path(model_path).name.lower()
    if "qwen3" in name and ("4b" in name or "4-b" in name):
        return _QWEN3_4B_LAYERS
    # Unknown model: return a large number llama.cpp interprets as "all"
    return 99


def enhance_prompt(prompt: str, cfg: Dict[str, Any],
                   progress_callback: Optional[Callable] = None,
                   _oom_status_callback: Optional[Callable[[str], None]] = None,
                   ) -> str:
    """
    Run llama-cli to expand the user prompt into a rich image-gen prompt.

    If the encoder backend is Vulkan and the process exits with an OOM
    indication, up to 3 retries are attempted with progressively fewer GPU
    layers (reductions of 1, then 2, then 4 from the previous attempt).
    On exhausting all retries the original prompt is returned and
    _oom_status_callback (if supplied) is called with a user-facing message.

    Retry schedule (example, Qwen3-4B at -1 → 36 layers):
        Attempt 1 : ngl = 36   (or configured value)
        Attempt 2 : ngl = 35   (-1 from previous)
        Attempt 3 : ngl = 33   (-2 from previous)
        Attempt 4 : ngl = 29   (-4 from previous)
    """
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

    base_args = [
        str(cli), "-m", model_path, "-p", full_prompt,
        "-c", str(cfg.get("encoder_ctx_size", 4096)),
        "-t", str(cfg.get("encoder_threads", configure.get_default_threads())),
        "-b", str(cfg.get("encoder_batch_size", 512)),
        "-n", "256", "--temp", "0.7", "--log-disable",
    ]
    if cfg.get("encoder_flash_attn", True):
        base_args.append("--flash-attn")

    backend = cfg.get("backend_encoder", "CPU")
    use_vulkan = "Vulkan" in backend
    env = os.environ.copy()

    if use_vulkan:
        env["GGML_VULKAN_DEVICE"] = str(cfg.get("vulkan_device", 1))
        raw_ngl = int(cfg.get("encoder_gpu_layers", -1))
        current_ngl = _resolve_gpu_layers(raw_ngl, model_path)
    else:
        current_ngl = 0

    # Retry dampening: reductions applied cumulatively each attempt.
    # Attempt 0 (first run) uses current_ngl unchanged.
    # Attempt 1 : subtract 1  (total -1)
    # Attempt 2 : subtract 2  (total -3)
    # Attempt 3 : subtract 4  (total -7)
    _retry_reductions: Tuple[int, ...] = (0, 1, 2, 4)
    max_attempts = len(_retry_reductions)  # 4

    phase_t0 = time.time()
    if progress_callback:
        progress_callback("Enhancing prompt with LLM...", 0.1,
                          {"phase": "encoding", "phase_start": phase_t0})

    for attempt in range(max_attempts):
        # Cumulative layer reduction across retries: 0, -1, -3, -7
        cumulative_cut = sum(_retry_reductions[1:attempt + 1])
        ngl = max(0, current_ngl - cumulative_cut)

        args = base_args + ["-ngl", str(ngl)]

        if attempt > 0:
            print(
                f"[enhance_prompt] OOM retry {attempt}/{max_attempts - 1}: "
                f"ngl={ngl} (was {ngl + sum(_retry_reductions[attempt:attempt + 1])})",
                flush=True,
            )
            if progress_callback:
                progress_callback(
                    f"Enhancing prompt (OOM retry {attempt}, ngl={ngl})...", 0.1,
                    {"phase": "encoding", "phase_start": phase_t0})

        try:
            result = subprocess.run(
                args, capture_output=True, text=True,
                timeout=120, encoding="utf-8", errors="replace", env=env,
            )
            combined_output = result.stdout + result.stderr

            # Check for OOM before inspecting the text output
            if result.returncode != 0 and _is_oom_output(combined_output):
                print(
                    f"[enhance_prompt] OOM detected (attempt {attempt + 1}, "
                    f"ngl={ngl}, exit={result.returncode})",
                    flush=True,
                )
                if attempt < max_attempts - 1:
                    continue  # retry with fewer layers
                # All retries exhausted
                oom_msg = (
                    "Out of memory — reduce encoder GPU layers in the "
                    "Configuration page."
                )
                print(f"[enhance_prompt] {oom_msg}", flush=True)
                if _oom_status_callback:
                    _oom_status_callback(oom_msg)
                return prompt

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
                encoder_elapsed = time.time() - phase_t0
                configure.update_timing_stat("encoder_seconds", round(encoder_elapsed, 3))
                if progress_callback:
                    progress_callback(
                        "Prompt enhanced.", 0.2,
                        {"phase": "encoding", "phase_done": True,
                         "phase_elapsed": encoder_elapsed})
                return output[:2000]

            # Non-OOM failure or empty output — no point retrying
            break

        except subprocess.TimeoutExpired:
            break
        except Exception:
            break

    return prompt


# ---------------------------------------------------------------------------
# sd.cpp step-progress parsing
# ---------------------------------------------------------------------------
# stable-diffusion.cpp prints per-step progress lines while sampling, in
# forms such as:
#   "  |==========>      | 2/4 - 1.23s/it"
#   "[sample] 3/4"
#   "sampling, step 4/4"
# This regex tolerates all of the above: it just looks for "N/M" where M
# matches --steps. It deliberately does NOT try to parse the progress-bar
# glyphs or the "s/it" suffix — those vary across builds and aren't needed.
_STEP_LINE_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _parse_step_line(line: str, total_steps: int) -> Optional[Tuple[int, int]]:
    """Return (current_step, total_steps) if line looks like step progress
    for THIS run (denominator matches total_steps), else None."""
    for m in _STEP_LINE_RE.finditer(line):
        cur, tot = int(m.group(1)), int(m.group(2))
        if tot == total_steps and 0 <= cur <= tot:
            return cur, tot
    return None


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

    # Enhance prompt — wire OOM status so the message surfaces in the UI.
    enhanced = prompt
    _oom_msg: List[str] = []  # populated by callback if OOM exhausted all retries
    if cfg.get("encoder_model_path") and Path(cfg["encoder_model_path"]).exists():
        try:
            enhanced = enhance_prompt(
                prompt, cfg, progress_callback,
                _oom_status_callback=lambda msg: _oom_msg.append(msg),
            )
        except Exception:
            enhanced = prompt
    # Surface OOM message to caller if enhance_prompt reported it.
    if _oom_msg:
        result["message"] = _oom_msg[-1]
        return result

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

    # ------------------------------------------------------------------
    # Device / component placement.
    #
    # sd.cpp has NO per-layer GPU offload for the diffusion model (unlike
    # llama.cpp's -ngl). Placement is whole-component only, controlled by
    # three real flags: --backend vulkanN (use the GPU device at all),
    # --clip-on-cpu (keep the text encoder — including the Qwen3 --llm
    # encoder used here — off VRAM), and --vae-on-cpu (keep the VAE off
    # VRAM). imagegen_placement selects between Full GPU / Split / Full CPU
    # via parse_diffuser_placement(), which returns exactly these booleans.
    #
    # Previously this only checked "Vulkan" in backend_imagegen and, when
    # true, always passed --backend vulkanN AND tried to keep the encoder
    # off VRAM with "--llm-to-cpu" — a flag that does not exist in sd.cpp.
    # That flag was silently ignored, so the encoder always loaded onto
    # VRAM alongside the diffusion model, which is what caused the
    # ErrorOutOfDeviceMemory crash. It also meant selecting "CPU" for the
    # ImageGen backend never actually stopped sd.cpp from using Vulkan,
    # since nothing forced --backend off.
    # ------------------------------------------------------------------
    placement_label = cfg.get("imagegen_placement", configure.DIFFUSER_PLACEMENT_FULL_GPU)
    placement = configure.parse_diffuser_placement(placement_label)

    env = os.environ.copy()
    if placement["use_vulkan_backend"]:
        vk_dev = cfg.get("vulkan_device", 1)
        # --backend replaces --vulkan in current sd.cpp builds.
        # Format: --backend vulkan{N}  (e.g. --backend vulkan1)
        args.extend(["--backend", f"vulkan{vk_dev}"])
        sdk = utilities.detect_vulkan().get("vulkan_sdk", "")
        if sdk:
            env["PATH"] = str(Path(sdk) / "Bin") + ";" + env.get("PATH", "")
    # else: deliberately omit --backend entirely so sd.cpp never attempts
    # a Vulkan device allocation — this is the actual CPU-only code path.

    if placement["clip_on_cpu"] and enc_path and Path(enc_path).exists():
        # Real sd.cpp flag (the text-encoder equivalent of llama.cpp's
        # --llm-to-cpu, which does not exist). Keeps the Qwen3 conditioner
        # weights (~3.6GB) off VRAM so only the diffusion model + VAE
        # (~4.6GB) occupy the GPU, fitting in an 8GB card.
        args.append("--clip-on-cpu")

    if placement["vae_on_cpu"]:
        args.append("--vae-on-cpu")

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

    total_steps = int(cfg.get("imagegen_steps", 4))
    diffusion_t0 = time.time()
    if progress_callback:
        progress_callback("Generating image...", 0.3,
                          {"phase": "diffusion", "phase_start": diffusion_t0,
                           "step": 0, "total_steps": total_steps})

    try:
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        output_lines: List[str] = []
        last_step = 0
        for line in process.stdout:
            output_lines.append(line)
            if progress_callback and ("step" in line.lower() or "%" in line or "/" in line):
                step_info = _parse_step_line(line, total_steps)
                info: Dict[str, Any] = {"phase": "diffusion", "phase_start": diffusion_t0}
                if step_info:
                    last_step = step_info[0]
                    info["step"] = step_info[0]
                    info["total_steps"] = step_info[1]
                try:
                    progress_callback(line.strip()[:100], 0.3, info)
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
        diffusion_elapsed = time.time() - diffusion_t0

        if process.returncode == 0 and output_path.exists():
            # Record diffusion timing for next-generation ETA, using
            # whichever step count we actually observed (falls back to the
            # configured total if no step lines matched the parser).
            steps_for_avg = last_step if last_step > 0 else total_steps
            configure.update_timing_stat("diffusion_total_seconds", round(diffusion_elapsed, 3))
            configure.update_timing_stat("diffusion_steps", steps_for_avg)
            if steps_for_avg > 0:
                configure.update_timing_stat(
                    "diffusion_per_step_seconds",
                    round(diffusion_elapsed / steps_for_avg, 3))
            result.update(
                success=True, output_path=str(output_path),
                message=f"Saved {output_name} ({elapsed:.1f}s)",
                seed_used=seed,
                elapsed_seconds=round(elapsed, 2),
            )
            if progress_callback:
                progress_callback(f"Done: {output_name}", 1.0,
                                  {"phase": "done", "phase_elapsed": diffusion_elapsed})
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


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def unload_models() -> None:
    """
    Unload / release models.

    The current backends (llama-cli, sd-cli) are launched as one-shot
    subprocesses, so there is no persistent process to kill.  This function
    resets APP_STATE so that a subsequent generation re-validates paths and
    starts fresh.  If persistent model-server processes are added later,
    terminate them here.
    """
    configure.APP_STATE["cancel_requested"] = False
    configure.APP_STATE["last_image_path"]  = ""
    configure.APP_STATE["models_unloaded"]  = True