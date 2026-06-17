#!/usr/bin/env python3
"""
display.py - Gradio 5 UI for Image Generator GGUF.
Three tabs: Generate | Configuration | Debug / Info
Build/install functionality lives in installer.py only.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import scripts.configure as configure
import scripts.inference as inference
import scripts.utilities as utilities


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cfg() -> Dict[str, Any]:
    """Fresh load from persistent.json."""
    return configure.load_persistent()


def _browse_file() -> str:
    """Open a native file dialog and return the chosen path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            initialdir=str(configure.get_models_dir()),
            filetypes=[
                ("Model files", "*.gguf *.safetensors"),
                ("GGUF",        "*.gguf"),
                ("Safetensors", "*.safetensors"),
                ("All files",   "*.*"),
            ],
        )
        root.destroy()
        return path or ""
    except Exception:
        return ""


def _backend_choices() -> List[str]:
    return configure.get_backend_choices()["all_choices"]


def _default_backend_value(key: str) -> str:
    """
    Load the saved backend string; if it's no longer in the current
    choices list (e.g. GPU was removed), fall back to CPU.
    """
    saved = _cfg().get(key, "CPU")
    choices = _backend_choices()
    return saved if saved in choices else "CPU"


def _thread_choices() -> List[int]:
    return configure.get_thread_choices()


# ---------------------------------------------------------------------------
# Tab 1 — Generate
# ---------------------------------------------------------------------------

def _build_generate_tab() -> None:
    cfg = _cfg()
    presets = configure.get_generation_presets()

    gr.Markdown("## Generate Image")
    gr.Markdown(
        "Enter a prompt and click **Generate**. "
        "Set model paths on the **Configuration** tab first."
    )

    with gr.Row():
        # Left column — prompt + output
        with gr.Column(scale=3):
            prompt_tb = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=4, max_lines=10,
                value=cfg.get("last_prompt", ""),
            )
            negative_tb = gr.Textbox(
                label="Negative Prompt",
                placeholder="Things to exclude...",
                lines=2,
                value=cfg.get("negative_prompt", ""),
            )
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                stop_btn     = gr.Button("Stop", variant="stop")
            output_gallery = gr.Gallery(
                label="Generated Images",
                columns=2, rows=1, height="auto", object_fit="contain",
            )
            status_tb = gr.Textbox(label="Status", interactive=False, lines=2)

        # Right column — generation settings
        with gr.Column(scale=2):
            gr.Markdown("### Settings")
            preset_dd = gr.Dropdown(
                label="Preset", choices=list(presets.keys()),
                value="Fast (Turbo)",
            )
            with gr.Row():
                width_dd  = gr.Dropdown(label="Width",  choices=configure.IMAGE_SIZES,
                                        value=cfg.get("imagegen_width", 512))
                height_dd = gr.Dropdown(label="Height", choices=configure.IMAGE_SIZES,
                                        value=cfg.get("imagegen_height", 512))
            steps_dd = gr.Dropdown(
                label="Steps", choices=configure.STEP_CHOICES,
                value=cfg.get("imagegen_steps", 4),
            )
            sampler_dd = gr.Dropdown(
                label="Sampler", choices=list(configure.SAMPLER_MAP.keys()),
                value=cfg.get("imagegen_sampling", "euler_a"),
            )
            cfg_scale_sld = gr.Slider(
                label="CFG Scale", minimum=0.5, maximum=20.0, step=0.5,
                value=cfg.get("imagegen_cfg_scale", 1.0),
            )
            with gr.Row():
                seed_num  = gr.Number(label="Seed (-1 = random)",
                                      value=cfg.get("imagegen_seed", -1), precision=0)
                batch_dd  = gr.Dropdown(label="Batch Count",
                                        choices=configure.BATCH_COUNT_CHOICES,
                                        value=cfg.get("imagegen_batch_count", 1))
            enhance_chk = gr.Checkbox(label="Enhance prompt with LLM encoder", value=True)
            save_btn    = gr.Button("Save as Default", size="sm")
            save_status = gr.Textbox(interactive=False, show_label=False)

    # ── Event: preset change ──
    def apply_preset(name: str):
        p = presets.get(name, {})
        return (p.get("imagegen_width", 512),
                p.get("imagegen_height", 512),
                p.get("imagegen_steps", 4),
                p.get("imagegen_sampling", "euler_a"),
                p.get("imagegen_cfg_scale", 1.0))

    preset_dd.change(
        apply_preset, inputs=preset_dd,
        outputs=[width_dd, height_dd, steps_dd, sampler_dd, cfg_scale_sld],
    )

    # ── Event: generate ──
    def do_generate(prompt, negative, width, height, steps, sampler,
                    cfg_scale, seed, batch, enhance,
                    progress=gr.Progress()):
        if not prompt or not prompt.strip():
            return [], "Please enter a prompt."
        c = _cfg()
        if not c.get("imagegen_model_path") or not Path(c["imagegen_model_path"]).exists():
            return [], "Image generation model not configured. Go to Configuration tab."
        if not c.get("vae_model_path") or not Path(c["vae_model_path"]).exists():
            return [], "VAE model not configured. Go to Configuration tab."

        gen_cfg = dict(c)
        gen_cfg.update(
            imagegen_width=int(width), imagegen_height=int(height),
            imagegen_steps=int(steps), imagegen_sampling=sampler,
            imagegen_cfg_scale=float(cfg_scale),
            imagegen_seed=int(seed), imagegen_batch_count=int(batch),
            negative_prompt=negative,
        )
        if not enhance:
            gen_cfg["encoder_model_path"] = ""

        configure.APP_STATE["cancel_requested"] = False

        def prog_cb(msg: str, pct: float):
            progress(pct, desc=msg)

        result = inference.generate_image(prompt.strip(), gen_cfg,
                                          progress_callback=prog_cb)
        if result["success"] and result["output_path"]:
            msg = (f"{result['message']} | Seed: {result['seed_used']} "
                   f"| Time: {result['elapsed_seconds']}s")
            return [result["output_path"]], msg
        return [], result.get("message", "Unknown error")

    generate_btn.click(
        do_generate,
        inputs=[prompt_tb, negative_tb, width_dd, height_dd, steps_dd,
                sampler_dd, cfg_scale_sld, seed_num, batch_dd, enhance_chk],
        outputs=[output_gallery, status_tb],
    )

    # ── Event: stop ──
    stop_btn.click(
        lambda: (configure.APP_STATE.__setitem__("cancel_requested", True),
                 "Cancel requested...")[1],
        outputs=status_tb,
    )

    # ── Event: save defaults ──
    def save_defaults(width, height, steps, sampler, cfg_scale,
                      seed, batch, neg):
        configure.update_persistent({
            "imagegen_width": int(width), "imagegen_height": int(height),
            "imagegen_steps": int(steps), "imagegen_sampling": sampler,
            "imagegen_cfg_scale": float(cfg_scale),
            "imagegen_seed": int(seed), "imagegen_batch_count": int(batch),
            "negative_prompt": neg,
        })
        return "Defaults saved!"

    save_btn.click(
        save_defaults,
        inputs=[width_dd, height_dd, steps_dd, sampler_dd,
                cfg_scale_sld, seed_num, batch_dd, negative_tb],
        outputs=save_status,
    )


# ---------------------------------------------------------------------------
# Tab 2 — Configuration
# ---------------------------------------------------------------------------

def _build_config_tab() -> None:
    cfg     = _cfg()
    choices = _backend_choices()
    threads = _thread_choices()
    dt      = configure.get_default_threads()

    gr.Markdown("## Configuration")
    gr.Markdown(
        "Set model paths and backend options. "
        "Use **Browse** to locate files or **Scan** to auto-detect in `models/`."
    )

    # ── Model paths ──
    gr.Markdown("### Model Paths")
    with gr.Row():
        with gr.Column():
            enc_path_tb = gr.Textbox(
                label="Encoder Model (GGUF)",
                value=cfg.get("encoder_model_path", ""),
                placeholder="Qwen3-4b-Uncensored-Z-Image-Engineer-V4-Q4_K_M.gguf",
                info="LLM for prompt enhancement. Any Q# quantization.",
            )
            enc_name_tb = gr.Textbox(label="Encoder Name",
                                     value=cfg.get("encoder_model_name", ""))
            with gr.Row():
                enc_browse_btn = gr.Button("Browse...", size="sm")
                enc_scan_btn   = gr.Button("Scan models/", size="sm")

        with gr.Column():
            diff_path_tb = gr.Textbox(
                label="Image Generation Model (GGUF)",
                value=cfg.get("imagegen_model_path", ""),
                placeholder="z_image_turbo-Q4_K_M.gguf",
                info="Diffusion model. Any Q# quantization.",
            )
            diff_name_tb = gr.Textbox(label="Diffusion Name",
                                      value=cfg.get("imagegen_model_name", ""))
            with gr.Row():
                diff_browse_btn = gr.Button("Browse...", size="sm")
                diff_scan_btn   = gr.Button("Scan models/", size="sm")

        with gr.Column():
            vae_path_tb = gr.Textbox(
                label="VAE Model (Safetensors)",
                value=cfg.get("vae_model_path", ""),
                placeholder="ae.safetensors",
                info="Autoencoder for image decoding.",
            )
            vae_name_tb = gr.Textbox(label="VAE Name",
                                     value=cfg.get("vae_model_name", ""))
            with gr.Row():
                vae_browse_btn = gr.Button("Browse...", size="sm")
                vae_scan_btn   = gr.Button("Scan models/", size="sm")

    # ── Backend selection ──
    gr.Markdown("### Backend Selection")
    gr.Markdown(
        "Choices are populated from `data/constants.ini` written during installation. "
        "CPU = run on processor. Vulkan GPU N = run on that GPU."
    )
    with gr.Row():
        enc_backend_dd = gr.Dropdown(
            label="Encoder Backend",
            choices=choices,
            value=_default_backend_value("backend_encoder"),
            info="Where to run the LLM prompt encoder.",
        )
        img_backend_dd = gr.Dropdown(
            label="ImageGen Backend",
            choices=choices,
            value=_default_backend_value("backend_imagegen"),
            info="Where to run image diffusion.",
        )

    # ── Encoder (LLM) settings ──
    gr.Markdown("### Encoder (LLM) Settings")
    with gr.Row():
        enc_threads_dd = gr.Dropdown(
            label="CPU Threads",
            choices=threads,
            value=cfg.get("encoder_threads", dt),
            info=f"Detected default: {dt} (85% of {configure.get_cpu_info()['cores_logical']} logical cores)",
        )
        enc_batch_dd = gr.Dropdown(label="Batch Size",
                                   choices=configure.BATCH_SIZE_CHOICES,
                                   value=cfg.get("encoder_batch_size", 512))
        enc_ctx_dd = gr.Dropdown(label="Context Size",
                                 choices=configure.CTX_SIZE_CHOICES,
                                 value=cfg.get("encoder_ctx_size", 4096))
        enc_ngl_dd = gr.Dropdown(label="GPU Layers (-1 = all)",
                                 choices=configure.GPU_LAYER_CHOICES,
                                 value=cfg.get("encoder_gpu_layers", -1),
                                 info="Layers to offload to GPU. Ignored for CPU backend.")
    enc_flash_chk = gr.Checkbox(
        label="Flash Attention",
        value=cfg.get("encoder_flash_attn", True),
        info="Reduces VRAM usage for long contexts.",
    )

    # ── ImageGen settings ──
    gr.Markdown("### Image Generation Settings")
    with gr.Row():
        img_threads_dd = gr.Dropdown(
            label="CPU Threads",
            choices=threads,
            value=cfg.get("imagegen_threads", dt),
        )
        img_clip_dd = gr.Dropdown(label="CLIP Skip",
                                  choices=configure.CLIP_SKIP_CHOICES,
                                  value=cfg.get("imagegen_clip_skip", 2))
        out_fmt_dd = gr.Dropdown(label="Output Format",
                                 choices=configure.OUTPUT_FORMATS,
                                 value=cfg.get("output_format", "png"))

    # ── Prompt template ──
    gr.Markdown("### Advanced")
    prompt_template_tb = gr.Textbox(
        label="Prompt Template  ({prompt} is replaced by the user input)",
        value=cfg.get("prompt_template",
                      "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
        lines=3,
    )

    # ── Save ──
    save_all_btn    = gr.Button("Save All Configuration", variant="primary", size="lg")
    save_all_status = gr.Textbox(interactive=False, show_label=False)

    # ── Events: browse ──
    def _browse():
        p = _browse_file()
        return (p, Path(p).stem) if p else (gr.update(), gr.update())

    enc_browse_btn.click(_browse,  outputs=[enc_path_tb,  enc_name_tb])
    diff_browse_btn.click(_browse, outputs=[diff_path_tb, diff_name_tb])
    vae_browse_btn.click(_browse,  outputs=[vae_path_tb,  vae_name_tb])

    # ── Events: scan ──
    def _scan(keyword: str, ext: str):
        models_dir = configure.get_models_dir()
        if not models_dir.exists():
            return "", ""
        models = inference.scan_directory(str(models_dir), (ext,))
        cats   = inference.categorize_models(models)
        for cat in ("encoder", "diffusion", "vae", "unknown"):
            for m in cats.get(cat, []):
                if keyword.lower() in m.get("filename", "").lower():
                    return m["file"], m.get("filename", "")
        return "", ""

    enc_scan_btn.click(lambda: _scan("qwen",  ".gguf"),
                       outputs=[enc_path_tb,  enc_name_tb])
    diff_scan_btn.click(lambda: _scan("turbo", ".gguf"),
                        outputs=[diff_path_tb, diff_name_tb])
    vae_scan_btn.click(lambda: _scan("ae",    ".safetensors"),
                       outputs=[vae_path_tb,  vae_name_tb])

    # ── Event: save all ──
    def save_all(ep, en, dp, dn, vp, vn,
                 enc_back, img_back,
                 et, eb, ec, engl, ef,
                 it, ic, of, pt):

        # Parse backend strings → vulkan flags for storage
        enc_parsed = configure.parse_backend_choice(enc_back)
        img_parsed = configure.parse_backend_choice(img_back)

        configure.update_persistent({
            "encoder_model_path":  ep,  "encoder_model_name":  en,
            "imagegen_model_path": dp,  "imagegen_model_name": dn,
            "vae_model_path":      vp,  "vae_model_name":      vn,
            "backend_encoder":     enc_back,
            "backend_imagegen":    img_back,
            # Store resolved Vulkan device for encoder and imagegen separately
            "encoder_vulkan_device": enc_parsed["vulkan_device"],
            "imagegen_vulkan_device": img_parsed["vulkan_device"],
            # Keep legacy vulkan_device = imagegen device for compatibility
            "vulkan_device":         img_parsed["vulkan_device"],
            "encoder_threads":     int(et),
            "encoder_batch_size":  int(eb),
            "encoder_ctx_size":    int(ec),
            "encoder_gpu_layers":  int(engl),
            "encoder_flash_attn":  bool(ef),
            "imagegen_threads":    int(it),
            "imagegen_clip_skip":  int(ic),
            "output_format":       of,
            "prompt_template":     pt,
            "first_run":           False,
        })
        return "All settings saved!"

    save_all_btn.click(
        save_all,
        inputs=[
            enc_path_tb, enc_name_tb, diff_path_tb, diff_name_tb,
            vae_path_tb, vae_name_tb,
            enc_backend_dd, img_backend_dd,
            enc_threads_dd, enc_batch_dd, enc_ctx_dd, enc_ngl_dd, enc_flash_chk,
            img_threads_dd, img_clip_dd, out_fmt_dd,
            prompt_template_tb,
        ],
        outputs=save_all_status,
    )


# ---------------------------------------------------------------------------
# Tab 3 — Debug / Info
# ---------------------------------------------------------------------------

def _collect_debug() -> str:
    import time
    cpu  = configure.get_cpu_info()
    vk   = configure.get_vulkan_info()
    mem  = utilities.get_memory_info()
    bs   = utilities.get_build_status()
    cfg  = configure.load_persistent()
    env  = utilities.get_relevant_env()

    enc_path  = cfg.get("encoder_model_path", "")
    diff_path = cfg.get("imagegen_model_path", "")
    vae_path  = cfg.get("vae_model_path", "")

    L: List[str] = []
    L.append("=" * 72)
    L.append(f"  DEBUG REPORT  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("=" * 72)

    L.append(f"\nPython  : {platform.python_version()}")
    try:
        import gradio as gr
        L.append(f"Gradio  : {gr.__version__}")
    except Exception:
        L.append("Gradio  : not importable")
    L.append(f"Platform: {platform.platform()}")

    L.append(f"\nCPU     : {cpu['brand']}  [{cpu['vendor']}]  {cpu['cores_logical']} threads")
    L.append(f"Default : {cpu['default_threads']} threads (85%)")
    L.append(f"AVX2:{cpu['has_avx2']}  F16C:{cpu['has_f16c']}  FMA:{cpu['has_fma']}  "
             f"AVX512:{cpu['has_avx512']}  AOCL:{cpu['has_aocl']}")

    if mem:
        L.append(f"\nRAM     : {mem.get('ram_used_mb','?')} / "
                 f"{mem.get('ram_total_mb','?')} MB  ({mem.get('ram_percent','?')}%)")

    L.append(f"\nVulkan  : {vk['available']}  ver={vk['version']}")
    L.append(f"SDK     : {vk['sdk'] or 'not set'}")
    for d in vk["devices"]:
        L.append(f"  GPU{d['index']}: {d['name']}")

    L.append(f"\nllama.cpp : {'built  ' + bs['llama_path'] if bs['llama_built'] else 'NOT BUILT'}")
    L.append(f"sd.cpp    : {'built  ' + bs['sd_path']    if bs['sd_built']    else 'NOT BUILT'}")

    ll = inference.find_llama_cli()
    sd = inference.find_sd_cpp()
    L.append(f"\nllama-cli : {ll or 'NOT FOUND'}")
    L.append(f"sd exe    : {sd or 'NOT FOUND'}")

    def _status(p: str) -> str:
        if not p:
            return "NOT SET"
        return ("EXISTS  " if Path(p).exists() else "MISSING ") + p

    L.append(f"\nEncoder  : {_status(enc_path)}")
    L.append(f"Diffusion: {_status(diff_path)}")
    L.append(f"VAE      : {_status(vae_path)}")

    L.append(f"\nEnc backend : {cfg.get('backend_encoder','?')}")
    L.append(f"Img backend : {cfg.get('backend_imagegen','?')}")
    L.append(f"Enc threads : {cfg.get('encoder_threads','?')}")
    L.append(f"Img threads : {cfg.get('imagegen_threads','?')}")
    L.append(f"Size        : {cfg.get('imagegen_width','?')}×{cfg.get('imagegen_height','?')}")
    L.append(f"Steps       : {cfg.get('imagegen_steps','?')}  "
             f"CFG: {cfg.get('imagegen_cfg_scale','?')}  "
             f"Sampler: {cfg.get('imagegen_sampling','?')}")

    if env:
        L.append(f"\nEnvironment:")
        for k, v in sorted(env.items()):
            L.append(f"  {k} = {v}")

    L.append("\n" + "=" * 72)
    return "\n".join(L)


def _copy_to_clipboard(text: str) -> str:
    if not text:
        return "Nothing to copy."
    try:
        if sys.platform == "win32":
            proc = subprocess.Popen(["clip.exe"],
                                    stdin=subprocess.PIPE, text=True)
            proc.communicate(text, timeout=10)
            return "Copied to clipboard!"
        return "Clipboard copy only supported on Windows."
    except Exception as e:
        return f"Copy failed: {e}"


def _build_debug_tab() -> gr.Textbox:
    gr.Markdown("## Debug / System Info")
    gr.Markdown(
        "Live system info sourced from `constants.ini` and runtime state. "
        "Click **Refresh** to update. Click **Copy** to copy to clipboard."
    )
    with gr.Row():
        refresh_btn = gr.Button("Refresh", variant="primary")
        copy_btn    = gr.Button("Copy to Clipboard")
    copy_status = gr.Textbox(interactive=False, show_label=False)
    info_text   = gr.Textbox(
        label="System Information",
        interactive=False,
        lines=38, max_lines=80,
        autoscroll=False,
    )

    refresh_btn.click(_collect_debug, outputs=info_text)
    copy_btn.click(_copy_to_clipboard, inputs=info_text, outputs=copy_status)
    return info_text   # <-- required for app.load wiring


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Assemble and return the Gradio application."""
    configure.ensure_data_dirs()

    # Theme removed from Blocks constructor (moved to launch() in launcher.py)
    with gr.Blocks(title="Image Generator GGUF") as app:   # <-- theme=... removed
        gr.Markdown("# Image Generator GGUF")
        gr.Markdown(
            "Local image generation using GGUF diffusion models "
            "with optional LLM prompt enhancement."
        )

        with gr.Tabs():
            with gr.TabItem("Generate"):
                _build_generate_tab()

            with gr.TabItem("Configuration"):
                _build_config_tab()

            with gr.TabItem("Debug / Info"):
                info_text = _build_debug_tab()   # capture the returned textbox

        # Load debug info into that textbox on startup – no more "too many outputs" warning
        app.load(_collect_debug, outputs=info_text)

    return app