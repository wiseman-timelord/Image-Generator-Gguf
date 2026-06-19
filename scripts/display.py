#!/usr/env python3
"""
display.py - Gradio 5 UI for Image-Gradio-Gguf.
Three tabs: Generate | Configuration | Debug / Info
Build/install functionality lives in installer.py only.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import scripts.configure as configure
import scripts.inference as inference


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
    choices list (e.g. GPU was removed, or this is a fresh install with
    a newly-branded CPU label), fall back to the CPU entry.
    """
    saved = _cfg().get(key, "")
    choices = _backend_choices()
    if saved in choices:
        return saved
    # First entry is always the CPU label
    return choices[0] if choices else "CPU"


def _thread_choices() -> List[int]:
    return configure.get_thread_choices()


def _detect_vae(diff_path_str: str) -> Tuple[str, str]:
    """
    Given a diffusion model path, look for ae.safetensors in the same directory.
    Returns (vae_full_path, vae_stem) or ("", "") if not found.
    """
    if not diff_path_str:
        return "", ""
    p = Path(diff_path_str).expanduser()
    if not p.exists():
        return "", ""
    parent = p.parent
    # Search for ae.safetensors (case-insensitive)
    for f in parent.iterdir():
        if f.is_file() and f.name.lower() == "ae.safetensors":
            return str(f), f.stem
    return "", ""


# ---------------------------------------------------------------------------
# Recent images helpers
# ---------------------------------------------------------------------------

def _get_recent_images(max_images: int = 50) -> List[str]:
    """Return paths of images in the output folder, newest first."""
    out_dir = configure.get_output_dir()
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    try:
        files = [
            f for f in out_dir.iterdir()
            if f.is_file() and f.suffix.lower() in exts
        ]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return [str(f) for f in files[:max_images]]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tab 1 — Generate  (UI widgets + event wiring split for shared status)
# ---------------------------------------------------------------------------

# Module-level refs so _wire_generate_events() can access them
_gen: Dict[str, Any] = {}


def _models_configured() -> bool:
    """Return True if all three model paths are set and exist on disk."""
    c = configure.load_persistent()
    return all(
        c.get(k) and Path(c[k]).exists()
        for k in ("encoder_model_path", "imagegen_model_path", "vae_model_path")
    )


def _build_generate_tab_inner() -> None:
    """Build Generate tab widgets; store refs in _gen for later wiring."""
    cfg = _cfg()
    presets = configure.get_generation_presets()
    configured = _models_configured()

    prompt_ph  = ("Describe the image you want to generate..."
                  if configured else "Set locations of models on Configuration page first...")
    neg_ph     = ("Things to exclude..."
                  if configured else "Set locations of models on Configuration page first...")

    with gr.Row():
        # ── Left column: settings ────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.Markdown("### Settings")
            _gen["prompt_tb"] = gr.Textbox(
                label="Prompt",
                placeholder=prompt_ph,
                lines=2, max_lines=10,
                value=cfg.get("last_prompt", ""),
            )
            _gen["negative_tb"] = gr.Textbox(
                label="Negative Prompt",
                placeholder=neg_ph,
                lines=2, max_lines=10,
                value=cfg.get("negative_prompt", ""),
            )
            with gr.Row():
                _gen["preset_dd"] = gr.Dropdown(
                    label="Preset", choices=list(presets.keys()), value="Fast (Turbo)",
                )
                _gen["sampler_dd"] = gr.Dropdown(
                    label="Sampler", choices=list(configure.SAMPLER_MAP.keys()),
                    value=cfg.get("imagegen_sampling", "euler_a"),
                )

            with gr.Row():
                _gen["batch_dd"] = gr.Dropdown(label="Batch Count",
                                               choices=configure.BATCH_COUNT_CHOICES,
                                               value=cfg.get("imagegen_batch_count", 1)
                )
                _gen["width_dd"]  = gr.Dropdown(label="Width",  choices=configure.IMAGE_SIZES,
                                                value=cfg.get("imagegen_width", 512))
                _gen["height_dd"] = gr.Dropdown(label="Height", choices=configure.IMAGE_SIZES,
                                                value=cfg.get("imagegen_height", 512))
                _gen["output_fmt_dd"] = gr.Dropdown(
                    label="Output Format", choices=configure.OUTPUT_FORMATS,
                    value=cfg.get("output_format", "png"),
                )

            with gr.Row():
                _gen["steps_dd"] = gr.Dropdown(
                    label="Steps", choices=configure.STEP_CHOICES,
                    value=cfg.get("imagegen_steps", 4),
                )
                _gen["cfg_scale_sld"] = gr.Slider(
                    label="CFG Scale", minimum=0.5, maximum=20.0, step=0.5,
                    value=cfg.get("imagegen_cfg_scale", 1.0),
                )
                _gen["seed_num"] = gr.Number(label="Seed (-1 = random)",
                                             value=cfg.get("imagegen_seed", -1), precision=0)


            _gen["save_btn"]    = gr.Button("Save as Default", size="sm")


        # ── Right column: generate + gallery ────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Output")
            with gr.Row(visible=configured) as _gen["generate_row"]:
                _gen["generate_btn"] = gr.Button("Generate", variant="primary", size="lg")
                _gen["stop_btn"]     = gr.Button("Stop", variant="stop")

            _gen["output_gallery"] = gr.Gallery(
                label="Generated Images",
                columns=2, rows=2,
                height=585,
                object_fit="contain",
            )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Gallery")
            # ── Recent images strip (horizontal scroll, thumbnails) ──
            _gen["recent_strip"] = gr.HTML(
                value="",
                label=None,
                elem_id="recent-strip-wrap",
            )

    # Preset change wired immediately
    presets_map = presets

    def apply_preset(name: str):
        p = presets_map.get(name, {})
        return (p.get("imagegen_width", 512), p.get("imagegen_height", 512),
                p.get("imagegen_steps", 4), p.get("imagegen_sampling", "euler_a"),
                p.get("imagegen_cfg_scale", 1.0))

    _gen["preset_dd"].change(
        apply_preset, inputs=_gen["preset_dd"],
        outputs=[_gen["width_dd"], _gen["height_dd"], _gen["steps_dd"],
                 _gen["sampler_dd"], _gen["cfg_scale_sld"]],
    )


def _wire_generate_events(status_box: gr.Textbox) -> None:
    """Register Generate tab event handlers that output to shared status_box."""
    import threading

    # ── Inactivity timeout (20 min after last generation finishes) ──
    _INACTIVITY_SECONDS = 20 * 60
    _timeout_timer: Dict[str, Any] = {"handle": None}

    def _reset_inactivity_timer():
        if _timeout_timer["handle"] is not None:
            _timeout_timer["handle"].cancel()
        t = threading.Timer(_INACTIVITY_SECONDS, inference.unload_models)
        t.daemon = True
        t.start()
        _timeout_timer["handle"] = t

    def _cancel_inactivity_timer():
        if _timeout_timer["handle"] is not None:
            _timeout_timer["handle"].cancel()
            _timeout_timer["handle"] = None

    # ── Recent images strip renderer ──
    def _render_recent_strip() -> str:
        paths = _get_recent_images()
        if not paths:
            return (
                '<div id="recent-strip-wrap">'
                '<div id="recent-strip-label">Recent Images</div>'
                '<div id="recent-strip"><span id="recent-empty">No images generated yet.</span></div>'
                '</div>'
            )
        thumbs = []
        for p in paths:
            # Convert to a file:// URL the browser can load; forward-slash for Windows
            url = Path(p).as_uri()
            fname = Path(p).name
            thumbs.append(
                f'<img class="recent-thumb" src="{url}" title="{fname}" '
                f'onclick="(function(src){{var g=document.querySelector(\'#output_gallery img\');'
                f'if(!g)return;'
                f'document.querySelectorAll(\'.recent-thumb\').forEach(function(t){{t.classList.remove(\'selected\');}});'
                f'this.classList.add(\'selected\');}}).call(this,\'{url}\')" />'
            )
        body = "\n".join(thumbs)
        return (
            '<div id="recent-strip-wrap">'
            '<div id="recent-strip-label">Recent Images</div>'
            f'<div id="recent-strip">{body}</div>'
            '</div>'
        )

    def do_generate(prompt, negative, width, height, steps, sampler,
                    cfg_scale, seed, batch, output_format,
                    progress=gr.Progress()):
        if not prompt or not prompt.strip():
            return [], "Please enter a prompt."
        c = _cfg()
        if not c.get("imagegen_model_path") or not Path(c["imagegen_model_path"]).exists():
            return [], "Image generation model not configured. Go to Configuration tab."
        if not c.get("vae_model_path") or not Path(c["vae_model_path"]).exists():
            return [], "VAE model not configured. Go to Configuration tab."

        # Cancel any pending unload while we are generating
        _cancel_inactivity_timer()

        gen_cfg = dict(c)
        gen_cfg.update(
            imagegen_width=int(width), imagegen_height=int(height),
            imagegen_steps=int(steps), imagegen_sampling=sampler,
            imagegen_cfg_scale=float(cfg_scale),
            imagegen_seed=int(seed), imagegen_batch_count=int(batch),
            negative_prompt=negative,
            output_format=output_format,
        )
        # Always use encoder if path is set; no toggle needed

        configure.APP_STATE["cancel_requested"] = False

        def prog_cb(msg: str, pct: float):
            progress(pct, desc=msg)

        result = inference.generate_image(prompt.strip(), gen_cfg,
                                          progress_callback=prog_cb)

        # Start inactivity timer now that generation is finished
        _reset_inactivity_timer()

        if result["success"] and result["output_path"]:
            msg = (f"{result['message']} | Seed: {result['seed_used']} "
                   f"| Time: {result['elapsed_seconds']}s")
            return [result["output_path"]], msg, _render_recent_strip()
        return [], result.get("message", "Unknown error"), _render_recent_strip()

    _gen["generate_btn"].click(
        do_generate,
        inputs=[_gen["prompt_tb"], _gen["negative_tb"],
                _gen["width_dd"], _gen["height_dd"], _gen["steps_dd"],
                _gen["sampler_dd"], _gen["cfg_scale_sld"],
                _gen["seed_num"], _gen["batch_dd"], _gen["output_fmt_dd"]],
        outputs=[_gen["output_gallery"], status_box, _gen["recent_strip"]],
    )

    _gen["stop_btn"].click(
        lambda: (configure.APP_STATE.__setitem__("cancel_requested", True),
                 "Cancel requested...")[1],
        outputs=status_box,
    )

    def save_defaults(width, height, steps, sampler, cfg_scale, seed, batch, neg, output_format):
        configure.update_persistent({
            "imagegen_width": int(width), "imagegen_height": int(height),
            "imagegen_steps": int(steps), "imagegen_sampling": sampler,
            "imagegen_cfg_scale": float(cfg_scale),
            "imagegen_seed": int(seed), "imagegen_batch_count": int(batch),
            "negative_prompt": neg,
            "output_format": output_format,
        })
        return "Defaults saved!"

    _gen["save_btn"].click(
        save_defaults,
        inputs=[_gen["width_dd"], _gen["height_dd"], _gen["steps_dd"],
                _gen["sampler_dd"], _gen["cfg_scale_sld"],
                _gen["seed_num"], _gen["batch_dd"], _gen["negative_tb"],
                _gen["output_fmt_dd"]],
        outputs=status_box,
    )

    # ── Refresh generate_row visibility when prompt changes ──
    def _refresh_generate_row(prompt_val):
        configured = _models_configured()
        prompt_ph = ("Describe the image you want to generate..."
                     if configured else "Set locations of models on Configuration page first...")
        neg_ph    = ("Things to exclude..."
                     if configured else "Set locations of models on Configuration page first...")
        return (
            gr.update(visible=configured),
            gr.update(placeholder=prompt_ph),
            gr.update(placeholder=neg_ph),
        )

    _gen["prompt_tb"].focus(
        _refresh_generate_row,
        inputs=_gen["prompt_tb"],
        outputs=[_gen["generate_row"], _gen["prompt_tb"], _gen["negative_tb"]],
    )


# ---------------------------------------------------------------------------
# Tab 2 — Configuration  (UI widgets + event wiring split for shared status)
# ---------------------------------------------------------------------------

_cfg_w: Dict[str, Any] = {}  # widget refs for wiring
_dbg:   Dict[str, Any] = {}  # debug tab widget refs


def _build_config_tab_inner() -> None:
    """Build Configuration tab widgets; store refs in _cfg_w for later wiring."""
    cfg     = _cfg()
    choices = _backend_choices()
    threads = _thread_choices()
    dt      = configure.get_default_threads()

    # ── Model paths (encoder + diffusion on one row) ──
    gr.Markdown("### Model Paths")
    # Path textboxes are hidden — used internally by browse callbacks and save
    _cfg_w["enc_path_tb"]  = gr.Textbox(value=cfg.get("encoder_model_path", ""),  visible=False)
    _cfg_w["diff_path_tb"] = gr.Textbox(value=cfg.get("imagegen_model_path", ""), visible=False)

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                _cfg_w["enc_name_tb"] = gr.Textbox(
                    label="Encoder Name",
                    value=cfg.get("encoder_model_name", ""),
                    placeholder="Qwen3-4b-Z-Image-Turbo",
                    interactive=True,
                    scale=8,
                )
                _cfg_w["enc_browse_btn"] = gr.Button("Browse...", size="sm", scale=1, min_width=90)

        with gr.Column(scale=4):
            with gr.Row():
                _cfg_w["diff_name_tb"] = gr.Textbox(
                    label="Diffusion Name",
                    value=cfg.get("imagegen_model_name", ""),
                    placeholder="z_image_turbo",
                    interactive=True,
                    scale=8,
                )
                _cfg_w["diff_browse_btn"] = gr.Button("Browse...", size="sm", scale=1, min_width=90)

    # ── VAE path is auto-detected from diffusion model folder; stored hidden ──
    _cfg_w["vae_path_tb"] = gr.Textbox(
        value=cfg.get("vae_model_path", ""), visible=False)
    _cfg_w["vae_name_tb"] = gr.Textbox(
        value=cfg.get("vae_model_name", ""), visible=False)

    # ── Backend selection + CPU threads (consolidated) ──
    is_cpu_only = configure.get_install_type() == "cpu_only"

    gr.Markdown("### Backend Selection")
    if is_cpu_only:
        gr.Markdown(
            "GPU options are not available (Cpu-only install). "
        )
    else:
        gr.Markdown(
            "GPU options are available (Vulkan install). "
        )
    with gr.Row():
        with gr.Column(scale=2):
            _cfg_w["enc_backend_dd"] = gr.Dropdown(
                label="Encoder Backend",
                choices=choices,
                value=_default_backend_value("backend_encoder"),
                interactive=not is_cpu_only,
            )

        with gr.Column(scale=1):
            _cfg_w["threads_dd"] = gr.Dropdown(
                label="CPU Threads",
                choices=threads,
                value=cfg.get("encoder_threads", dt),
            )

        with gr.Column(scale=2):
            _cfg_w["img_backend_dd"] = gr.Dropdown(
                label="ImageGen Backend",
                choices=choices,
                value=_default_backend_value("backend_imagegen"),
                interactive=not is_cpu_only,
            )

    # Initial interactive/value state for the two GPU-dependent controls
    # below is driven by the ACTUAL selected backend, not just install type.
    # A Vulkan-capable install with "CPU" currently selected must still show
    # GPU Layers / Diffuser Placement as locked at their CPU-forced values —
    # otherwise the controls look interactive but silently have no effect,
    # which was the root of the original bug report.
    enc_backend_val = _default_backend_value("backend_encoder")
    enc_is_vulkan   = (not is_cpu_only) and ("Vulkan" in enc_backend_val)
    img_backend_val = _default_backend_value("backend_imagegen")
    img_is_vulkan   = (not is_cpu_only) and ("Vulkan" in img_backend_val)

    with gr.Row():
        # ── Encoder (LLM) settings ──
        with gr.Column(scale=2):
            gr.Markdown("### Encoder (LLM) Settings")
            with gr.Row():
                _cfg_w["enc_batch_dd"] = gr.Dropdown(label="Batch Size",
                                           choices=configure.BATCH_SIZE_CHOICES,
                                           value=cfg.get("encoder_batch_size", 512))
                _cfg_w["enc_ctx_dd"] = gr.Dropdown(label="Context Size",
                                         choices=configure.CTX_SIZE_CHOICES,
                                         value=cfg.get("encoder_ctx_size", 4096))

            with gr.Row():
                _cfg_w["enc_ngl_dd"] = gr.Dropdown(
                    label="GPU Layers",
                    choices=configure.GPU_LAYER_CHOICES,
                    value=cfg.get("encoder_gpu_layers", -1) if enc_is_vulkan else 0,
                    info=("(-1 = all layers)." if enc_is_vulkan
                          else "Encoder Backend is CPU — all layers run on CPU."),
                    interactive=enc_is_vulkan,
                )
                _cfg_w["enc_flash_chk"] = gr.Checkbox(
                    label="Flash Attention",
                    value=cfg.get("encoder_flash_attn", True),
                    info="Reduces VRAM usage for long contexts.",
                )

        # ── ImageGen settings ──
        with gr.Column(scale=2):
            gr.Markdown("### Image Generation Settings")
            with gr.Row():
                _cfg_w["img_clip_dd"] = gr.Dropdown(label="CLIP Skip",
                                          choices=configure.CLIP_SKIP_CHOICES,
                                          value=cfg.get("imagegen_clip_skip", 2))
            with gr.Row():
                # sd.cpp has no per-layer GPU offload for the diffuser (no
                # -ngl equivalent) — only whole-component placement, so this
                # is a 3-way choice rather than a layer-count dropdown. See
                # configure.DIFFUSER_PLACEMENT_CHOICES / parse_diffuser_placement().
                _cfg_w["img_placement_dd"] = gr.Dropdown(
                    label="Diffuser Placement",
                    choices=configure.DIFFUSER_PLACEMENT_CHOICES,
                    value=(cfg.get("imagegen_placement", configure.DIFFUSER_PLACEMENT_FULL_GPU)
                          if img_is_vulkan else configure.DIFFUSER_PLACEMENT_FULL_CPU),
                    info=("Split keeps the encoder+VAE on CPU, diffusion model on GPU."
                          if img_is_vulkan else
                          "ImageGen Backend is CPU — sd.cpp will not touch the GPU at all."),
                    interactive=img_is_vulkan,
                )

    # ── Prompt template ──
    gr.Markdown("### Advanced")
    _cfg_w["prompt_template_tb"] = gr.Textbox(
        label="Prompt Template",
        value=cfg.get("prompt_template",
                      "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
        lines=2,
    )

    # ── Save ──
    with gr.Row():
        _cfg_w["save_all_btn"]    = gr.Button("Save All Configuration", variant="primary", size="lg")
        _cfg_w["unload_btn"]      = gr.Button("Unload Models", variant="secondary", size="lg")

    # ── Events: browse & scan — no status output, wire immediately ──
    def _browse_encoder():
        p = _browse_file()
        return (p, Path(p).stem) if p else (gr.update(), gr.update())

    def _browse_diffusion():
        p = _browse_file()
        if p:
            stem = Path(p).stem
            vae_path, vae_name = _detect_vae(p)
            return p, stem, vae_path, vae_name
        return gr.update(), gr.update(), gr.update(), gr.update()

    _cfg_w["enc_browse_btn"].click(
        _browse_encoder,
        outputs=[_cfg_w["enc_path_tb"], _cfg_w["enc_name_tb"]]
    )
    _cfg_w["diff_browse_btn"].click(
        _browse_diffusion,
        outputs=[_cfg_w["diff_path_tb"], _cfg_w["diff_name_tb"],
                 _cfg_w["vae_path_tb"], _cfg_w["vae_name_tb"]]
    )

    # When diffusion path is changed manually, auto-detect VAE
    def _on_diff_path_change(path: str):
        vae_path, vae_name = _detect_vae(path)
        return vae_path, vae_name

    _cfg_w["diff_path_tb"].change(
        _on_diff_path_change,
        inputs=_cfg_w["diff_path_tb"],
        outputs=[_cfg_w["vae_path_tb"], _cfg_w["vae_name_tb"]]
    )

    # ── Keep GPU Layers / Diffuser Placement in sync with their backend
    # dropdowns. Switching a backend to CPU must force the dependent
    # control to its CPU value and lock it (0 layers / Full CPU placement);
    # switching back to Vulkan must restore the last value the user had
    # and re-enable editing. Without this, the dropdowns stayed interactive
    # and showed stale numbers no matter what backend was actually selected
    # — which is what made it look like the CPU selection "did nothing".
    _last_ngl_value: Dict[str, int] = {"v": cfg.get("encoder_gpu_layers", -1)}
    _last_placement_value: Dict[str, str] = {
        "v": cfg.get("imagegen_placement", configure.DIFFUSER_PLACEMENT_FULL_GPU)
    }

    def _on_enc_backend_change(backend_choice: str, current_ngl):
        if "Vulkan" in backend_choice:
            restore = _last_ngl_value["v"]
            return gr.update(value=restore, interactive=True,
                             info="(-1 = all layers).")
        # Remember the value the user had before forcing it to 0, so
        # switching back to Vulkan restores it instead of resetting to -1.
        if current_ngl is not None:
            try:
                _last_ngl_value["v"] = int(current_ngl)
            except (TypeError, ValueError):
                pass
        return gr.update(value=0, interactive=False,
                         info="Encoder Backend is CPU — all layers run on CPU.")

    _cfg_w["enc_backend_dd"].change(
        _on_enc_backend_change,
        inputs=[_cfg_w["enc_backend_dd"], _cfg_w["enc_ngl_dd"]],
        outputs=_cfg_w["enc_ngl_dd"],
    )

    def _on_img_backend_change(backend_choice: str, current_placement):
        if "Vulkan" in backend_choice:
            restore = _last_placement_value["v"]
            return gr.update(value=restore, interactive=True,
                             info="Split keeps the encoder+VAE on CPU, diffusion model on GPU.")
        if current_placement and current_placement != configure.DIFFUSER_PLACEMENT_FULL_CPU:
            _last_placement_value["v"] = current_placement
        return gr.update(value=configure.DIFFUSER_PLACEMENT_FULL_CPU, interactive=False,
                         info="ImageGen Backend is CPU — sd.cpp will not touch the GPU at all.")

    _cfg_w["img_backend_dd"].change(
        _on_img_backend_change,
        inputs=[_cfg_w["img_backend_dd"], _cfg_w["img_placement_dd"]],
        outputs=_cfg_w["img_placement_dd"],
    )


def _wire_config_events(status_box: gr.Textbox) -> None:
    """Register Configuration tab save event that outputs to shared status_box."""
    w = _cfg_w

    def save_all(ep, en, dp, dn, vp, vn,
                 enc_back, img_back, threads,
                 eb, ec, engl, ef,
                 ic, img_placement, pt):
        enc_parsed = configure.parse_backend_choice(enc_back)
        img_parsed = configure.parse_backend_choice(img_back)
        configure.update_persistent({
            "encoder_model_path":  ep,  "encoder_model_name":  en,
            "imagegen_model_path": dp,  "imagegen_model_name": dn,
            "vae_model_path":      vp,  "vae_model_name":      vn,
            "backend_encoder":     enc_back,
            "backend_imagegen":    img_back,
            "encoder_vulkan_device": enc_parsed["vulkan_device"],
            "imagegen_vulkan_device": img_parsed["vulkan_device"],
            "vulkan_device":         img_parsed["vulkan_device"],
            "encoder_threads":     int(threads),
            "imagegen_threads":    int(threads),
            "encoder_batch_size":  int(eb),
            "encoder_ctx_size":    int(ec),
            "encoder_gpu_layers":  int(engl),
            "encoder_flash_attn":  bool(ef),
            "imagegen_clip_skip":  int(ic),
            "imagegen_placement":  img_placement,
            "prompt_template":     pt,
            "first_run":           False,
        })
        return "All settings saved!"

    w["save_all_btn"].click(
        save_all,
        inputs=[
            w["enc_path_tb"], w["enc_name_tb"], w["diff_path_tb"], w["diff_name_tb"],
            w["vae_path_tb"], w["vae_name_tb"],
            w["enc_backend_dd"], w["img_backend_dd"], w["threads_dd"],
            w["enc_batch_dd"], w["enc_ctx_dd"],
            w["enc_ngl_dd"], w["enc_flash_chk"],
            w["img_clip_dd"], w["img_placement_dd"], w["prompt_template_tb"],
        ],
        outputs=status_box,
    )

    w["unload_btn"].click(
        lambda: (inference.unload_models(), "Models unloaded.")[1],
        outputs=status_box,
    )


# ---------------------------------------------------------------------------
# Tab 3 — Debug / Info
# ---------------------------------------------------------------------------

def _collect_debug() -> str:
    """
    Show the raw contents of data/constants.ini so the user
    can see exactly what was detected and written during installation.
    """
    try:
        width = 48
        L: List[str] = []
        
        # --- constants.ini ---
        L.append("=" * width)
        L.append("  CONSTANTS.INI")
        L.append("=" * width)
        L.append("")
        constants_path = configure.get_constants_path()
        if constants_path.exists():
            try:
                with open(constants_path, "r", encoding="utf-8") as _f:
                    L.append(_f.read())
            except Exception as _e:
                L.append(f"  (error reading constants.ini: {_e})")
        else:
            L.append(f"  (constants.ini not found at: {constants_path})")
        L.append("=" * width)
        return "\n".join(L)
    except Exception:
        return f"Error collecting debug info:\n{traceback.format_exc()}"


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


def _build_debug_tab_inner() -> gr.Textbox:
    """Build Debug tab widgets; info section above, debug info below."""
    with gr.Group():
        gr.Markdown("### Image-Gradio-Gguf")
        gr.HTML(
            "<p>A Windows local image generator using Gradio, llama.cpp and stable-diffusion.cpp, by "
            "<a href=\"mailto:wiseman-timelord@mail.com\">WiseMan-Time-Lord</a> at "
            "<a href=\"http://wisetime.rf.gd/\">WiseTime.Rf.Gd</a></p>"
            "<p><strong>Where you may find this and my other programming projects on </strong>"
            "<a href=\"https://github.com/wiseman-timelord\">GitHub</a></p>"
            "<p><strong>Support/Donate to assist in the continuation of my projects at, </strong>"
            "<a href=\"https://patreon.com/WiseManTimeLord\">Patreon</a>, "
            "<a href=\"https://ko-fi.com/WiseManTimeLord\">Ko-Fi</a></p>",
            elem_classes=["info-textbox-match"],
        )
        with gr.Row():
            _dbg["refresh_btn"] = gr.Button("Refresh", variant="primary")
            _dbg["copy_btn"]    = gr.Button("Copy to Clipboard")

        # Pre-populate with debug info (no need for app.load)
        _dbg["info_text"] = gr.Textbox(
            label="Debug Info",
            interactive=False,
            lines=14, max_lines=30,
            autoscroll=False,
            value=_collect_debug(),   # Call directly to show info on load
        )

        _dbg["refresh_btn"].click(_collect_debug, outputs=_dbg["info_text"])

    return _dbg["info_text"]


def _wire_debug_events(status_box: gr.Textbox) -> None:
    """Register Debug tab copy event that outputs to shared status_box."""
    _dbg["copy_btn"].click(
        _copy_to_clipboard, inputs=_dbg["info_text"], outputs=status_box
    )


def _render_recent_strip_standalone() -> str:
    """Module-level wrapper so app.load can reference it before _wire_generate_events runs."""
    paths = _get_recent_images()
    if not paths:
        return (
            '<div id="recent-strip-wrap">'
            '<div id="recent-strip-label">Recent Images</div>'
            '<div id="recent-strip"><span id="recent-empty">No images generated yet.</span></div>'
            '</div>'
        )
    thumbs = []
    for p in paths:
        url = Path(p).as_uri()
        fname = Path(p).name
        thumbs.append(
            f'<img class="recent-thumb" src="{url}" title="{fname}" />'
        )
    body = "\n".join(thumbs)
    return (
        '<div id="recent-strip-wrap">'
        '<div id="recent-strip-label">Recent Images</div>'
        f'<div id="recent-strip">{body}</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------

def build_app():
    """Assemble and return (app, css) for Gradio 6+.
    css must be passed to launch() rather than the Blocks constructor."""
    configure.ensure_data_dirs()

    _css = """
#exit-btn {
    min-height: 3.5rem !important;
    background: #a93226 !important;
    border-color: #922b21 !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}
#exit-btn:hover { background: #c0392b !important; }

#recent-strip-wrap {
    border: 1px solid var(--border-color-primary, #444);
    border-radius: 6px;
    padding: 6px 8px;
    background: var(--background-fill-secondary, #1a1a1a);
}
#recent-strip-label {
    font-size: 0.75rem;
    color: var(--body-text-color-subdued, #888);
    margin-bottom: 4px;
    user-select: none;
}
#recent-strip {
    display: flex;
    flex-direction: row;
    gap: 6px;
    overflow-x: auto;
    overflow-y: hidden;
    scroll-snap-type: x mandatory;
    -webkit-overflow-scrolling: touch;
    padding-bottom: 4px;
    min-height: 80px;
}
#recent-strip::-webkit-scrollbar { height: 5px; }
#recent-strip::-webkit-scrollbar-track { background: transparent; }
#recent-strip::-webkit-scrollbar-thumb { background: #555; border-radius: 3px; }
.recent-thumb {
    flex: 0 0 auto;
    scroll-snap-align: start;
    width: 80px;
    height: 80px;
    object-fit: cover;
    border-radius: 4px;
    cursor: pointer;
    border: 2px solid transparent;
    transition: border-color 0.15s, transform 0.1s;
}
.recent-thumb:hover { border-color: var(--color-accent, #888); transform: scale(1.05); }
.recent-thumb.selected { border-color: var(--button-primary-background-fill, #6e7bff); }
#recent-empty {
    font-size: 0.8rem;
    color: var(--body-text-color-subdued, #888);
    line-height: 80px;
    padding-left: 4px;
}
"""

    with gr.Blocks(title="Image-Gradio-Gguf") as app:
        gr.Markdown("# Image-Gradio-Gguf")

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs():
            with gr.TabItem("Generate"):
                _build_generate_tab_inner()

            with gr.TabItem("Configuration"):
                _build_config_tab_inner()

            with gr.TabItem("Debug / Info"):
                info_text = _build_debug_tab_inner()

        # ── Unified bottom bar (spans below all tabs) ─────────────────────────
        with gr.Row(elem_id="bottom-bar"):
            shared_status = gr.Textbox(
                value="Ready.",
                label=None,
                show_label=False,
                interactive=False,
                container=False,
                placeholder="Ready.",
                elem_id=configure.STATUS_BAR_KEY,
                scale=9,
            )
            exit_btn = gr.Button(
                "Exit Program",
                variant="stop",
                scale=1,
                elem_id="exit-btn",
                min_width=140,
            )

        exit_btn.click(lambda: os._exit(0), inputs=[], outputs=[])

        # Wire all per-tab events to the shared status box.
        _wire_generate_events(shared_status)
        _wire_config_events(shared_status)
        _wire_debug_events(shared_status)

        # Populate recent images strip on load
        app.load(_render_recent_strip_standalone, outputs=_gen["recent_strip"])

    return app, _css