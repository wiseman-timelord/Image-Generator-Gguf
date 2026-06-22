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
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])

import scripts.configure as configure
import scripts.inference as inference


# ---------------------------------------------------------------------------
# Exit handling
# ---------------------------------------------------------------------------
# The "Exit Program" button click runs on a Gradio/Starlette worker thread,
# not the Qt GUI thread launcher.py creates. launcher.py registers a
# thread-safe handler here (via set_exit_handler) that signals the Qt
# window to close, so window-geometry saving and shutdown all happen in
# one place (launcher.py's _shutdown()). If nothing registers a handler
# (e.g. display.py is ever driven without launcher.py), fall back to a
# plain process exit so the button still works.
_exit_handler: Optional[Any] = None


def set_exit_handler(handler) -> None:
    """Register the function to call when 'Exit Program' is clicked.
    Called once by launcher.py during startup."""
    global _exit_handler
    _exit_handler = handler


def _handle_exit_click() -> None:
    if _exit_handler is not None:
        _exit_handler()
    else:
        os._exit(0)


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
        
        # Determine the initial directory for the file dialog
        cfg = configure.load_persistent()
        last_dir = cfg.get("last_model_browse_dir", "")
        initial_dir = str(configure.get_models_dir())
        
        if last_dir:
            # Resolve the path (handles relative paths like ".\models")
            p = Path(last_dir)
            if not p.is_absolute():
                p = configure._get_project_root() / p
            if p.exists() and p.is_dir():
                initial_dir = str(p)
                
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[
                ("Model files", "*.gguf *.safetensors"),
                ("GGUF",        "*.gguf"),
                ("Safetensors", "*.safetensors"),
                ("All files",   "*.*"),
            ],
        )
        root.destroy()
        
        # If a file was selected, save its directory for next time
        if path:
            selected_dir = str(Path(path).parent)
            configure.update_persistent({"last_model_browse_dir": selected_dir})
            return path
            
        return ""
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

# Module-level cache for recent images – stores the full list, not limited
_gallery_cache: Dict[str, Any] = {
    "full_list": None,      # list of all image paths (unlimited)
    "mtime": 0.0,
}

def _get_recent_images(max_images: int = 50) -> List[str]:
    """Return paths of images in the output folder, newest first.
       Caches the full list of images and slices it according to max_images.
       Prints a concise start/end summary only when a rescan occurs.
    """
    out_dir = configure.get_output_dir()
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    try:
        current_mtime = out_dir.stat().st_mtime if out_dir.exists() else 0.0
        # If the directory hasn't changed, reuse the cached full list
        if (_gallery_cache["full_list"] is not None and
            abs(current_mtime - _gallery_cache["mtime"]) < 0.1):
            full = _gallery_cache["full_list"]
            # Return the first max_images entries (sliced)
            return full[:max_images]

        # Directory changed (or first run) – perform a fresh scan
        print("[gallery] Scanning for Thumbnails....", flush=True)
        files = [
            f for f in out_dir.iterdir()
            if f.is_file() and f.suffix.lower() in exts
        ]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        full_list = [str(f) for f in files]   # store all, no limit
        count = len(full_list)
        print(f"[gallery] Rescanned {out_dir}: {count} image{'s' if count != 1 else ''}", flush=True)
        print()
       
        # Update cache
        _gallery_cache["full_list"] = full_list
        _gallery_cache["mtime"] = current_mtime
        return full_list[:max_images]
    except Exception as e:
        print(f"[gallery] rescan FAILED: {e}", flush=True)
        return []

# ---------------------------------------------------------------------------
# Preview box status images (.\media\program_*.jpg)
# ---------------------------------------------------------------------------

def _status_image(name: str) -> Optional[str]:
    """Return str path to a media/program_<name>.jpg, or None if missing."""
    p = configure.get_media_dir() / f"program_{name}.jpg"
    return str(p) if p.exists() else None


def _idle_preview_image() -> Optional[str]:
    """Preview box image when no generation is running: the newest output
    image if one exists, otherwise the 'no media' placeholder."""
    recent = _get_recent_images(max_images=1)
    if recent:
        return recent[0]
    return _status_image("no_media")


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
                label="Positive Prompt",
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
                    label="Quality Preset", choices=list(presets.keys()), value="Fast (Turbo)",
                )
                _gen["sampler_dd"] = gr.Dropdown(
                    label="Sampler Type", choices=list(configure.SAMPLER_MAP.keys()),
                    value=cfg.get("imagegen_sampling", "euler_a"),
                )

            with gr.Row():
                _gen["batch_dd"] = gr.Dropdown(label="Batch Count",
                                               choices=configure.BATCH_COUNT_CHOICES,
                                               value=cfg.get("imagegen_batch_count", 1)
                )
                _gen["width_dd"]  = gr.Dropdown(label="Image Width",  choices=configure.IMAGE_SIZES,
                                                value=cfg.get("imagegen_width", 512))
                _gen["height_dd"] = gr.Dropdown(label="Image Height", choices=configure.IMAGE_SIZES,
                                                value=cfg.get("imagegen_height", 512))
                _gen["output_fmt_dd"] = gr.Dropdown(
                    label="Output Format", choices=configure.OUTPUT_FORMATS,
                    value=cfg.get("output_format", "png"),
                )

            with gr.Row():
                _gen["steps_dd"] = gr.Dropdown(
                    label="Diffuse Steps", choices=configure.STEP_CHOICES,
                    value=cfg.get("imagegen_steps", 4),
                )
                _gen["cfg_scale_sld"] = gr.Slider(
                    label="CFG Scale", minimum=0.5, maximum=20.0, step=0.5,
                    value=cfg.get("imagegen_cfg_scale", 1.0),
                )
                _gen["seed_num"] = gr.Number(label="Gen Seed (-1 = random)",
                                             value=cfg.get("imagegen_seed", -1), precision=0)

            # NOTE: no manual "Save as Default" button — successful
            # generations auto-save their settings panel values (see
            # on_generate_click / do_generate's success branch), so the
            # next launch picks up the last settings that actually worked.

        # ── Right column: image preview only ────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Output")
            with gr.Row(visible=configured) as _gen["generate_row"]:
                # Single dynamic button: shows "Generate" (primary) when idle
                # and switches to "..Please Wait.." (disabled) while a
                # generation is running. do_generate() flips it to Please Wait
                # on entry and back to Generate on its final yield.
                _gen["generate_btn"] = gr.Button("Generate Image", variant="primary", size="lg")

            # Single currently-selected/in-progress image — the ONLY image
            # shown here is either the most recent generation, the live
            # encoding/diffusion phase status image, or whatever the user
            # has clicked in the gallery below. Never shows Gradio's
            # built-in progress bar.
            # Gradio 6 replaced show_download_button/show_share_button with a
            # single buttons=[...] list, so build the kwarg for whichever
            # major version is actually installed.
            _img_button_kwargs = (
                {"buttons": ["download"]} if _GRADIO_MAJOR >= 6
                else {"show_download_button": True, "show_share_button": False}
            )
            _gen["preview_img"] = gr.Image(
                label="Image Preview",
                type="filepath",
                value=_idle_preview_image(),
                height=configure.PREVIEW_IMAGE_HEIGHT,
                interactive=False,
                show_label=True,
                container=True,
                elem_id="preview-img",
                **_img_button_kwargs,
            )

    # ── Gallery: full width, underneath BOTH columns ─────────────────────────
    # Its only job is to show thumbnails of everything in .\output, populated
    # solely via full rescans (_get_recent_images), never a per-call image
    # list. Clicking a thumbnail here updates the preview box above — it does
    # not show generation progress, and it is not itself the preview.
    gr.Markdown("### Thumbnails Gallery")
    _gen["output_gallery"] = gr.Gallery(
        label="Generated Images",
        value=_get_recent_images(),
        columns=16, rows=1,
        height=123,
        object_fit="contain",
        allow_preview=False,
        show_label=False,
        fit_columns=False,
        elem_id="output-gallery",
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

    def do_generate(prompt, negative, width, height, steps, sampler,
    cfg_scale, seed, batch, output_format):
        """
        Generator: yields (preview_img, gallery, status, btn_update) tuples
        so the preview box can switch between program_encoding.jpg /
        program_diffusion.jpg while generation runs, WITHOUT using Gradio's
        built-in progress bar anywhere (gallery, preview, or otherwise).
        btn_update flips the single Generate/Please Wait button to its
        "..Please Wait.." appearance for the duration of the run and back
        to "Generate" on the final yield (including early-exit validation
        failures, which never started a run and so should leave the button
        as Generate). Final yield swaps the preview to the finished image
        and rescans .\\output into the gallery — the gallery never receives
        a per-call image list, only full folder rescans.
        """
        gallery_now  = _get_recent_images()
        preview_now  = _idle_preview_image()
        _btn_generate = gr.update(value="Generate Image", variant="primary", interactive=True)
        _btn_wait     = gr.update(value="..Please Wait..", variant="primary", interactive=False)

        if not prompt or not prompt.strip():
            yield preview_now, gallery_now, "Please enter a prompt.", _btn_generate
            return
        c = _cfg()
        if not c.get("imagegen_model_path") or not Path(c["imagegen_model_path"]).exists():
            yield preview_now, gallery_now, "Image generation model not configured. Go to Configuration tab.", _btn_generate
            return
        if not c.get("vae_model_path") or not Path(c["vae_model_path"]).exists():
            yield preview_now, gallery_now, "VAE model not configured. Go to Configuration tab.", _btn_generate
            return

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

        configure.APP_STATE["cancel_requested"] = False

        # ── Run generation on a worker thread; main thread yields preview
        #    + status updates based on phase, polled from a shared mutable
        #    holder ("_phase"). The status string shows which of the two
        #    phases is active (1/2 Encoding, 2/2 Diffusing) plus a live
        #    timer for that phase. Once configure.TIMING_STATS has data from
        #    a prior generation this session, the timer becomes an ETA
        #    countdown-style display ("~Ns left"); until then it just counts
        #    up, since there's nothing to estimate against yet. ──
        _batch_count = int(batch)
        _phase: Dict[str, Any] = {
            "name": "encoding", "result": None, "done": False,
            "phase_start": time.time(), "step": 0, "total_steps": 0,
            "batch_current": 1, "batch_total": _batch_count,
            "last_step_seen": 0,
        }

        def prog_cb(msg: str, pct: float, info: Dict[str, Any] = None):
            info = info or {}
            phase = info.get("phase")
            if phase in ("encoding", "diffusion"):
                _phase["name"] = phase
            else:
                # Fallback for any caller that didn't pass phase info.
                m = msg.lower()
                if "enhanc" in m or "encod" in m:
                    _phase["name"] = "encoding"
                elif "generat" in m or "step" in m or "%" in m:
                    _phase["name"] = "diffusion"
            if "phase_start" in info:
                _phase["phase_start"] = info["phase_start"]
            if "step" in info:
                new_step = info["step"]
                # Detect when the step counter resets (new image in batch):
                # a step value of 1 arriving after we've already seen a
                # higher step means sd.cpp has moved on to the next image.
                if (new_step == 1 and _phase["last_step_seen"] > 1
                        and _phase["batch_current"] < _phase["batch_total"]):
                    _phase["batch_current"] += 1
                _phase["last_step_seen"] = new_step
                _phase["step"] = new_step
            if "total_steps" in info:
                _phase["total_steps"] = info["total_steps"]

        def _format_status() -> str:
            """Build the 'Batch Number X/Y; Generate Stage N/2; ... Phase
            {step}/{total}...###s (prev_batch_Ns)' status string. Seconds are
            always whole numbers (no split seconds) per the fixed status-bar
            format — never decimals."""
            name = _phase["name"]
            elapsed_s = int(time.time() - _phase["phase_start"])
            batch_cur = _phase["batch_current"]
            batch_tot = _phase["batch_total"]
            batch_prefix = f"Batch Number {batch_cur}/{batch_tot}; "

            # Previous batch elapsed suffix — only shown when we have a
            # recorded time from a completed batch earlier this session.
            prev_elapsed = configure.APP_STATE.get("last_batch_elapsed_seconds", 0)
            prev_suffix = f" ({int(prev_elapsed)}s)" if prev_elapsed else ""

            if name == "encoding":
                # enhance_prompt() (inference.py) is a single-shot llama-cli
                # call with no per-token step/total reported back through
                # progress_callback, so step/total are only ever populated
                # once a step-aware encoder backend supplies them. Until
                # then this degrades to a plain running timer rather than
                # showing a fabricated "0/0".
                step = _phase.get("step", 0)
                total = _phase.get("total_steps", 0)
                step_part = f" {step}/{total}" if total else ""
                return (f"{batch_prefix}Generate Stage 1/2; Encoding Phase{step_part}..."
                        f"{elapsed_s}s{prev_suffix}")

            # diffusion
            step = _phase.get("step", 0)
            total = _phase.get("total_steps", 0) or int(gen_cfg.get("imagegen_steps", 0))
            step_part = f" {step}/{total}" if total else ""
            return (f"{batch_prefix}Generate Stage 2/2; Diffusing Phase{step_part}..."
                    f"{elapsed_s}s{prev_suffix}")

        def worker():
            try:
                _phase["result"] = inference.generate_image(
                    prompt.strip(), gen_cfg, progress_callback=prog_cb)
            except Exception as e:
                _phase["result"] = {"success": False, "output_path": "",
                                    "message": f"Error: {e}"}
            finally:
                _phase["done"] = True

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        last_shown_img = None
        last_shown_second = -1
        first_tick = True
        # Button switches to "..Please Wait.." the moment the worker thread starts.
        # Only send that update on the FIRST poll tick — re-sending the same
        # gr.update() on every 0.15s tick forces Gradio to re-render the
        # button node repeatedly, which was cascading into a layout
        # recalculation of sibling nodes in the same column (incl. the
        # preview image) and intermittently knocking out its object-fit
        # CSS override. Once the button is already showing "..Please Wait..", later
        # ticks pass a true no-op (gr.update()) for it.
        #
        # The status string itself is also throttled to once per whole
        # second (last_shown_second), independent of the 0.15s poll
        # cadence — the timer display only ever shows whole seconds, so
        # there is no reason to push a new status string more than once a
        # second even though we keep polling faster for image/button
        # responsiveness.
        while not _phase["done"]:
            img = _status_image(_phase["name"])
            current_second = int(time.time() - _phase["phase_start"])
            btn_update = _btn_wait if first_tick else gr.update()
            first_tick = False

            status_update = gr.update()
            if current_second != last_shown_second:
                last_shown_second = current_second
                status_update = _format_status()

            if img and img != last_shown_img:
                last_shown_img = img
                yield img, gr.update(), status_update, btn_update
            else:
                yield gr.update(), gr.update(), status_update, btn_update
            time.sleep(0.15)
        t.join()

        # Start inactivity timer now that generation is finished
        _reset_inactivity_timer()

        result = _phase["result"] or {"success": False, "message": "Unknown error"}

        if result.get("success") and result.get("output_path"):
            out_path = Path(result["output_path"])
            try:
                sz = out_path.stat().st_size
                print(f"[generate] output file: {out_path}  ({sz} bytes)", flush=True)
            except Exception as e:
                print(f"[generate] output file STAT FAILED: {out_path}  {e}", flush=True)
            # Record the total batch elapsed time so the next generation can
            # display it as a reference in the status bar (previous batch time).
            batch_elapsed = result.get("elapsed_seconds", 0.0)
            configure.APP_STATE["last_batch_elapsed_seconds"] = int(round(batch_elapsed))
            msg = (f"{result['message']} | Seed: {result['seed_used']} "
                   f"| Time: {int(round(result['elapsed_seconds']))}s")
            new_gallery = _get_recent_images()
            yield str(out_path), new_gallery, msg, _btn_generate
            # Auto-save the settings that just successfully produced an
            # image (fix #3 — replaces the old manual "Save as Default"
            # button). Only on success, so a failed/cancelled run never
            # overwrites the last known-good settings.
            configure.update_persistent({
                "imagegen_width": int(width), "imagegen_height": int(height),
                "imagegen_steps": int(steps), "imagegen_sampling": sampler,
                "imagegen_cfg_scale": float(cfg_scale),
                "imagegen_seed": int(seed), "imagegen_batch_count": int(batch),
                "negative_prompt": negative,
                "output_format": output_format,
            })
        else:
            new_gallery = _get_recent_images()
            yield _idle_preview_image(), new_gallery, result.get("message", "Unknown error"), _btn_generate


    def on_generate_click(prompt, negative, width, height, steps, sampler,
    cfg_scale, seed, batch, output_format):
        """Dispatch a click on the single dynamic button. The button reads
        "Generate" when idle and starts a run (delegating to the do_generate
        generator, which yields its own button-state updates). While a run
        is in progress the button is disabled and shows "..Please Wait..",
        preventing concurrent runs."""
        yield from do_generate(prompt, negative, width, height, steps, sampler,
    cfg_scale, seed, batch, output_format)

    _gen["generate_btn"].click(
        on_generate_click,
        inputs=[_gen["prompt_tb"], _gen["negative_tb"],
        _gen["width_dd"], _gen["height_dd"], _gen["steps_dd"],
        _gen["sampler_dd"], _gen["cfg_scale_sld"],
        _gen["seed_num"], _gen["batch_dd"], _gen["output_fmt_dd"]],
        outputs=[_gen["preview_img"], _gen["output_gallery"], status_box,
        _gen["generate_btn"]],
    )

    def on_gallery_select(evt: gr.SelectData):
        """When the user clicks a thumbnail, display it in the preview box."""
        if evt.value and isinstance(evt.value, dict):
            path = evt.value.get("image", {}).get("path") or evt.value.get("path")
            if path:
                return path
        elif evt.value and isinstance(evt.value, str):
            return evt.value
        return gr.update()

    _gen["output_gallery"].select(
        on_gallery_select,
        inputs=None,
        outputs=_gen["preview_img"],
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
    _cfg_w["enc_path_tb"]  = gr.Textbox(value=cfg.get("encoder_model_path", ""),  visible=False)
    _cfg_w["diff_path_tb"] = gr.Textbox(value=cfg.get("imagegen_model_path", ""), visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                _cfg_w["enc_name_tb"] = gr.Textbox(
                    label="Encoder Name",
                    value=cfg.get("encoder_model_name", ""),
                    placeholder="Qwen3-4b-Z-Image-Turbo",
                    interactive=True,
                    scale=8,
                )
                _cfg_w["enc_browse_btn"] = gr.Button("Browse...", size="sm", scale=1, min_width=90)

        with gr.Column(scale=1):
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
            "(Cpu-only install). "
        )
    else:
        gr.Markdown(
            "(Vulkan install). "
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

/* ── Preview box: the box height is driven by configure.PREVIEW_IMAGE_HEIGHT
(see the __PREVIEW_IMG_HEIGHT__px placeholder below, substituted once at the
end of this string) — the SAME value passed to the gr.Image(height=...)
kwarg above, so the two can never disagree. gr.Image in Gradio 6.19.0 has
NO object_fit kwarg (only gr.Gallery does), so fit-to-box behavior must be
driven entirely by CSS here. Gradio's own native rule is `.image-frame img
{ width:100%; height:100%; object-fit:scale-down }` — scale-down only ever
shrinks an oversized image, it never enlarges one smaller than the box
(e.g. a 256x256 generation inside this box), which is why small images
rendered tiny. We replicate the same width/height:100% sizing (required for
object-fit to have any box to fit against) but swap in object-fit:contain
so it scales BOTH directions — shrinking large images and enlarging small
ones, always preserving aspect ratio. The Svelte scope hash on Gradio's own
rule can out-rank generic selectors depending on load order, so target the
structural classes directly with #id + class stacking to win the cascade
regardless of load order. ─────────────────────────────────────────────── */
#preview-img.gradio-container,
#preview-img,
#preview-img .image-container,
#preview-img .image-container.svelte-12vrxzd,
#preview-img .image-frame,
#preview-img .image-frame.svelte-12vrxzd {
height: __PREVIEW_IMG_HEIGHT__px !important;
max-height: __PREVIEW_IMG_HEIGHT__px !important;
}
#preview-img img,
#preview-img .image-frame img,
#preview-img .image-frame.svelte-12vrxzd img {
width: 100% !important;
height: 100% !important;
object-fit: contain !important;
}

/* ── Gallery Thumbnails: force contain to prevent clipping ───────────────
Gradio 6's grid gallery renders each cell as <div class="thumbnail-lg ...">
  <img> inside, with object-fit driven by a `--object-fit` CSS variable that
  Gradio sets inline on .grid-container. That variable mechanism is fragile
  (relies on the object_fit= prop reaching the right node), and the rule
  that consumes it carries a Svelte scope hash (e.g. .svelte-7anmrz) which
  changes between Gradio versions and can out-rank generic selectors. So we
  target the real structural class directly with #id + class stacking and
  !important, the same approach used for #preview-img above — this wins
  the cascade regardless of the scope hash or load order, and no longer
  depends on the --object-fit variable being set correctly at all. We also
  keep the old .gallery-item/.grid-wrap/.thumbnail-item selectors as a
  harmless fallback in case a future Gradio version reintroduces them. ── */
#output-gallery .thumbnail-lg > img,
#output-gallery .thumbnail-lg img,
#output-gallery .grid-wrap img,
#output-gallery .gallery-item img,
#output-gallery .thumbnail-item img,
#output-gallery img {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    max-height: 100% !important;
}
#output-gallery .thumbnail-lg,
#output-gallery .gallery-item,
#output-gallery .grid-wrap .gallery-item {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow-y: hidden !important;
    overflow-x: auto !important;   /* keep horizontal scroll if needed */
    background: var(--background-fill-secondary) !important;
}

/* ── Gallery: kill the dead vertical scrollbar on the right ──────────────
The scrollbar the user sees is NOT on the individual .gallery-item cells
(those were already overflow-y:hidden above) — it's on .grid-wrap itself,
which is the actual scrolling viewport Gradio sizes to the `height=` prop
passed to gr.Gallery(). With rows=1 the row content fits inside that
height, so the vertical scrollbar that still appears is just an empty,
non-functional track. We disable vertical scroll on the viewport directly
and hide its scrollbar cross-browser, while still allowing horizontal
scroll/wrap behavior to pass through to the row of thumbnails. ── */
#output-gallery .grid-wrap {
    overflow-y: hidden !important;
    scrollbar-width: none !important;      /* Firefox */
    -ms-overflow-style: none !important;   /* old Edge/IE */
}
#output-gallery .grid-wrap::-webkit-scrollbar {
    display: none !important;              /* Chrome/Edge/Safari */
    width: 0 !important;
    height: 0 !important;
}
"""
    # Substitute the preview-box height placeholder with the single shared
    # constant (configure.PREVIEW_IMAGE_HEIGHT) — same value used for the
    # gr.Image(height=...) kwarg, so the two can never drift apart again.
    _css = _css.replace("__PREVIEW_IMG_HEIGHT__", str(configure.PREVIEW_IMAGE_HEIGHT))

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

        exit_btn.click(_handle_exit_click, inputs=[], outputs=[])

        # Wire all per-tab events to the shared status box.
        _wire_generate_events(shared_status)
        _wire_config_events(shared_status)
        _wire_debug_events(shared_status)

    return app, _css