#!/usr/bin/env python3
"""
launcher.py - Startup, shutdown, and main loop.
Activated from the venv by the batch menu (option 1).
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio is not installed.")
    print("Run option 2 (Installation) from the batch menu first.")
    sys.exit(1)

_major = int(gr.__version__.split(".")[0])
if _major < 5:
    print(f"WARNING: Gradio {gr.__version__} detected. This app targets Gradio 5.x.")

import scripts.configure as configure
import scripts.utilities as utilities
import scripts.display as display


def _print_banner() -> None:
    cpu = configure.get_cpu_info()
    vk  = configure.get_vulkan_info()
    bs  = utilities.get_build_status()
    cfg = configure.load_persistent()

    print(f"  Versioning: Python {platform.python_version()}; Gradio {gr.__version__}")
    print(f"\n  CPU     : {cpu['brand']}")
    print(f"  Threads : {cpu['default_threads']} (85% of {cpu['cores_logical']} logical cores)")
    print(f"  Vulkan  : {vk['available']}  ({vk['version']})")
    for d in vk["devices"]:
        print(f"    GPU{d['index']}: {d['name']}")

    if not bs["llama_built"] or not bs["sd_built"]:
        print("\n  NOTE: Backends not yet built.")
        print("        Run option 2 (Installation) from the batch menu.")
    else:
        print(f"\n  llama-cli : {bs['llama_path']}")
        print(f"  sd        : {bs['sd_path']}")

    enc  = cfg.get("encoder_model_path", "")
    diff = cfg.get("imagegen_model_path", "")
    vae  = cfg.get("vae_model_path", "")
    print(f"\n  Encoder  : {'OK — ' + enc   if enc  and Path(enc).exists()  else 'NOT SET'}")
    print(f"  Diffusion: {'OK — ' + diff  if diff and Path(diff).exists() else 'NOT SET'}")
    print(f"  VAE      : {'OK — ' + vae   if vae  and Path(vae).exists()  else 'NOT SET'}")
    print()

def main() -> None:
    configure.ensure_data_dirs()
    _print_banner()
    app, _css = display.build_app()

    # Suppress the Starlette deprecation warning from Gradio internals
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".*HTTP_422_UNPROCESSABLE_ENTITY.*"
    )

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
        css=_css,
    )

if __name__ == "__main__":
    main()