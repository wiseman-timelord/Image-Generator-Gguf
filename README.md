![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/banner_llama.jpg)
# Image-Gradio-Gguf
Status: Development/Alpha

### Description
A local python image generator from prompt using Qwen 3 Z-Image Engineer encoder and Z-Image Turbo. The program WiseMan-TimeLord probably should have done before doing other image based llm applications WiseMan-TimeLord have produced, this will be a simple image generation project, but eventually cover several encoders and image generation models.

### Media
- Program starts something like this (A005)...
```

  Starting application at http://127.0.0.1:7860


  ============================================================
    Image Generator GGUF
    Python 3.12.4  |  Gradio 6.18.0
  ============================================================

  CPU     : AMD Ryzen 9 3900X 12-Core Processor
  Threads : 21 (85% of 24 logical cores)
  Vulkan  : True  (1.4.341)
    GPU0: NVIDIA GeForce GTX 1060 3GB
    GPU1: Radeon (TM) RX 470 Graphics

  llama-cli : C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\data\llama_cpp_binaries\llama-cli.exe
  sd        : C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\data\stable_diffusion_binaries\sd-cli.exe

  Encoder  : NOT SET
  Diffusion: NOT SET
  VAE      : NOT SET

* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```
- Installer script has Check/Reinstall feature, incase issues during install (A007)...
```
  ==============================================================================
      Image-Generator-Gguf — Installation
  ==============================================================================

  constants.ini written → C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\data\constants.ini

  Python virtual environment...
  -----------------------------
  venv already exists at C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\venv

  Python dependencies...
  ----------------------
  Upgrading pip inside venv...
    Installing gradio>=5.0...
    gradio>=5.0 OK
    Installing Pillow>=10.0...
    Pillow>=10.0 OK
    Installing numpy>=1.26...
    numpy>=1.26 OK
  All packages installed OK.

  Backend compile  (Vulkan)  —  llama.cpp + stable-diffusion.cpp...
  -----------------------------------------------------------------
  llama.cpp...
    llama-cli.exe already present, skipping clone and build.
    llama.cpp  →  success (binary already present)

  stable-diffusion.cpp...
    sd-cli.exe (or sd.exe) already present, skipping clone and build.
    stable-diffusion.cpp  →  success (binary already present)

  Installation summary
  --------------------
  Time elapsed : 4.4s
  constants.ini: C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\data\constants.ini
  persistent   : C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\data\persistent.json
  venv         : C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\venv
  llama bins   : C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\data\llama_cpp_binaries
  sd bins      : C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\data\stable_diffusion_binaries

  Press Enter to return to the batch menu...
```

### Requirements:
- Platform - Programming towards and testing upon, Windows 10 22h2 with Python 3.12, it may work on Windows 11 and nearby versions of Python. T.B.A.
- Graphics- Programming towards and testing upon, Vulkan 1.3 card with Vulkan 1.4 driver installed. Additionally CPU arcitecture features will be detected used when building for Vulkan including list shown below.
- Processor - Detecting/supporting, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX512, F16C, FMA, architecture features. Testing upon Zen 2 with AOCL installed. CPU will always use by default 85% threads on the CPU.
- Libraries - Required libraries is handled by the installer script, but they include both, Llama.Cpp and Stable Diffusion.

### Models...
I put Q# because it should support any quantization, the model variety will be expanded upon later....
- intended Encoding model, "Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q#.gguf" and "Qwen3-4b-Uncensored-Z-Image-Engineer-V4-Q#.gguf
- intended Image generation model "z_image_turbo-Q#.gguf" and "ae.safetensors". (again this should cover all quantizations)

### Structure:
- Current plan for sripts is...
```
Image-Generator-Gguf/
├── Image-Generator-Gguf.bat      # Windows launcher batch file
├── launcher.py                    # Startup, Shutdown, Main Loop.
├── installer.py                   # Download, Setup & Build, Install, creation/rectiation of json, creation/recreation of constants.ini.
├── data/
│   ├── constants.ini              # System constants & defaults
│   └── persistent.json            # User configuration (auto-managed)
├── scripts/                       # All scripts have 6-10 letter names
│   ├── __init__.py                # Empty init file.
│   ├── configure.py                # Configuration, and all global variables/constants/maps/lists are here.
│   ├── display.py                  # Gradio, Browser, Python Displays.
│   ├── inference.py                  # Image generation, Model Handling, Text Generation.
│   ├── utilities.py                  # General code, and code that is not more appropriate to be in other scripts.
├── models/                        # Default model directory
└── output/                        # Generated images directory
```

### Development:
A small program in python with gradio 5...
- Fixing compiling in installer.
- Implementing PyQt (???), for built-in browser, for fake application style interface.
- Rename to "Image-Gradio-Gguf". Uncertain.
- Program needs to install and load without issus.
- Test inference.
- Add edit image feature.

### Design
- Page 1 the Interaction page- it will have a text box with a generate button underneath, to the side of that will be configurations for image generation, with dropdown list for reasonable values and sensible default settings and sensible ranges in the lists.
- Page 2 the Configuration page - Where the user configures, model location used for, 1. encoding and 2. image generation, these would require display of current path and a browse button, Additionally page 2 would have whatever configurations are required for these models, individually, with dropdown list for reasonable values and sensible default settings and sensible ranges in the lists.
- Page 3 the Debug/Info page - showing useful values, that actually change, not constants, ensure its all in a text box too, with a little copy button so I could paste it back during development.
