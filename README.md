# Image-Generator-Gguf
Status: Development/Alpha

### Description
A local python image generator from prompt using Qwen 3 Z-Image Engineer encoder and Z-Image Turbo. The program WiseMan-TimeLord probably should have done before doing other image based llm applications WiseMan-TimeLord have produced, this will be a simple image generation project, but eventually cover several encoders and image generation models.

### Media
- Program starts with hidden errors (~A005)...
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

C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\scripts\display.py:558: UserWarning: The parameters have been moved from the Blocks constructor to the launch() method in Gradio 6.0: theme. Please pass these parameters to launch() instead.
  with gr.Blocks(title="Image Generator GGUF",
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\venv\Lib\site-packages\gradio\routes.py:1379: StarletteDeprecationWarning: 'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated. Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead.
  return await queue_join_helper(body, request, username)
C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\venv\Lib\site-packages\gradio\blocks.py:1938: UserWarning: A function (_collect_debug) returned too many output values (needed: 0, returned: 1). Ignoring extra values.
    Output components:
        []
    Output values returned:
        ["========================================================================
  DEBUG REPORT  2026-06-17 09:26:52
========================================================================

Python  : 3.12.4
Gradio  : 6.18.0
Platform: Windows-10-10.0.19045-SP0

CPU     : AMD Ryzen 9 3900X 12-Core Processor  [AMD]  24 threads
Default : 21 threads (85%)
AVX2:True  F16C:True  FMA:True  AVX512:False  AOCL:False

RAM     : 13205 / 65460 MB  (20%)

Vulkan  : True  ver=1.4.341
SDK     : C:\Program Files\VulkanSDK\1.4.341.1
  GPU0: NVIDIA GeForce GTX 1060 3GB
  GPU1: Radeon (TM) RX 470 Graphics

llama.cpp : built  C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\data\llama_cpp_binaries\llama-cli.exe
sd.cpp    : built  C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\data\stable_diffusion_binaries\sd-cli.exe

llama-cli : C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\data\llama_cpp_binaries\llama-cli.exe
sd exe    : C:\Inference_Files\Image-Generator-Gguf\Image-Generator-Gguf\data\stable_diffusion_binaries\sd-cli.exe

Encoder  : NOT SET
Diffusion: NOT SET
VAE      : NOT SET

Enc backend : CPU
Img backend : CPU
Enc threads : 21
Img threads : 21
Size        : 512×512
Steps       : 4  CFG: 1.0  Sampler: euler_a

Environment:
  NUMBER_OF_PROCESSORS = 24
  PATH = C:\Windows\System32;C:\Windows\System32\Wbem;C:\Network_Files\4allDownloader&Converter\bin;C:\Program Files\Python311\Scripts\;C:\Program Files\Python311\;C:\Program Files\VulkanSDK\1.4.341.1\Bin;C:\P...
  PROCESSOR_ARCHITECTURE = AMD64
  VULKAN_SDK = C:\Program Files\VulkanSDK\1.4.341.1

========================================================================"]
  warnings.warn(

```
- Installer is looking nice now (~A003)...
```
  ==============================================================================
      Image-Generator-Gguf — Install Method
  ==============================================================================



  System Detections...
     Platform: Windows 10; Python 3.12.4
     Build Tools: Git OK; CMake OK
     Architecture: SSE, SSE2, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, F16C, FMA
     Hardware: CPUs 24; GPUs 0, 1; Vulkan 1.4.341


  -------------------------------------------------------------------------------


     1. Clean Install (Purge First)

     2. Check/Install (Fix Missing Packages/Libraries)

     3. Refresh Configs (Only Remake Ini/Json)



  ===============================================================================
  Selection; Menu Options = 1-3, Abandon Install = A:


```

### Requirements:
- We will be programming towards Python ~3.12 and windows 10 22h2.
- Vulkan 1.3 card is available with Vulkan 1.4 installed, but its on gpu 1 not gpu 0. On gpu 1 there is =>8GB. we are not using GPU 0, GPU 0 is for the monitors. When compiling for Vulkan we will also ensure to include cpu optimizations too, so its optimized for, Vulkan, F16C, AVX, AVX2, FMA, where possible.
- CPU is aimed at zen 2 with AOCL installed. if cannot load to Vulkan. We compile CPU libraries/packages for F16C, AVX, AVX2, FMA, where possible. CPU will always use by default 85% threads on the CPU, where multi-thread will enhance performance during significant phases. Installer should detect the number of threads/cores during installation, and write this down to a key in ".\data\constants.ini". .
- Libraries for encoder, possibly we could have llama.cpp vulkan, and stick that on the 8GB of vram on the passive secondary rx 470 (this is not hip, so stick to vulkan). 
- Libraries for image generation, Stable Diffusion on CPU, if not able to be done on vulkan. If vulkan is an option then put the image generation model on the GPU, and have the encoder on the CPU instead.
- Arcitecture features detected/supported now includes, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX512, F16C, FMA.

### Models...
I put Q# because it should support any quantization, the model variety will be expanded upon later....
- intended Encoding model, "Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q#.gguf" and "Qwen3-4b-Uncensored-Z-Image-Engineer-V4-Q#.gguf
- intended Image generation model "z_image_turbo-Q#.gguf" and "ae.safetensors". (again this should cover all quantizations)

### Structure:
- Initial plan...
```
.\Image-Generator-Gguf.bat  (name of program)
.\launcher.py
.\installer.py
.\scripts\ (folder containing other required scripts, all having one word logical relating name with 6-10 letters)
.\data\constants.ini  (any constants the main program needs to be aware of from install)
.\data\persistent.json   (all variables reqired for the configuration page).
.\output\   (folder with output images)
.\models\   (default folder for models, though it should handle it if they are not there. The user is expected to set where the models are in the configuration page when the program loads).
```
...current result...
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
- Completing installer.
- Implementing PyQt (???), for built-in browser, for fake application style interface.
- Rename to "Image-Gradio-Gguf"
- Program needs to install and load without issus.
- Test UI and json loading/saving.
- Test inference.
- Add edit image feature.

### Design
- Page 1 the Interaction page- it will have a text box with a generate button underneath, to the side of that will be configurations for image generation, with dropdown list for reasonable values and sensible default settings and sensible ranges in the lists.
- Page 2 the Configuration page - Where the user configures, model location used for, 1. encoding and 2. image generation, these would require display of current path and a browse button, Additionally page 2 would have whatever configurations are required for these models, individually, with dropdown list for reasonable values and sensible default settings and sensible ranges in the lists.
- Page 3 the Debug/Info page - showing useful values, that actually change, not constants, ensure its all in a text box too, with a little copy button so I could paste it back during development.
