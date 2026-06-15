# Image-Generator-Gguf
Status: Development/Alpha

### Description
A local python image generator from prompt using Qwen 3 Z-Image Engineer encoder and Z-Image Turbo. The program WiseMan-TimeLord probably should have done before doing other image based llm applications WiseMan-TimeLord have produced, this will be a simple image generation project, but eventually cover several encoders and image generation models.

### Requirements:
- We will be programming to whatever is optimal between Python 3.11-3.13, and windows 10 22h2.
- Vulkan 1.3 card is available with Vulkan 1.4 installed, but its on gpu 1 not gpu 0. On gpu 1 there is 8GB. we are not using GPU 0, gpu 1 is for the monitors. When compiling for Vulkan we will also ensure to include cpu optimizations too, so its optimized for, Vulkan, F16C, AVX, AVX2, FMA, if possible.
- CPU is aimed at zen 2 with AOCL installed. if cannot load to Vulkan. We compile CPU libraries/packages for F16C, AVX, AVX2, FMA, where possible. CPU will always use by default 85% threads on the CPU, where multi-thread will enhance performance during significant phases. Installer should detect the number of threads/cores during installation, and write this down to a key in ".\data\constants.ini". .
- Libraries for encoder, possibly we could have llama.cpp vulkan, and stick that on the 8GB of vram on the passive secondary rx 470 (this is not hip, so stick to vulkan). 
- Libraries for image generation, Stable Diffusion on CPU, if not able to be done on vulkan. If vulkan is an option then put the image generation model on the GPU, and have the encoder on the CPU instead.

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
│   ├── configure.py                # Configuration, and all global variables/constants/maps/lists are here.
│   ├── display.py                  # Gradio, Browser, Python Displays.
│   ├── inference.py                  # Image generation, Model Handling, Text Generation.
│   ├── utilities.py                  # General code, and code that is not more appropriate to be in other scripts.
├── models/                        # Default model directory
└── output/                        # Generated images directory
```

### Development:
A small program in python with gradio 5...
- Page 1 the Interaction page- it will have a text box with a generate button underneath, to the side of that will be configurations for image generation, with dropdown list for reasonable values and sensible default settings and sensible ranges in the lists.
- Page 2 the Configuration page - Where the user configures, model location used for, 1. encoding and 2. image generation, these would require display of current path and a browse button, Additionally page 2 would have whatever configurations are required for these models, individually, with dropdown list for reasonable values and sensible default settings and sensible ranges in the lists.
- Page 3 the Debug/Info page - showing useful values, that actually change, not constants, ensure its all in a text box too, with a little copy button so I could paste it back during development.
