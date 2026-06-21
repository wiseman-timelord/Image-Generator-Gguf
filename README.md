![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/banner_llama.jpg)
# Image-Gradio-Gguf
Status: Beta (working but further development possible)
- [v1.07](https://github.com/wiseman-timelord/Image-Gradio-Gguf/releases/tag/v1.07) has installer that will likely work for AMD, may be iffy on AMD newer than Zen 2, but the program is working well enough.
- v1.08+ will have advanced processor support, mainly, improvements and proper compiling, for Intel chips. I cannot confirm updates after this will install correctly, due to resrictions on internet credit due to funding issues, if you have issues installing and like the project, then please support via sponsor or donate on kofi/patreon, and if enough people do so, then I will endevour to add image editing via SD Klein and Qwen3 for SD. So development may grind to a halt here at some unexpected point.

### Description
A local python image generator from prompt using Qwen 3 Z-Image Engineer encoder and Z-Image Turbo. This is a simple image generation project, covering what is possible currently to be most compitent through GGUF models, through but eventually cover several encoders and image generation models. While this program work great for what it does, it is also example scripts for AI on how to do image inference with such libraries/models, and useful in the production of other progreams that require such things. As you can see it generates OK images, they can look pretty real, but has the typical issues one would exoect under such restrictions and with AI image generation in general, but for simple images I think it will be effective in generating your result in a few iterations, so long as the request is not too barmy.   

### Media
- Generation page, now refined and featuring a dynamic button (v1.05)...  
![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/generate_page.jpg)

- Configure page now featuring Diffuser Placement (A012)...  
![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/configuration_page.jpg)

- Program starts something like this (A013)...
```
================================================================================
      Image-Gradio-Gguf: Starting Program...
================================================================================

  Versioning: Python 3.12.4; Gradio 6.19.0

  CPU     : AMD Ryzen 9 3900X 12-Core Processor
  Threads : 21 (85% of 24 logical cores)
  Vulkan  : True  (1.4.341)
    GPU0: NVIDIA GeForce GTX 1060 3GB
    GPU1: Radeon (TM) RX 470 Graphics

  llama-cli : C:\Inference_Files\Image-Gradio-Gguf\Image-Gradio-Gguf\data\llama_cpp_binaries\llama-cli.exe
  sd        : C:\Inference_Files\Image-Gradio-Gguf\Image-Gradio-Gguf\data\stable_diffusion_binaries\sd-cli.exe

  Encoder  : OK — G:/LargeModels/Text and Image/Qwen3-4b-Z-Image-Turbo-AbliteratedV1-GGUF/Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q4_K_M.gguf
  Diffusion: OK — G:/LargeModels/Text and Image/Z-Image-Turbo-GGUF/z_image_turbo-Q4_0.gguf
  VAE      : OK — G:\LargeModels\Text and Image\Z-Image-Turbo-GGUF\ae.safetensors

[gallery] Scanning for Thumbnails....
[gallery] Rescanned C:\Inference_Files\Image-Gradio-Gguf\Image-Gradio-Gguf\output: 3 images

* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.

```

### Requirements:
- Operating System - Deepseek assesed this as Windows 10-11, however I am testing on Windows 10 22h2.
- Python - Deepseek assesed this as Python versions 3.11-3.13, however I am testing on Python 3.12.
- Graphics- Testing on Vulkan 1.3 card with newer driver. Also its a min DirectX 11.1 GPU for gradio/pyqt6 (browser interface).
- Processor - Detecting/supporting, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX512, F16C, FMA, architecture features. Testing upon Zen 2 with AOCL installed. 
- Libraries - Required libraries is handled by the installer script, but they include both, Llama.Cpp and Stable Diffusion.
- Building - VS 2022 C++, specifically the Desktop Build Tools including CMake. Additionally the Vulkan 1.4 SDK. Additionally Windows 10/11 SDK relevant to your os version.

### Models...
I put Q# because it should support any quantization, the model variety will be expanded upon later....
- intended Encoding model, "Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q#.gguf" and "Qwen3-4b-Uncensored-Z-Image-Engineer-V4-Q#.gguf". Such as for example [Qwen3-Uncensored-TextEncoders-FLUX-Klein-Z-Image-Turbo-GGUF](https://huggingface.co/LuffyTheFox/Qwen3-Uncensored-TextEncoders-FLUX-Klein-Z-Image-Turbo-GGUF) or [Qwen3-4b-Z-Image-Turbo-AbliteratedV1-GGUF](https://huggingface.co/mradermacher/Qwen3-4b-Z-Image-Turbo-AbliteratedV1-GGUF)
- intended Image generation model "z_image_turbo-Q#.gguf" and "ae.safetensors". Such as for example [Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF).

### Instructions:
Currently...
```
1. Ensure you have downloaded the correct models from Huggingface, put them on your model storage drive (if you have one). You can put them in .\models if you like, but then this will cause a bloated folder, and you may not remember later. 
2. Ensure to download the latest release version to a suitable location, then unpack to the place you intend to have the program.
3. Run the program via right click run as admin on `Image-Gradio-Gguf.bat`, this will launch the batch menu.
4. Ensure that Python/Pip has internet access, these may request it during install (if that is an issue you may need to start install again), and the libraries/packages will install appropriately to the program folder, not globally. After which there will be a summary, and you will be returned to the batch menu. hopefully everything went ok for you. If there are any issues at this stage, I would suggest the installer should indicate what the issue was, so maybe paste that into AI with the installer script to fix your system compatibility.
5. Back on the batch menu, select 1 to run the application, the server will start up, and then the built-in browser will pop-up its own window with the Interface displayed. Ensure to go to Configuration page, to set model paths, and what is going to be loaded where, if there is not enough ram on the relating device, it will say in the output (see notes below). Adter configuring, go back to the Generation page, type in your positive prompt, and then hit generate. After you done your first image and everything is confirmed working, then possibly configure the settings further and produce a new prompt, and keep going til you have your images.
6. Upon exiting the program correctly through the exit button, the user will be returned to the batch menu, and one would then exit from there, or otherwise one could just click the [x] in the top right of all windows associated.
```

### Examples:
- If you want to test the image generation, then I suggested just write something like `A picture of a Woodchuck standing next to a pile of wood while juggling small logs of wood.`, I cant remember the exact prompt, but you can compare it to my picture of a Woodchuck standing next to a pile of wood and juggling small logs of wood.

### Notation:
- If you have older hardware, then I strongly advise generating 256x256 images unless you need them larger, as 256x256 images took 22 seconds where as 512x512 images took multiple minuts, on my older hardware. This is not such a problem if you have newer hardware such as 4090. 
- As shown in the images on the readme.md, I was not able to load Z-Image-Turbo in Q4_KM completely to a 8GB GPU without using the option to put the ae.safetensors on the CPU, but I would not use less than a Q4_KM z-image-turbo. This is in addition in my case the Qwen3 model being on the CPU already, but this is no issue as the Encoding is a small task.
- The Qwen3 encoder will do a good job of turning a bad prompt into something workable from a small input, but this is all experimental and an experiment too, so do not expect premium AI image quality, gguf versions of image generation models are sparse, you can see its slightly dated now because Qwen3 is the only encoder, but not that old that its naff. 

### Development:
Development is somewhat stopped for now due to, v1 done (working + nice) and funding issues, in the mean time you could check out my donation/sponsorship links on profile page, but there are still some improvements possible/planned...
- Add AI edit image feature, ie outpaint, etc. Requires Flux-Klein support because Z-image-turbo only supports generation and not editing. The UI woudl also require a rethink, ie what are the options and how am I going to fit them in. It would also need a Qwen 3 Flux Klein Encoder.

### Structure:
- Current plan for scripts is...
```
Image-Gradio-Gguf/
├── Image-Gradio-Gguf.bat      # Windows launcher batch file
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
