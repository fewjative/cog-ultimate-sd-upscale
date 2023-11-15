# Cog Model for Ultimate SD Upscale with ControlNet tile via ComfyUI

This is an implementation of Ultimate-SD-Upscale via Comfy UI as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

Model can be run here:
https://replicate.com/fewjative/ultimate-sd-upscale

Initial logic to use ComfyUI is thanks to @Lucataco
https://github.com/lucataco/cog-comfyui-sdxl-txt2img

First, download the models. There are 4 upscalers, 1 base model, and 1 controlnet model:

    These go into 'upscaler-cache'
    https://openmodeldb.info/models/4x-realesrgan-x4plus
    https://openmodeldb.info/models/4x-realesrgan-x4plus-anime-6b
    https://openmodeldb.info/models/4x-NMKD-Siax-CX
    https://civitai.com/models/116225/4x-ultrasharp

    These go into 'model-cache'
    https://huggingface.co/XpucT/Deliberate/blob/main/Deliberate_v2.safetensors

    These go into 'controlnet-cache'
    https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main

This also requires ComfyUI. git clone this repo in the src dir:
    https://github.com/comfyanonymous/ComfyUI

And follow the instructions for the UltimateSDUpscale extension ( git clone in the ComfyUI/custom_nodes folders):
    https://github.com/ssitu/ComfyUI_UltimateSDUpscale

Lastly,in order to use the cache folder, you must modify this file to add new search entry points. Note you won't see this file until you clone ComfyUI:
\cog-ultimate-sd-upscale\ComfyUI\extra_model_paths.yaml

other_ui:
    base_path: /src
    checkpoints: model-cache/
    upscale_models: upscaler-cache/
    controlnet: controlnet-cache/

Then, you can run predictions like such:

    cog predict -i image=@toupscale.png
    cog predict -i image=@jesko.png -i positive_prompt="A car from need for speed, in a garage, cinematic"

The workflow for ControlNet Tile and non Controlnet Tile Ultimate SD Upscale used for this repo is found under:

    custom_workflows/ultimatesdupscale.json
    custom_workflows/ultimatesdupscalecontrolnet.json

## Example:

![image to scale](toupscale.png)

vs

![after scale](output.png)
