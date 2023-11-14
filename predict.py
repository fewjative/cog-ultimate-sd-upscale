import subprocess
import threading
import time
from cog import BasePredictor, Input, Path
# from typing import List
import os
import torch
import uuid
import json
import urllib
import websocket
from PIL import Image
from urllib.error import URLError
import random

SAMPLER = [
    "euler",
    "euler_ancestral",
    "heun",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "dpmpp",
    "ddim",
    "uni_pc",
    "uni_pc_bh2"
]

SCHEDULER = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform"
]

MODE_TYPE = [
    "Linear",
    "Chess",
    "None"
]

SEAM_FIX_MODE = [
    "None",
    "Band Pass",
    "Half Tile",
    "Half Tile + Intersections"
]

UPSCALEMODEL = [
    "4x_NMKD-Siax_200k",
    "4x-UltraSharp",
    "RealESRGAN_x4plus",
    "RealESRGAN_x4plus_anime_6B"
]

class Predictor(BasePredictor):
    def setup(self):
        # start server
        self.server_address = "127.0.0.1:8188"
        self.start_server()

    def start_server(self):
        server_thread = threading.Thread(target=self.run_server)
        server_thread.start()

        while not self.is_server_running():
            time.sleep(1)  # Wait for 1 second before checking again

        print("Server is up and running!")

    def run_server(self):
        command = "python ./ComfyUI/main.py"
        server_process = subprocess.Popen(command, shell=True)
        server_process.wait()

    # hacky solution, will fix later
    def is_server_running(self):
        try:
            with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, "123")) as response:
                return response.status == 200
        except URLError:
            return False
    
    def queue_prompt(self, prompt, client_id):
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request("http://{}/prompt".format(self.server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        print(folder_type)
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(self.server_address, url_values)) as response:
            return response.read()

    def get_images(self, ws, prompt, client_id):
        prompt_id = self.queue_prompt(prompt, client_id)['prompt_id']
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break #Execution is done
            else:
                continue #previews are binary data

        history = self.get_history(prompt_id)[prompt_id]
        for o in history['outputs']:
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                print("node output: ", node_output)

                if 'images' in node_output:
                    images_output = []
                    for image in node_output['images']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        images_output.append(image_data)
                output_images[node_id] = images_output

        return output_images

    def get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, prompt_id)) as response:
            return json.loads(response.read())

    def predict(
        self,
        image: Path = Input(description="Input image"),
        positive_prompt: str = Input(description="Positive Prompt", default="Hey! Have a nice day :D"),
        negative_prompt: str = Input(description="Negative Prompt", default=""),
        upscaler: str = Input(description="Upscaler", default="4x-UltraSharp", choices=UPSCALEMODEL),
        upscale_by: float = Input(description="Upscale By", default=2.0),
        use_controlnet_tile: bool = Input(description="Use ControlNet Tile", default=True),
        controlnet_strength: float = Input(description="ControlNet Strength", default=1.0),
        seed: int = Input(description="Sampling seed, leave Empty for Random", default=None),
        steps: int = Input(description="Steps", default=20),
        cfg: float = Input(description="CFG", default=8.0),
        sampler_name: str = Input(description="Sampler", default="euler", choices=SAMPLER),
        scheduler: str = Input(description="Scheduler", default="normal", choices=SCHEDULER),
        denoise: float = Input(description="Denoise", default=0.2),
        mode_type: str = Input(description="Mode Type", default="Linear", choices=MODE_TYPE),
        tile_width: int = Input(description="Tile Width", default=512),
        tile_height: int = Input(description="Tile Height", default=512),
        mask_blur: int = Input(description="Mask Blur", default=8),
        tile_padding: int = Input(description="Tile Padding", default=32),
        seam_fix_mode: str = Input(description="Seam Fix Mode", default="None", choices=SEAM_FIX_MODE),
        seam_fix_denoise: float = Input(description="Seam Fix Denoise", default=1),
        seam_fix_width: int = Input(description="Seam Fix Width", default=64),
        seam_fix_mask_blur: int = Input(description="Seam Fix Mask Blur", default=8),
        seam_fix_padding: int = Input(description="Seam Fix Padding", default=16),
        force_uniform_tiles: bool = Input(description="Force Uniform Tiles", default=True)
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")

        print(f"Using seed: {seed}")

        workflow = None
        if use_controlnet_tile:
            print('Using ControlNet tile with Ultimate SD Upscale')
            workflow_config = "./custom_workflows/ultimatesdupscalecontrolnet.json"
        else:
            workflow_config = "./custom_workflows/ultimatesdupscale.json"

        with open(workflow_config, 'r') as file:
            workflow = json.load(file)

        if not workflow:
            raise Exception('no workflow config found')

        # set input variables
        workflow["3"]["inputs"]["upscale_by"] = upscale_by
        workflow["3"]["inputs"]["seed"] = seed
        workflow["3"]["inputs"]["steps"] = steps
        workflow["3"]["inputs"]["cfg"] = cfg
        workflow["3"]["inputs"]["sampler_name"] = sampler_name
        workflow["3"]["inputs"]["scheduler"] = scheduler
        workflow["3"]["inputs"]["denoise"] = denoise
        workflow["3"]["inputs"]["mode_type"] = mode_type
        workflow["3"]["inputs"]["tile_width"] = tile_width
        workflow["3"]["inputs"]["tile_height"] = tile_height
        workflow["3"]["inputs"]["mask_blur"] = mask_blur
        workflow["3"]["inputs"]["tile_padding"] = tile_padding
        workflow["3"]["inputs"]["seam_fix_mode"] = seam_fix_mode
        workflow["3"]["inputs"]["seam_fix_denoise"] = seam_fix_denoise
        workflow["3"]["inputs"]["seam_fix_width"] = seam_fix_width
        workflow["3"]["inputs"]["seam_fix_mask_blur"] = seam_fix_mask_blur
        workflow["3"]["inputs"]["seam_fix_padding"] = seam_fix_padding

        if force_uniform_tiles:
            workflow["3"]["inputs"]["force_uniform_tiles"] = "enable"
        else:
            workflow["3"]["inputs"]["force_uniform_tiles"] = "disable"

        workflow["4"]["inputs"]["ckpt_name"] = "deliberate_v2.safetensors"

        workflow["6"]["inputs"]["text"] = positive_prompt
        workflow["7"]["inputs"]["text"] = negative_prompt

        workflow["10"]["inputs"]["image"] = "/src/toupscale.png" #image

        workflow["11"]["inputs"]["model_name"] = upscaler + ".pth"

        if use_controlnet_tile:
            workflow["12"]["inputs"]["control_net_name"] = "control_v11f1e_sd15_tile.pth"
            workflow["13"]["inputs"]["strength"] = controlnet_strength
  
        # start the process
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(self.server_address, client_id))
        images = self.get_images(ws, workflow, client_id)

        for node_id in images:
            for image_data in images[node_id]:
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_data))
                image.save("out-"+str(seed)+".png")
                return Path("out-"+str(seed)+".png")