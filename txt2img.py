import base64
import datetime
import io

import requests
from PIL import Image

BASE_URL = "http://192.168.202.55:7860"
IMG_DATA_DIR = r"D:/sd/img/"

prompt = 'best quality,masterpiece,ultra detailed,4K、8K,UHD,HDR,photographic,1 people'
negative_prompt = ''

payload = {
    # 模型设置
    # "override_settings": {
    #    "sd_model_checkpoint": "v1-5-pruned.ckpt",
    #    "sd_vae": "animevae.pt",
    #    "CLIP_stop_at_last_layers": 2,
    # },

    # 基本参数
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "steps": 20,
    "sampler_name": "Euler a",
    "width": 512,
    "height": 512,
    "batch_size": 1,
    "n_iter": 1,
    "seed": -1,
    "CLIP_stop_at_last_layers": 2,

    # 面部修复 face fix
    "restore_faces": False,

    # 高清修复 highres fix
    # "enable_hr": True,
    # "denoising_strength": 0.4,
    # "hr_scale": 2,
    # "hr_upscaler": "Latent",

}

response = requests.post(url=f'{BASE_URL}/sdapi/v1/txt2img', json=payload)
r = response.json()
image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))

image.show()
target_name = datetime.datetime.now().strftime('%m%d%H%M%S%f') + '.png'
image.save(IMG_DATA_DIR + target_name)
print('Draw over:' + target_name)
