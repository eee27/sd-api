import datetime
import io
import base64
import cv2
import requests
from PIL import Image
import json

with open('config.json', 'r', encoding='utf-8') as configFile:
    config = json.load(configFile)

BASE_URL = config['BASE_URL']
IMG_DATA_DIR = config['IMG2IMG_DIR']

prompt = ''
negative_prompt = ''

src_img = cv2.imread(config['IMG2IMG_SRC_PATH'])

# 编码图像
retval, bytes = cv2.imencode('.png', src_img)
encoded_image = base64.b64encode(bytes).decode('utf-8')

payload = {

    #     # 模型设置
    #     "override_settings":{
    #           "sd_model_checkpoint": "v1-5-pruned.ckpt",
    #           "sd_vae": "animevae.pt",
    #           "CLIP_stop_at_last_layers": 2,
    #     },

    # 基本参数
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "steps": config['STEPS'],
    "sampler_name": config['SAMPLER_NAME'],
    "width": config['WIDTH'],
    "height": config['HEIGHT'],
    "batch_size": config['BATCH_SIZE'],
    "n_iter": config['N_ITER'],
    "seed": config['SEED'],
    "cfg_scale": config['CFG_SCALE'],
    "CLIP_stop_at_last_layers": 2,

    "init_images": [encoded_image],

    # 面部修复 face fix
    "restore_faces": False,

    # 高清修复 highres fix
    # "enable_hr": True,
    # "denoising_strength": 0.4,
    # "hr_scale": 2,
    # "hr_upscaler": "Latent",

}

response = requests.post(url=f'{BASE_URL}/sdapi/v1/img2img', json=payload)
r = response.json()
image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
image.show()
target_name = datetime.datetime.now().strftime('%m%d%H%M%S%f') + '.png'
image.save(IMG_DATA_DIR + target_name)
print('Draw over:' + target_name)
