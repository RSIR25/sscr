import torch
import rasterio
import numpy as np


import os
import sys
import json
import argparse
import torch

from parse_args import create_parser
from src import utils
from src.model_utils import get_model, load_checkpoint
from train_reconstruct import prepare_output, seed_packages
from torchvision.transforms.functional import to_pil_image

def read_tif_as_tensor(path):
    with rasterio.open(path) as src:
        img = src.read() 
        img = img.astype(np.float32)

    tensor = torch.from_numpy(img)
    return tensor


def load_optical_sar(optical_path, sar_path):
    optical_tensor = read_tif_as_tensor(optical_path)  # [C1, H, W]
    sar_tensor = read_tif_as_tensor(sar_path)          # [C2, H, W]

    input_tensor = torch.cat([optical_tensor, sar_tensor], dim=0)  # [C1+C2, H, W]
    return input_tensor

parser = create_parser('infer')
infer_config = parser.parse_args()

dirname = os.path.dirname(os.path.abspath(__file__))
conf_path = os.path.join(dirname, infer_config.weight_folder, infer_config.experiment_name, "conf.json") \
            if not infer_config.load_config else infer_config.load_config

if os.path.isfile(conf_path):
    with open(conf_path) as file:
        model_config = json.loads(file.read())
    t_args = argparse.Namespace()
    no_overwrite = ['pid', 'device', 'resume_at', 'trained_checkp', 'res_dir', 'weight_folder']
    conf_dict = {key: val for key, val in model_config.items() if key not in no_overwrite}
    for key, val in vars(infer_config).items():
        if key in no_overwrite:
            conf_dict[key] = val
    t_args.__dict__.update(conf_dict)
    config = parser.parse_args(namespace=t_args)
else:
    config = infer_config

config = utils.str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])
print(config)
print(os.path.isfile(conf_path))
device = torch.device(config.device)
seed_packages(config.rdm_seed)
# -----------------------
model = get_model(config).to(device)
load_checkpoint(config, config.weight_folder, model, f"model")
model.eval()

print("model loaded successfully")

gt_path = ""
optical_path = ""
sar_path = ""
input_tensor = load_optical_sar(optical_path, sar_path)  # [C1+C2, H, W]
gt_tensor = read_tif_as_tensor(gt_path)
print(input_tensor)
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.unsqueeze(0)
print(input_tensor.shape)
x = input_tensor.to(device)
inputs = {'A': x, 'B': gt_tensor, 'dates': None, 'masks': None}
with torch.no_grad():
    model.set_input(inputs)
    model.forward()

print("finish:", model.fake_B.shape)
output_tensor = model.fake_B[0, 0, :3, :, :]
print(output_tensor)
output_tensor = output_tensor.cpu() 

img = to_pil_image(output_tensor)
img.save("output.png")