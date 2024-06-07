import torch
from torchvision import transforms
from submodule.Palette.models.network import Network
from PIL import Image
import numpy as np

import argparse
import submodule.Palette.core.praser as Praser


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='submodule/Palette/config/colorization_mirflickr25k.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='test')
    parser.add_argument('-b', '--batch', type=int, default=16, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='6')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    args = parser.parse_args()
    opt = Praser.parse(args)
    print("Loaded config: ")
    print("========================")
    print(opt)
    print("========================\n")
    return opt


def load_palette(ckpt_path="results/train_texture_240531_141727/checkpoint/560_Network.pth", model_args=None):

    model = Network(**model_args)
    # model = Network(unet=)

    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    device = torch.device('cuda:0')
    model.to(device)
    model.set_new_noise_schedule(phase='test')
    model.eval()
    print("Loaded checkpoint: ", ckpt_path)
    return model, device


def predict_img(img_path, model, device):

    tfs = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])

    print("Predicting image: ", img_path)
    img = Image.open(img_path).convert('RGB')
    img_gr = img.convert('L').convert('RGB')

    x = tfs(img_gr)
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        p = model.restoration(y_cond=x,)

    # print(p)

    predict_img = Image.fromarray(p[0][0][0].cpu().numpy()*255)
    predict_img = predict_img.convert("RGB")
    predict_img.save("result.jpg")

    return predict_img




if __name__ == "__main__":
    opt = parse_config()
    model, device = load_palette("results/train_texture_240531_141727/checkpoint/560_Network.pth", opt["model"]["which_networks"][0]["args"])
    img_path = "dataset/LeafData/Bael/healthy/output/Bael_healthy_0007_mask_aligned_sdf.png"
    predict_img(img_path, model, device)