import os
import os.path as osp
import cv2
import numpy as np
import numpy.random as npr
import torch
import torch.nn.functional as F
import torchvision.transforms as tvtrans
import PIL.Image
from tqdm import tqdm
from PIL import Image
import copy
import json
from collections import OrderedDict

#######
# css #
#######

css_empty = ""

css_version_4_11_0 = """
    #customized_imbox {
        min-height: 450px;
        max-height: 450px;
    }
    #customized_imbox>div[data-testid="image"] {
        min-height: 450px;
    }
    #customized_imbox>div[data-testid="image"]>span[data-testid="source-select"] {
        max-height: 0px;
    }
    #customized_imbox>div[data-testid="image"]>span[data-testid="source-select"]>button {
        max-height: 0px;
    }
    #customized_imbox>div[data-testid="image"]>div.upload-container>div.image-frame>img {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translateX(-50%) translateY(-50%);
        width: unset;
        height: unset;
        max-height: 450px;
    }        
    #customized_imbox>div.unpadded_box {
        min-height: 450px;
    }
    #myinst {
        font-size: 0.8rem; 
        margin: 0rem;
        color: #6B7280;
    }
    #maskinst {
        text-align: justify;
        min-width: 1200px;
    }
    #maskinst>img {
        min-width:399px;
        max-width:450px;
        vertical-align: top;
        display: inline-block;
    }
    #maskinst:after {
        content: "";
        width: 100%;
        display: inline-block;
    }
"""

##########
# helper #
##########

def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')

def auto_dropdown(name, choices_od, value):
    import gradio as gr
    option_list = [pi for pi in choices_od.keys()]
    return gr.Dropdown(label=name, choices=option_list, value=value)

def load_sd_from_file(target):
    if osp.splitext(target)[-1] == '.ckpt':
        sd = torch.load(target, map_location='cpu')['state_dict']
    elif osp.splitext(target)[-1] == '.pth':
        sd = torch.load(target, map_location='cpu')
    elif osp.splitext(target)[-1] == '.safetensors':
        from safetensors.torch import load_file as stload
        sd = OrderedDict(stload(target, device='cpu'))
    else:
        assert False, "File type must be .ckpt or .pth or .safetensors"
    return sd

def torch_to_numpy(x):
    return x.detach().to('cpu').numpy()

if __name__ == '__main__':
    pass
