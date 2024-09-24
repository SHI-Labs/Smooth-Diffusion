################################################################################
# Copyright (C) 2023 Jiayi Guo, Xingqian Xu, Manushree Vasu - All Rights Reserved                         #
################################################################################

import gradio as gr
import os
import os.path as osp
import PIL
from PIL import Image
import numpy as np
from collections import OrderedDict
from easydict import EasyDict as edict
from functools import partial

import torch
import torchvision.transforms as tvtrans
import time
import argparse
import json
import hashlib
import copy
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from app_utils import auto_dropdown

version = "Smooth Diffusion Demo v1.0"
refresh_symbol = "\U0001f504" # 🔄
recycle_symbol = '\U0000267b' #

##############
# model_book #
##############

choices = edict()
choices.diffuser = OrderedDict([
    ['SD-v1-5' , "runwayml/stable-diffusion-v1-5"],
    ['OJ-v4' , "prompthero/openjourney-v4"],
    ['RR-v2', "SG161222/Realistic_Vision_V2.0"],
])

choices.lora = OrderedDict([
    ['empty', ""],
    ['Smooth-LoRA-v1', 'assets/models/smooth_lora.safetensors'],
])

choices.scheduler = OrderedDict([
    ['DDIM', DDIMScheduler],
])

choices.inversion = OrderedDict([
    ['NTI', 'NTI'],
    ['DDIM w/o text', 'DDIM w/o text'],
    ['DDIM', 'DDIM'], 
])

default = edict()
default.diffuser = 'RR-v2'
default.scheduler = 'DDIM'
default.lora = 'Smooth-LoRA-v1'
default.inversion = 'NTI'
default.step = 50
default.cfg_scale = 7.5
default.framen = 24
default.fps = 16
default.nullinv_inner_step = 10
default.threshold = 0.8
default.variation = 0.8

##########
# helper #
##########

def lerp(t, v0, v1):
    if isinstance(t, float):
        return v0*(1-t) + v1*t
    elif isinstance(t, (list, np.ndarray)):
        return [v0*(1-ti) + v1*ti for ti in t]

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    # mostly copied from
    # https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    v0_unit = v0 / np.linalg.norm(v0)
    v1_unit = v1 / np.linalg.norm(v1)
    dot = np.sum(v0_unit * v1_unit)
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0, v1)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t

    if isinstance(t, float):
        tlist = [t]
    elif isinstance(t, (list, np.ndarray)):
        tlist = t

    v2_list = []

    for ti in tlist:
        theta_t = theta_0 * ti
        sin_theta_t = np.sin(theta_t)
        # Finish the slerp algorithm
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
        v2_list.append(v2)

    if isinstance(t, float):
        return v2_list[0]
    else:
        return v2_list

def offset_resize(image, width=512, height=512, left=0, right=0, top=0, bottom=0):
   
    image = np.array(image)[:, :, :3]
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = Image.fromarray(image).resize((width, height))
    return image

def auto_dtype_device_shape(tlist, v0, v1, func,):
    vshape = v0.shape
    assert v0.shape == v1.shape
    assert isinstance(tlist, (list, np.ndarray))
    
    if isinstance(v0, torch.Tensor):
        is_torch = True
        dtype, device = v0.dtype, v0.device
        v0 = v0.to('cpu').numpy().astype(float).flatten()
        v1 = v1.to('cpu').numpy().astype(float).flatten()
    else:
        is_torch = False
        dtype = v0.dtype
        assert isinstance(v0, np.ndarray)
        assert isinstance(v1, np.ndarray)
        v0 = v0.astype(float).flatten()
        v1 = v1.astype(float).flatten()

    r = func(tlist, v0, v1)

    if is_torch:
        r = [torch.Tensor(ri).view(*vshape).to(dtype).to(device) for ri in r]
    else:
        r = [ri.astype(dtype) for ri in r]
    return r

auto_lerp = partial(auto_dtype_device_shape, func=lerp)
auto_slerp = partial(auto_dtype_device_shape, func=slerp)

def frames2mp4(vpath, frames, fps):
    import moviepy.editor as mpy
    frames = [np.array(framei) for framei in frames]
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(vpath, fps=fps)

def negseed_to_rndseed(seed):
    if seed < 0:
        seed = np.random.randint(0, np.iinfo(np.uint32).max-100)
    return seed

def regulate_image(pilim):
    w, h = pilim.size
    w = int(round(w/64)) * 64
    h = int(round(h/64)) * 64
    return pilim.resize([w, h], resample=PIL.Image.BILINEAR)

def txt_to_emb(model, prompt):
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",)
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_embeddings

def hash_pilim(pilim):
    hasha = hashlib.md5(pilim.tobytes()).hexdigest()
    return hasha

def hash_cfgdict(cfgdict):
    hashb = hashlib.md5(json.dumps(cfgdict, sort_keys=True).encode('utf-8')).hexdigest()
    return hashb

def remove_earliest_file(path, max_allowance=500, remove_ratio=0.1, ext=None):
    if len(os.listdir(path)) <= max_allowance:
        return
    def get_mtime(fname):
        return osp.getmtime(osp.join(path, fname))
    if ext is None:
        flist = sorted(os.listdir(path), key=get_mtime)
    else:
        flist = [fi for fi in os.listdir(path) if fi.endswith(ext)]
        flist = sorted(flist, key=get_mtime)
    exceedn = max(len(flist)-max_allowance, 0)
    removen = int(max_allowance*remove_ratio)
    removen = max(1, removen) + exceedn
    for fi in flist[0:removen]:
        os.remove(osp.join(path, fi))

def remove_decoupled_file(path, exta='.mp4', extb='.json'):
    tag_a = [osp.splitext(fi)[0] for fi in os.listdir(path) if fi.endswith(exta)]
    tag_b = [osp.splitext(fi)[0] for fi in os.listdir(path) if fi.endswith(extb)]
    tag_a_extra = set(tag_a) - set(tag_b)
    tag_b_extra = set(tag_b) - set(tag_a)
    [os.remove(osp.join(path, tagi+exta)) for tagi in tag_a_extra]
    [os.remove(osp.join(path, tagi+extb)) for tagi in tag_b_extra]

@torch.no_grad()
def t2i_core(model, xt, emb, nemb, step=30, cfg_scale=7.5, return_list=False):
    from nulltxtinv_wrapper import diffusion_step, latent2image
    model.scheduler.set_timesteps(step)
    xi = xt
    emb = txt_to_emb(model, "") if emb is None else emb
    nemb = txt_to_emb(model, "") if nemb is None else nemb
    if return_list:
        xi_list = [xi.clone()]
    for i, t in enumerate(tqdm(model.scheduler.timesteps)):
        embi = emb[i] if isinstance(emb, list) else emb
        nembi = nemb[i] if isinstance(nemb, list) else nemb
        context = torch.cat([nembi, embi])
        xi = diffusion_step(model, xi, context, t, cfg_scale, low_resource=False)
        if return_list:
            xi_list.append(xi.clone())
    x0 = xi
    im = latent2image(model.vae, x0, return_type='pil')

    if return_list:
        return im, xi_list
    else:
        return im

########
# main #
########

class wrapper(object):
    def __init__(self, 
                 fp16=False, 
                 tag_diffuser=None, 
                 tag_lora=None,
                 tag_scheduler=None,):

        self.device = "cuda"
        if fp16:
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        self.load_all(tag_diffuser, tag_lora, tag_scheduler)

        self.image_latent_dim = 4
        self.batchsize = 8
        self.seed = {}

        self.cache_video_folder = "temp/video"
        self.cache_video_maxn = 500
        self.cache_image_folder = "temp/image"
        self.cache_image_maxn = 500
        self.cache_inverse_folder = "temp/inverse"
        self.cache_inverse_maxn = 500

    def load_all(self, tag_diffuser, tag_lora, tag_scheduler):
        self.load_diffuser_lora(tag_diffuser, tag_lora)
        self.load_scheduler(tag_scheduler)
        return tag_diffuser, tag_lora, tag_scheduler

    def load_diffuser_lora(self, tag_diffuser, tag_lora):
        self.net = StableDiffusionPipeline.from_pretrained(
            choices.diffuser[tag_diffuser], torch_dtype=self.torch_dtype).to(self.device)
        self.net.safety_checker = None
        if tag_lora != 'empty':
            self.net.unet.load_attn_procs(
                choices.lora[tag_lora], use_safetensors=True,)
        self.tag_diffuser = tag_diffuser
        self.tag_lora = tag_lora
        return tag_diffuser, tag_lora

    def load_scheduler(self, tag_scheduler):
        self.net.scheduler = choices.scheduler[tag_scheduler].from_config(self.net.scheduler.config)
        self.tag_scheduler = tag_scheduler
        return tag_scheduler

    def reset_seed(self, which='ltintp'):
        return -1

    def recycle_seed(self, which='ltintp'):
        if which not in self.seed:
            return self.reset_seed(which=which)
        else:
            return self.seed[which]

    ##########
    # helper #
    ##########

    def precheck_model(self, tag_diffuser, tag_lora, tag_scheduler):
        if (tag_diffuser != self.tag_diffuser) or (tag_lora != self.tag_lora):
            self.load_all(tag_diffuser, tag_lora, tag_scheduler)
        if tag_scheduler != self.tag_scheduler:
            self.load_scheduler(tag_scheduler)

    ########
    # main #
    ########

    def ddiminv(self, img, cfgdict):
        txt, step, cfg_scale = cfgdict['txt'], cfgdict['step'], cfgdict['cfg_scale']
        from nulltxtinv_wrapper import NullInversion
        null_inversion_model = NullInversion(self.net, step, cfg_scale)
        with torch.no_grad():
            emb = txt_to_emb(self.net, txt)
            nemb = txt_to_emb(self.net, "")
        xt = null_inversion_model.ddim_invert(img, txt)
        data = {
            'step' : step, 'cfg_scale' : cfg_scale, 'txt' : txt,
            'diffuser' : self.tag_diffuser, 'lora' : self.tag_lora,
            'xt': xt, 'emb': emb, 'nemb': nemb,}
        return data

    def nullinv_or_loadcache(self, img, cfgdict, force_reinvert=False):
        hash = hash_pilim(img) + "--" + hash_cfgdict(cfgdict)
        cdir = self.cache_inverse_folder
        cfname = osp.join(cdir, hash+'.pth')

        if osp.isfile(cfname) and (not force_reinvert):
            cache_data = torch.load(cfname)
            dtype = next(self.net.unet.parameters()).dtype
            device = next(self.net.unet.parameters()).device
            cache_data['xt'] = cache_data['xt'].to(device=device, dtype=dtype)
            cache_data['emb'] = cache_data['emb'].to(device=device, dtype=dtype)
            cache_data['nemb'] = [
                nembi.to(device=device, dtype=dtype)
                    for nembi in cache_data['nemb']]
            return cache_data
        else:
            txt, step, cfg_scale = cfgdict['txt'], cfgdict['step'], cfgdict['cfg_scale']
            inner_step = cfgdict['inner_step']
            from nulltxtinv_wrapper import NullInversion
            null_inversion_model = NullInversion(self.net, step, cfg_scale)
            with torch.no_grad():
                emb = txt_to_emb(self.net, txt)
            xt, nemb = null_inversion_model.null_invert(img, txt, num_inner_steps=inner_step)
            cache_data = {
                'step' : step, 'cfg_scale' : cfg_scale, 'txt' : txt,
                'inner_step' : inner_step,
                'diffuser' : self.tag_diffuser, 'lora' : self.tag_lora,
                'xt' : xt.to('cpu'),
                'emb' : emb.to('cpu'),
                'nemb' : [nembi.to('cpu') for nembi in nemb],}
            os.makedirs(cdir, exist_ok=True)
            remove_earliest_file(cdir, max_allowance=self.cache_inverse_maxn)
            torch.save(cache_data, cfname)
            data = {
                'step' : step, 'cfg_scale' : cfg_scale, 'txt' : txt,
                'inner_step' : inner_step,
                'diffuser' : self.tag_diffuser, 'lora' : self.tag_lora,
                'xt' : xt, 'emb' : emb, 'nemb' : nemb,}
            return data

    def nullinvdual_or_loadcachedual(self, img0, img1, cfgdict, force_reinvert=False):
        hash = hash_pilim(img0) + "--" + hash_pilim(img1) + "--" + hash_cfgdict(cfgdict)
        cdir = self.cache_inverse_folder
        cfname = osp.join(cdir, hash+'.pth')

        if osp.isfile(cfname) and (not force_reinvert):
            cache_data = torch.load(cfname)
            dtype = next(self.net.unet.parameters()).dtype
            device = next(self.net.unet.parameters()).device
            cache_data['xt0'] = cache_data['xt0'].to(device=device, dtype=dtype)
            cache_data['xt1'] = cache_data['xt1'].to(device=device, dtype=dtype)
            cache_data['emb0'] = cache_data['emb0'].to(device=device, dtype=dtype)
            cache_data['emb1'] = cache_data['emb1'].to(device=device, dtype=dtype)
            cache_data['nemb'] = [
                nembi.to(device=device, dtype=dtype)
                    for nembi in cache_data['nemb']]

            cache_data_a = copy.deepcopy(cache_data)
            cache_data_a['xt'] = cache_data_a.pop('xt0')
            cache_data_a['emb'] = cache_data_a.pop('emb0')
            cache_data_a.pop('xt1'); cache_data_a.pop('emb1')

            cache_data_b = cache_data
            cache_data_b['xt'] = cache_data_b.pop('xt1')
            cache_data_b['emb'] = cache_data_b.pop('emb1')
            cache_data_b.pop('xt0'); cache_data_b.pop('emb0')

            return cache_data_a, cache_data_b
        else:
            txt0, txt1, step, cfg_scale, inner_step = \
                cfgdict['txt0'], cfgdict['txt1'], cfgdict['step'], \
                cfgdict['cfg_scale'], cfgdict['inner_step']
            
            from nulltxtinv_wrapper import NullInversion
            null_inversion_model = NullInversion(self.net, step, cfg_scale)
            with torch.no_grad():
                emb0 = txt_to_emb(self.net, txt0)
                emb1 = txt_to_emb(self.net, txt1)
            
            xt0, xt1, nemb = null_inversion_model.null_invert_dual(
                img0, img1, txt0, txt1, num_inner_steps=inner_step)
            cache_data = {
                'step' : step, 'cfg_scale' : cfg_scale, 
                'txt0' : txt0, 'txt1' : txt1,
                'inner_step' : inner_step,
                'diffuser' : self.tag_diffuser, 'lora' : self.tag_lora,
                'xt0' : xt0.to('cpu'), 'xt1' : xt1.to('cpu'),
                'emb0' : emb0.to('cpu'), 'emb1' : emb1.to('cpu'),
                'nemb' : [nembi.to('cpu') for nembi in nemb],}
            os.makedirs(cdir, exist_ok=True)
            remove_earliest_file(cdir, max_allowance=self.cache_inverse_maxn)
            torch.save(cache_data, cfname)
            data0 = {
                'step' : step, 'cfg_scale' : cfg_scale, 'txt' : txt0,
                'inner_step' : inner_step,
                'diffuser' : self.tag_diffuser, 'lora' : self.tag_lora,
                'xt' : xt0, 'emb' : emb0, 'nemb' : nemb,}
            data1 = {
                'step' : step, 'cfg_scale' : cfg_scale, 'txt' : txt1,
                'inner_step' : inner_step,
                'diffuser' : self.tag_diffuser, 'lora' : self.tag_lora,
                'xt' : xt1, 'emb' : emb1, 'nemb' : nemb,}
            return data0, data1

    def image_inversion(
            self, img, txt, 
            cfg_scale, step, 
            inversion, inner_step, force_reinvert):
        from nulltxtinv_wrapper import text2image_ldm
        if inversion == 'DDIM w/o text':
            txt = ''
        if not inversion == 'NTI':
            data = self.ddiminv(img, {'txt':txt, 'step':step, 'cfg_scale':cfg_scale,})
        else:
            data = self.nullinv_or_loadcache(
                img, {'txt':txt, 'step':step,
                      'cfg_scale':cfg_scale, 'inner_step':inner_step,
                      'diffuser' : self.tag_diffuser, 'lora' : self.tag_lora,}, force_reinvert)
        
        if inversion == 'NTI':
            img_inv, _ = text2image_ldm(
                self.net, [txt], step, cfg_scale, 
                latent=data['xt'], uncond_embeddings=data['nemb'])
        else:
            img_inv, _ = text2image_ldm(
            self.net, [txt], step, cfg_scale,
            latent=data['xt'], uncond_embeddings=None)
            
        return img_inv

    def image_editing(
        self, img, txt_0, txt_1,
        cfg_scale, step, thresh, 
        inversion, inner_step, force_reinvert):
        from nulltxtinv_wrapper import text2image_ldm_imedit
        if inversion == 'DDIM w/o text':
            txt_0 = ''
        if not inversion == 'NTI':
            data = self.ddiminv(img, {'txt':txt_0, 'step':step, 'cfg_scale':cfg_scale,})
            img_edited, _ = text2image_ldm_imedit(
                self.net, thresh, [txt_0], [txt_1], step, cfg_scale,
                latent=data['xt'], uncond_embeddings=None)
        else:
            data = self.nullinv_or_loadcache(
                img, {'txt':txt_0, 'step':step,
                      'cfg_scale':cfg_scale, 'inner_step':inner_step,
                      'diffuser' : self.tag_diffuser, 'lora' : self.tag_lora,}, force_reinvert)
            img_edited, _ = text2image_ldm_imedit(
                self.net, thresh, [txt_0], [txt_1], step, cfg_scale,
                latent=data['xt'], uncond_embeddings=data['nemb'])
        
        return img_edited

    def general_interpolation(
            self, xset0, xset1,
            cfg_scale, step, tlist,):

        xt0, emb0, nemb0 = xset0['xt'], xset0['emb'], xset0['nemb']
        xt1, emb1, nemb1 = xset1['xt'], xset1['emb'], xset1['nemb']
        framen = len(tlist)

        xt_list = auto_slerp(tlist, xt0, xt1)
        emb_list = auto_lerp(tlist, emb0, emb1)
        
        if isinstance(nemb0, list) and isinstance(nemb1, list):
            assert len(nemb0) == len(nemb1)
            nemb_list = [auto_lerp(tlist, e0, e1) for e0, e1 in zip(nemb0, nemb1)]
            nemb_islist = True
        else:
            nemb_list = auto_lerp(tlist, nemb0, nemb1)
            nemb_islist = False

        im_list = []
        for frameidx in range(0, len(xt_list), self.batchsize):
            xt_batch = [xt_list[idx] for idx in range(frameidx, min(frameidx+self.batchsize, framen))]
            xt_batch = torch.cat(xt_batch, dim=0)
            emb_batch = [emb_list[idx] for idx in range(frameidx, min(frameidx+self.batchsize, framen))]
            emb_batch = torch.cat(emb_batch, dim=0)
            if nemb_islist:
                nemb_batch = []
                for nembi in nemb_list:
                    nembi_batch = [nembi[idx] for idx in range(frameidx, min(frameidx+self.batchsize, framen))]
                    nembi_batch = torch.cat(nembi_batch, dim=0)
                    nemb_batch.append(nembi_batch)
            else:
                nemb_batch = [nemb_list[idx] for idx in range(frameidx, min(frameidx+self.batchsize, framen))]
                nemb_batch = torch.cat(nemb_batch, dim=0)

            im = t2i_core(
                self.net, xt_batch, emb_batch, nemb_batch, step, cfg_scale)
            im_list += im if isinstance(im, list) else [im]

        return im_list

    def run_iminvs(
            self, img, text, 
            cfg_scale, step, 
            force_resize, width, height,
            inversion, inner_step, force_reinvert,
            tag_diffuser, tag_lora, tag_scheduler, ):
        
        self.precheck_model(tag_diffuser, tag_lora, tag_scheduler)
        
        if force_resize:
            img = offset_resize(img, width, height)
        else:
            img = regulate_image(img)

        recon_output = self.image_inversion(
            img, text, cfg_scale, step, 
            inversion, inner_step, force_reinvert)

        idir = self.cache_image_folder
        os.makedirs(idir, exist_ok=True)
        remove_earliest_file(idir, max_allowance=self.cache_image_maxn)
        sname = "time{}_iminvs_{}_{}".format(
            int(time.time()), self.tag_diffuser, self.tag_lora,)
        ipath = osp.join(idir, sname+'.png')
        recon_output.save(ipath)
        
        return [recon_output]

    def run_imedit(
            self, img, txt_0,txt_1, 
            threshold, cfg_scale, step, 
            force_resize, width, height,
            inversion, inner_step, force_reinvert,
            tag_diffuser, tag_lora, tag_scheduler, ):
        
        self.precheck_model(tag_diffuser, tag_lora, tag_scheduler)
        if force_resize:
            img = offset_resize(img, width, height)
        else:
            img = regulate_image(img)

        edited_img= self.image_editing(
            img, txt_0,txt_1, cfg_scale, step, threshold,
            inversion, inner_step, force_reinvert)

        idir = self.cache_image_folder
        os.makedirs(idir, exist_ok=True)
        remove_earliest_file(idir, max_allowance=self.cache_image_maxn)
        sname = "time{}_imedit_{}_{}".format(
            int(time.time()), self.tag_diffuser, self.tag_lora,)
        ipath = osp.join(idir, sname+'.png')
        edited_img.save(ipath)
     
        return [edited_img]


    def run_imintp(
            self, 
            img0, img1, txt0, txt1,
            cfg_scale, step, 
            framen, fps, 
            force_resize, width, height,
            inversion, inner_step, force_reinvert,
            tag_diffuser, tag_lora, tag_scheduler,):
        
        self.precheck_model(tag_diffuser, tag_lora, tag_scheduler)
        if txt1 == '':
            txt1 = txt0
        if force_resize:
            img0 = offset_resize(img0, width, height)
            img1 = offset_resize(img1, width, height)
        else:
            img0 = regulate_image(img0)
            img1 = regulate_image(img1)

        if inversion == 'DDIM':
            data0 = self.ddiminv(img0, {'txt':txt0, 'step':step, 'cfg_scale':cfg_scale,})
            data1 = self.ddiminv(img1, {'txt':txt1, 'step':step, 'cfg_scale':cfg_scale,})
        elif inversion == 'DDIM w/o text':
            data0 = self.ddiminv(img0, {'txt':"", 'step':step, 'cfg_scale':cfg_scale,})
            data1 = self.ddiminv(img1, {'txt':"", 'step':step, 'cfg_scale':cfg_scale,})
        else:
            data0, data1 = self.nullinvdual_or_loadcachedual(
                img0, img1, {'txt0':txt0, 'txt1':txt1, 'step':step,
                             'cfg_scale':cfg_scale, 'inner_step':inner_step,
                             'diffuser' : self.tag_diffuser, 'lora' : self.tag_lora,}, force_reinvert)

        tlist = np.linspace(0.0, 1.0, framen)

        iminv0 = t2i_core(self.net, data0['xt'], data0['emb'], data0['nemb'], step, cfg_scale)
        iminv1 = t2i_core(self.net, data1['xt'], data1['emb'], data1['nemb'], step, cfg_scale)
        frames = self.general_interpolation(data0, data1, cfg_scale, step, tlist)

        vdir = self.cache_video_folder
        os.makedirs(vdir, exist_ok=True)
        remove_earliest_file(vdir, max_allowance=self.cache_video_maxn)
        sname = "time{}_imintp_{}_{}_framen{}_fps{}".format(
            int(time.time()), self.tag_diffuser, self.tag_lora, framen, fps)
        vpath = osp.join(vdir, sname+'.mp4')
        frames2mp4(vpath, frames, fps)
        jpath = osp.join(vdir, sname+'.json')
        cfgdict = {
            "method" : "image_interpolation",
            "txt0" : txt0, "txt1" : txt1,
            "cfg_scale" : cfg_scale, "step" : step, 
            "framen" : framen, "fps" : fps,
            "force_resize" : force_resize, "width" : width, "height" : height,
            "inversion" : inversion, "inner_step" : inner_step, 
            "force_reinvert" : force_reinvert, 
            "tag_diffuser" : tag_diffuser, "tag_lora" : tag_lora, "tag_scheduler" : tag_scheduler,}
        with open(jpath, 'w') as f:
            json.dump(cfgdict, f, indent=4)

        return frames, vpath, [iminv0, iminv1]

#################
# get examples #
#################
cache_examples = False
def get_imintp_example():
    case = [
        [
            'assets/images/interpolation/cityview1.png', 
            'assets/images/interpolation/cityview2.png', 
            'A city view',],
        [
            'assets/images/interpolation/woman1.png', 
            'assets/images/interpolation/woman2.png', 
            'A woman face',],
        [
            'assets/images/interpolation/land1.png', 
            'assets/images/interpolation/land2.png', 
            'A beautiful landscape',],
        [
            'assets/images/interpolation/dog1.png', 
            'assets/images/interpolation/dog2.png', 
            'A realistic dog',],
        [
            'assets/images/interpolation/church1.png', 
            'assets/images/interpolation/church2.png', 
            'A church',],
        [
            'assets/images/interpolation/rabbit1.png', 
            'assets/images/interpolation/rabbit2.png', 
            'A cute rabbit',],
        [
            'assets/images/interpolation/horse1.png', 
            'assets/images/interpolation/horse2.png', 
            'A robot horse',],
    ]
    return case

def get_iminvs_example():
    case = [
        [
            'assets/images/inversion/000000560011.jpg', 
            'A mouse is next to a keyboard on a desk',],
        [
            'assets/images/inversion/000000029596.jpg', 
            'A room with a couch, table set with dinnerware and a television.',],
    ]
    return case


def get_imedit_example():
    case = [
        [
            'assets/images/editing/rabbit.png', 
            'A rabbit is eating a watermelon on the table', 
            'A cat is eating a watermelon on the table', 
            0.7,],
        [
            'assets/images/editing/cake.png', 
            'A chocolate cake with cream on it', 
            'A chocolate cake with strawberries on it', 
            0.9,],
        [
            'assets/images/editing/banana.png', 
            'A banana on the table', 
            'A banana and an apple on the table', 
            0.8,],
        
    ]
    return case


#################
# sub interface #
#################


def interface_imintp(wrapper_obj):
    with gr.Row():
        with gr.Column():
            img0 = gr.Image(label="Image Input 0", type='pil',  elem_id='customized_imbox')
        with gr.Column():
            img1 = gr.Image(label="Image Input 1", type='pil', elem_id='customized_imbox')
        with gr.Column():
            video_output = gr.Video(label="Video Result", format='mp4', elem_id='customized_imbox')
    with gr.Row(): 
        with gr.Column():      
            txt0 = gr.Textbox(label='Text Input', lines=1, placeholder="Input prompt...", )
        with gr.Column(): 
            with gr.Row():
                inversion = auto_dropdown('Inversion', choices.inversion, default.inversion)
                inner_step = gr.Slider(label="Inner Step (NTI)", value=default.nullinv_inner_step, minimum=1, maximum=10, step=1)
                force_reinvert = gr.Checkbox(label="Force ReInvert (NTI)", value=False)
                    

    with gr.Row():
        with gr.Column(): 
            with gr.Row():
                framen = gr.Slider(label="Frame Number", minimum=8, maximum=default.framen, value=default.framen, step=1)
                fps = gr.Slider(label="Video FPS", minimum=4, maximum=default.fps, value=default.fps, step=4)
            with gr.Row():
                button_run = gr.Button("Run") 
            

        with gr.Column():
            with gr.Accordion('Frame Results', open=False):
                frame_output = gr.Gallery(label="Frames", elem_id='customized_imbox')
            with gr.Accordion("Inversion Results", open=False):
                inv_output = gr.Gallery(label="Inversion Results", elem_id='customized_imbox')
            with gr.Accordion('Advanced Settings', open=False):
                with gr.Row():
                    tag_diffuser = auto_dropdown('Diffuser', choices.diffuser, default.diffuser)
                    tag_lora = auto_dropdown('Use LoRA', choices.lora, default.lora)
                    tag_scheduler = auto_dropdown('Scheduler', choices.scheduler, default.scheduler)
                with gr.Row():
                    cfg_scale = gr.Number(label="Scale", minimum=1, maximum=10, value=default.cfg_scale, step=0.5)
                    step = gr.Number(default.step, label="Step", precision=0)
                with gr.Row():
                    force_resize = gr.Checkbox(label="Force Resize", value=True)
                    inp_width = gr.Slider(label="Width", minimum=256, maximum=1024, value=512, step=64)
                    inp_height = gr.Slider(label="Height", minimum=256, maximum=1024, value=512, step=64)
                with gr.Row():
                    txt1 = gr.Textbox(label='Optional Different Text Input for Image Input 1', lines=1, placeholder="Input prompt...", )
            

    tag_diffuser.change(
        wrapper_obj.load_all,
        inputs = [tag_diffuser, tag_lora, tag_scheduler],
        outputs = [tag_diffuser, tag_lora, tag_scheduler],)

    tag_lora.change(
        wrapper_obj.load_all,
        inputs = [tag_diffuser, tag_lora, tag_scheduler],
        outputs = [tag_diffuser, tag_lora, tag_scheduler],)

    tag_scheduler.change(
        wrapper_obj.load_scheduler,
        inputs = [tag_scheduler],
        outputs = [tag_scheduler],)

    button_run.click(
        wrapper_obj.run_imintp,
        inputs=[img0, img1, txt0, txt1,
                cfg_scale, step, 
                framen, fps, 
                force_resize, inp_width, inp_height,
                inversion, inner_step, force_reinvert,
                tag_diffuser, tag_lora, tag_scheduler,],
        outputs=[frame_output, video_output, inv_output])

    gr.Examples(
        label='Examples', 
        examples=get_imintp_example(), 
        fn=wrapper_obj.run_imintp,
        inputs=[img0, img1, txt0,],
        outputs=[frame_output, video_output, inv_output],
        cache_examples=cache_examples,)

def interface_iminvs(wrapper_obj):
    with gr.Row():
        image_input = gr.Image(label="Image input", type='pil', elem_id='customized_imbox')
        recon_output = gr.Gallery(label="Reconstruction output", elem_id='customized_imbox')
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label='Text Input', lines=1, placeholder="Input prompt...", )
            with gr.Row():
                button_run = gr.Button("Run")
            
            
        with gr.Column():
            with gr.Row():
                inversion = auto_dropdown('Inversion', choices.inversion, default.inversion)
                inner_step = gr.Slider(label="Inner Step (NTI)", value=default.nullinv_inner_step, minimum=1, maximum=10, step=1)
                force_reinvert = gr.Checkbox(label="Force ReInvert (NTI)", value=False)
            with gr.Accordion('Advanced Settings', open=False):
                with gr.Row():
                    tag_diffuser = auto_dropdown('Diffuser', choices.diffuser, default.diffuser)
                    tag_lora = auto_dropdown('Use LoRA', choices.lora, default.lora)
                    tag_scheduler = auto_dropdown('Scheduler', choices.scheduler, default.scheduler)
                with gr.Row():
                    cfg_scale = gr.Number(label="Scale", minimum=1, maximum=10, value=default.cfg_scale, step=0.5)
                    step = gr.Number(default.step, label="Step", precision=0)
                with gr.Row():
                    force_resize = gr.Checkbox(label="Force Resize", value=True)
                    inp_width = gr.Slider(label="Width", minimum=256, maximum=1024, value=512, step=64)
                    inp_height = gr.Slider(label="Height", minimum=256, maximum=1024, value=512, step=64)
            

    tag_diffuser.change(
        wrapper_obj.load_all,
        inputs = [tag_diffuser, tag_lora, tag_scheduler],
        outputs = [tag_diffuser, tag_lora, tag_scheduler],)

    tag_lora.change(
        wrapper_obj.load_all,
        inputs = [tag_diffuser, tag_lora, tag_scheduler],
        outputs = [tag_diffuser, tag_lora, tag_scheduler],)

    tag_scheduler.change(
        wrapper_obj.load_scheduler,
        inputs = [tag_scheduler],
        outputs = [tag_scheduler],)

    button_run.click(
        wrapper_obj.run_iminvs,
        inputs=[image_input, prompt,  
                cfg_scale, step, 
                force_resize, inp_width, inp_height,
                inversion, inner_step, force_reinvert, 
                tag_diffuser, tag_lora, tag_scheduler,],
        outputs=[recon_output])
    
    gr.Examples(
        label='Examples', 
        examples=get_iminvs_example(), 
        fn=wrapper_obj.run_iminvs,
        inputs=[image_input, prompt,],
        outputs=[recon_output],
        cache_examples=cache_examples,)


def interface_imedit(wrapper_obj):
    with gr.Row():
        image_input = gr.Image(label="Image input", type='pil', elem_id='customized_imbox')
        edited_output = gr.Gallery(label="Edited output", elem_id='customized_imbox')
    with gr.Row():
        with gr.Column():
            prompt_0 = gr.Textbox(label='Source Text', lines=1, placeholder="Source prompt...", )
            prompt_1 = gr.Textbox(label='Target Text', lines=1, placeholder="Target prompt...", )
            with gr.Row():
                button_run = gr.Button("Run")
            
        with gr.Column():
            with gr.Row():
                inversion = auto_dropdown('Inversion', choices.inversion, default.inversion)
                inner_step = gr.Slider(label="Inner Step (NTI)", value=default.nullinv_inner_step, minimum=1, maximum=10, step=1)
                force_reinvert = gr.Checkbox(label="Force ReInvert (NTI)", value=False)
                threshold = gr.Slider(label="Threshold", minimum=0, maximum=1, value=default.threshold, step=0.1)
            with gr.Accordion('Advanced Settings', open=False):
                with gr.Row():
                    tag_diffuser = auto_dropdown('Diffuser', choices.diffuser, default.diffuser)
                    tag_lora = auto_dropdown('Use LoRA', choices.lora, default.lora)
                    tag_scheduler = auto_dropdown('Scheduler', choices.scheduler, default.scheduler)
                with gr.Row():
                    cfg_scale = gr.Number(label="Scale", minimum=1, maximum=10, value=default.cfg_scale, step=0.5)
                    step = gr.Number(default.step, label="Step", precision=0)
                with gr.Row():
                    force_resize = gr.Checkbox(label="Force Resize", value=True)
                    inp_width = gr.Slider(label="Width", minimum=256, maximum=1024, value=512, step=64)
                    inp_height = gr.Slider(label="Height", minimum=256, maximum=1024, value=512, step=64)
            

    tag_diffuser.change(
        wrapper_obj.load_all,
        inputs = [tag_diffuser, tag_lora, tag_scheduler],
        outputs = [tag_diffuser, tag_lora, tag_scheduler],)

    tag_lora.change(
        wrapper_obj.load_all,
        inputs = [tag_diffuser, tag_lora, tag_scheduler],
        outputs = [tag_diffuser, tag_lora, tag_scheduler],)

    tag_scheduler.change(
        wrapper_obj.load_scheduler,
        inputs = [tag_scheduler],
        outputs = [tag_scheduler],)

    button_run.click(
        wrapper_obj.run_imedit,
        inputs=[image_input, prompt_0, prompt_1, 
                threshold, cfg_scale, step, 
                force_resize, inp_width, inp_height,
                inversion, inner_step, force_reinvert, 
                tag_diffuser, tag_lora, tag_scheduler,],
        outputs=[edited_output])
    
    gr.Examples(
        label='Examples', 
        examples=get_imedit_example(), 
        fn=wrapper_obj.run_imedit,
        inputs=[image_input, prompt_0, prompt_1, threshold,],
        outputs=[edited_output],
        cache_examples=cache_examples,)
        

#############
# Interface #
#############

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=None)
    args = parser.parse_args()
    from app_utils import css_empty, css_version_4_11_0
    # css = css_empty
    css = css_version_4_11_0

    wrapper_obj = wrapper(
        fp16=False, 
        tag_diffuser=default.diffuser,
        tag_lora=default.lora,
        tag_scheduler=default.scheduler)

    if True:
        with gr.Blocks(css=css) as demo:
            gr.HTML(
                """
                <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
                <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                    {}
                </h1>
                </div>
                """.format(version))

            with gr.Tab('Image Interpolation'):
                interface_imintp(wrapper_obj)
            with gr.Tab('Image Inversion'):
                interface_iminvs(wrapper_obj)
            with gr.Tab('Image Editing'):
                interface_imedit(wrapper_obj)

        demo.launch(share=True)
