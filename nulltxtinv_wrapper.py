import numpy as np
import torch
import PIL.Image
from tqdm import tqdm
from typing import Optional, Union, List
import warnings
warnings.filterwarnings('ignore')

from torch.optim.adam import Adam
import torch.nn.functional as nnf

from diffusers import DDIMScheduler

##########
# helper #
##########

def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents

def image2latent(vae, image):
    with torch.no_grad():
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
        if isinstance(image, np.ndarray):
            dtype = next(vae.parameters()).dtype
            device = next(vae.parameters()).device
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
        latents = vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
    return latents

def latent2image(vae, latents, return_type='np'):
    assert isinstance(latents, torch.Tensor)
    latents = 1 / 0.18215 * latents.detach()
    image = vae.decode(latents)['sample']
    if return_type in ['np', 'pil']:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        if return_type == 'pil':
            pilim = [PIL.Image.fromarray(imi) for imi in image]
            pilim = pilim[0] if len(pilim)==1 else pilim
            return pilim
        else:
            return image

def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

def txt_to_emb(model, prompt):
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",)
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_embeddings

@torch.no_grad()
def text2image_ldm(
        model,
        prompt:  List[str],
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        start_time=50,
        return_type='pil', ):

    batch_size = len(prompt)
    height = width = 512
    if latent is not None:
        height = latent.shape[-2] * 8
        width = latent.shape[-1] * 8
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",)
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt",)
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = diffusion_step(model, latents, context, t, guidance_scale, low_resource=False)

    if return_type in ['pil', 'np']:
        image = latent2image(model.vae, latents, return_type=return_type)
    else:
        image = latents
    return image, latent

@torch.no_grad()
def text2image_ldm_imedit(
    model,
    thresh,
    prompt:  List[str],
    target_prompt:  List[str],
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='pil'
):
    batch_size = len(prompt)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    target_text_input = model.tokenizer(
        target_prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    target_text_embeddings = model.text_encoder(target_text_input.input_ids.to(model.device))[0]

    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if i < (1 - thresh) * num_inference_steps:
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            latents = diffusion_step(model, latents, context, t, guidance_scale, low_resource=False)
        else:
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*target_text_embeddings.shape), target_text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, target_text_embeddings])
            latents = diffusion_step(model, latents, context, t, guidance_scale, low_resource=False)

    if return_type in ['pil', 'np']:
        image = latent2image(model.vae, latents, return_type=return_type)
    else:
        image = latents
    return image, latent


###########
# wrapper #
###########

class NullInversion(object):
    def __init__(self, model, num_ddim_steps, guidance_scale, device='cuda'):
        self.model = model
        self.device = device
        self.num_ddim_steps=num_ddim_steps
        self.guidance_scale = guidance_scale
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, noise_pred, timestep, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, emb):
        # uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, emb)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_invert(self, image, prompt):
        assert isinstance(image, PIL.Image.Image)

        scheduler_save = self.model.scheduler
        self.model.scheduler = DDIMScheduler.from_config(self.model.scheduler.config)
        self.model.scheduler.set_timesteps(self.num_ddim_steps)

        with torch.no_grad():
            emb = txt_to_emb(self.model, prompt)
            latent = image2latent(self.model.vae, image)
        ddim_latents = self.ddim_loop(latent, emb)

        self.model.scheduler = scheduler_save
        return ddim_latents[-1]

    def null_optimization(self, latents, emb, nemb=None, num_inner_steps=10, epsilon=1e-5):
        # force fp32
        dtype = latents[0].dtype
        uncond_embeddings = nemb.float() if nemb is not None else txt_to_emb(self.model, "").float()
        cond_embeddings = emb.float()
        latents = [li.float() for li in latents]
        self.model.unet.to(torch.float32)

        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.num_ddim_steps)
        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()

        uncond_embeddings_list = [ui.to(dtype) for ui in uncond_embeddings_list]
        self.model.unet.to(dtype)
        return uncond_embeddings_list

    def null_invert(self, im, txt, ntxt=None, num_inner_steps=10, early_stop_epsilon=1e-5):
        assert isinstance(im, PIL.Image.Image)

        scheduler_save = self.model.scheduler
        self.model.scheduler = DDIMScheduler.from_config(self.model.scheduler.config)
        self.model.scheduler.set_timesteps(self.num_ddim_steps)

        with torch.no_grad():
            nemb = txt_to_emb(self.model, ntxt) \
                if ntxt is not None else txt_to_emb(self.model, "")
            emb  = txt_to_emb(self.model, txt) 
            latent = image2latent(self.model.vae, im)

        # ddim inversion
        ddim_latents = self.ddim_loop(latent, emb)
        # nulltext inversion
        uncond_embeddings = self.null_optimization(
            ddim_latents, emb, nemb, num_inner_steps, early_stop_epsilon)

        self.model.scheduler = scheduler_save
        return ddim_latents[-1], uncond_embeddings

    def null_optimization_dual(
            self, latents0, latents1, emb0, emb1, nemb=None, 
            num_inner_steps=10, epsilon=1e-5):

        # force fp32
        dtype = latents0[0].dtype
        uncond_embeddings = nemb.float() if nemb is not None else txt_to_emb(self.model, "").float()
        cond_embeddings0, cond_embeddings1 = emb0.float(), emb1.float()
        latents0 = [li.float() for li in latents0]
        latents1 = [li.float() for li in latents1]
        self.model.unet.to(torch.float32)
        
        uncond_embeddings_list = []
        latent_cur0 = latents0[-1]
        latent_cur1 = latents1[-1]

        bar = tqdm(total=num_inner_steps * self.num_ddim_steps)
        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))

            latent_prev0 = latents0[len(latents0) - i - 2]
            latent_prev1 = latents1[len(latents1) - i - 2]

            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond0 = self.get_noise_pred_single(latent_cur0, t, cond_embeddings0)
                noise_pred_cond1 = self.get_noise_pred_single(latent_cur1, t, cond_embeddings1)
            for j in range(num_inner_steps):
                noise_pred_uncond0 = self.get_noise_pred_single(latent_cur0, t, uncond_embeddings)
                noise_pred_uncond1 = self.get_noise_pred_single(latent_cur1, t, uncond_embeddings)

                noise_pred0 = noise_pred_uncond0 + self.guidance_scale*(noise_pred_cond0-noise_pred_uncond0)
                noise_pred1 = noise_pred_uncond1 + self.guidance_scale*(noise_pred_cond1-noise_pred_uncond1)

                latents_prev_rec0 = self.prev_step(noise_pred0, t, latent_cur0)
                latents_prev_rec1 = self.prev_step(noise_pred1, t, latent_cur1)

                loss = nnf.mse_loss(latents_prev_rec0, latent_prev0) + \
                       nnf.mse_loss(latents_prev_rec1, latent_prev1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())

            with torch.no_grad():
                context0 = torch.cat([uncond_embeddings, cond_embeddings0])
                context1 = torch.cat([uncond_embeddings, cond_embeddings1])
                latent_cur0 = self.get_noise_pred(latent_cur0, t, False, context0)
                latent_cur1 = self.get_noise_pred(latent_cur1, t, False, context1)

        bar.close()

        uncond_embeddings_list = [ui.to(dtype) for ui in uncond_embeddings_list]
        self.model.unet.to(dtype)
        return uncond_embeddings_list

    def null_invert_dual(
            self, im0, im1, txt0, txt1, ntxt=None, 
            num_inner_steps=10, early_stop_epsilon=1e-5, ):
        assert isinstance(im0, PIL.Image.Image)
        assert isinstance(im1, PIL.Image.Image)

        scheduler_save = self.model.scheduler
        self.model.scheduler = DDIMScheduler.from_config(self.model.scheduler.config)
        self.model.scheduler.set_timesteps(self.num_ddim_steps)

        with torch.no_grad():
            nemb = txt_to_emb(self.model, ntxt) \
                if ntxt is not None else txt_to_emb(self.model, "")
            latent0 = image2latent(self.model.vae, im0)
            latent1 = image2latent(self.model.vae, im1)
            emb0 = txt_to_emb(self.model, txt0)
            emb1 = txt_to_emb(self.model, txt1)

        # ddim inversion
        ddim_latents_0 = self.ddim_loop(latent0, emb0)
        ddim_latents_1 = self.ddim_loop(latent1, emb1)

        # nulltext inversion
        nembs = self.null_optimization_dual(
            ddim_latents_0, ddim_latents_1, emb0, emb1, nemb, num_inner_steps, early_stop_epsilon)

        self.model.scheduler = scheduler_save
        return ddim_latents_0[-1], ddim_latents_1[-1], nembs
