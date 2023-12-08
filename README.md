# Smooth Diffusion

This repository is the official Pytorch implementation for [Smooth Diffusion](https://arxiv.org/pdf/2312).

[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://shi-labs.github.io/Smooth-Diffusion/) [![arXiv](https://img.shields.io/badge/arXiv-SmoothDiffusion-b31b1b.svg)](https://arxiv.org/abs/2312.04410) 

> **Smooth Diffusion: Crafting Smooth Latent Spaces in Diffusion Models**  
> [Jiayi Guo](https://www.jiayiguo.net)\*,
> [Xingqian Xu](https://www.linkedin.com/in/xingqian-xu-97b46526/)\*,
> [Yifan Pu](https://scholar.google.com/citations?user=oM9rnYQAAAAJ&hl=en),
> [Zanlin Ni](https://scholar.google.com/citations?user=Yibz_asAAAAJ&hl=en),
> [Chaofei Wang](https://scholar.google.com/citations?user=-hwGMHcAAAAJ&hl=en),
> [Manushree Vasu](https://in.linkedin.com/in/v-manushree),
> [Shiji Song](https://scholar.google.com/citations?user=rw6vWdcAAAAJ&hl=en&oi=ao),
> [Gao Huang](https://www.gaohuang.net),
> [Humphrey Shi](https://www.humphreyshi.com)


https://github.com/SHI-Labs/Smooth-Diffusion/assets/53193040/b5fcf520-1b8b-4f17-97a3-1a06d440ec25


<p align="center">
<strong>Smooth Diffusion</strong> is a new category of diffusion models that is simultaneously high-performing and smooth.
</p>

<p align="center">
<img src="asserts/Picture1.jpg" width="1080px"/>
Our method formally introduces latent space smoothness to diffusion models like Stable Diffusion. This smoothness dramatically aids in: 1) improving the continuity of transitions in image interpolation, 2) reducing approximation errors in image inversion, and 3) better preserving unedited contents in image editing.
</p>

## News
- [2023.12.08] Paper released!

## ToDo
- [ ] Release code and model weights
- [ ] Gradio Demo


## Overview
<p align="center">
<img src="asserts/Picture2.jpg" width="1080px"/>
<strong>Smooth Diffusion</strong> (c) enforces the ratio between the variation of the input latent and the variation of the output prediction is a constant. We propose <strong>Training-time Smooth Diffusion</strong> (d) to optimize a "single-step snapshot" of the variation constraint in (c). DM: Diffusion model. Please refer to our paper for additional details.
</p>

## Code
Coming soon.


## Visualizations
### Image Interpolation  
  
> Using the Smooth LoRA trained atop Stable Diffusion V1.5.

<p align="center">
<img src="asserts/Picture3.jpg" width="1080px"/>
</p>

> Intergrating the above Smooth LoRA into other community models.

<p align="center">
<img src="asserts/Picture4.jpg" width="1080px"/>
</p>

### Image Inversion

<p align="center">
<img src="asserts/Picture5.jpg" width="1080px"/>
</p>

### Image Editing

<p align="center">
<img src="asserts/Picture6.jpg" width="1080px"/>
</p>

## Citation

If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. Thanks for your support!

```
@article{guo2023smooth,
  title={Smooth Diffusion: Crafting Smooth Latent Spaces in Diffusion Models},
  author={Jiayi Guo and Xingqian Xu and Yifan Pu and Zanlin Ni and Chaofei Wang and Manushree Vasu and Shiji Song and Gao Huang and Humphrey Shi},
  journal={arXiv preprint arXiv:2312.04410},
  year={2023}
}
```


## Contact
guo-jy20 at mails dot tsinghua dot edu dot cn
