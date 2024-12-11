# DEEPTalk: Dynamic Emotion Embedding for Probabilistic Speech-Driven 3D Face Animation [AAAI2025]
[![arXiv](https://img.shields.io/badge/arXiv-<2408.06010>-red.svg)](https://arxiv.org/abs/2408.06010)
[![ProjectPage](https://img.shields.io/badge/ProjectPage-DEEPTalk-<COLOR>.svg)](https://whwjdqls.github.io/deeptalk.github.io/)

<p align="center">
  <img src="./demo/teaser_final.png" alt="alt text" width="400">
</p>

Official pytorch code release of "[DEEPTalk: Dynamic Emotion Embedding for Probabilistic Speech-Driven 3D Face Animation](https://arxiv.org/abs/2408.06010)"

```
@misc{kim2024deeptalkdynamicemotionembedding,
      title={DEEPTalk: Dynamic Emotion Embedding for Probabilistic Speech-Driven 3D Face Animation}, 
      author={Jisoo Kim and Jungbin Cho and Joonho Park and Soonmin Hwang and Da Eun Kim and Geon Kim and Youngjae Yu},
      year={2024},
      eprint={2408.06010},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.06010}, 
}
```
## News
🛩️ **10/Dec/24** - DEEPTalk is accepted to AAAI2025

## Settings
REPOSITORY UNDER CONSTRUCTION
### Environment
Make environment and install pytorch
```bash
conda create -n deeptalk python=3.9
conda activate deeptalk
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Install pytorch3d and other requirements. Refer to this [page](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for pytorch3d details.
```bash
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html
pip install -r requirements.txt
```
Install osmesa and ffmpeg for headless rendering and audio, video processing. 
```bash
conda install menpo::osmesa
conda install conda-forge::ffmpeg

```
### Download Checkpoints
Download DEE, FER, TH-VQVAE, DEEPTalk checkpoints from [here](https://drive.google.com/drive/u/0/folders/1vmgJCvAq96C83eU4JuUFooubL-y7Py44).
- **DEE.pt**: Place in `./DEE/checkpoint`
- **FER.pth**: Place in `./FER/checkpoint`
- **TH-VQVAE.pth**: Place in `./DEEPTalk/checkpoint/TH-VQVAE`
- **DEEPTalk.pth**: Place in `./DEEPTalk/checkpoint/DEEPTalk`

Download emotion2vec_bast.pt from the [emotion2vec repository](https://huggingface.co/emotion2vec/emotion2vec_base).
- **emotion2vec_base.pt**: Place in `./DEE/models/emo2vec/checkpoint`

Donload FLAME_sample.ply from [Ringnet project](https://github.com/soubhiksanyal/RingNet/tree/master/flame_model).
- **FLAME_sample.ply**: Place in `./DEEPTalk/models/flame_models`
- **flame_dynamic_embedding.npy**: Place in `./DEEPTalk/models/flame_models`
- **flame_static_embedding.pkl**: Place in `./DEEPTalk/models/flame_models`

Download Flame generic_model.pkl from [FLAME website](https://flame.is.tue.mpg.de/).
- **generic_model.pkl**: Place in `./DEEPTalk/models/flame_models`
  
## Inference
```
cd DEEPTalk
python demo.py \
--DEMOTE_ckpt_path ./checkpoint/DEEPTalk/DEEPTalk.pth \
--DEE_ckpt_path ../DEE/checkpoint/DEE.pth \
--audio_path ../demo/sample_audio.wav

```
## Training



## Acknowledgements
We gratefully acknowledge the open-source projects that served as the foundation for our work:

- [EMOTE](https://github.com/radekd91/inferno)
- [learning2listen](https://github.com/evonneng/learning2listen)
- [PCME++](https://github.com/naver-ai/pcmepp)

## License
This code is released under the MIT License.

Please note that our project relies on various other libraries, including FLAME, PyTorch3D, and Spectre, as well as several datasets.
