# DPM-Solver-v3 (Latent-Diffusion, Stable-Diffusion)

## Preparation

Install the packages

```shell
pip install opencv-python omegaconf tqdm einops pytorch-lightning==1.6.5 transformers kornia
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -e ./src/clip/
pip install -e ./src/taming-transformers/
```


For Latent-Diffusion on LSUN-Bedroom:

- Download the pretrained models

  ```shell
  mkdir -p models/first_stage_models/vq-f4
  wget -O models/first_stage_models/vq-f4/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
  cd models/first_stage_models/vq-f4
  unzip -o model.zip
  cd ../../..
  
  mkdir -p models/ldm/lsun_beds256
  wget -O models/ldm/lsun_beds256/lsun_beds-256.zip https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip
  cd models/ldm/lsun_beds256
  unzip -o lsun_beds-256.zip
  cd ../../..
  ```

- Download the folder `lsun_beds256` from https://drive.google.com/drive/folders/1sWq-htX9c3Xdajmo1BG-QvkbaeVtJqaq and put it under the folder `statistics/`.

For Stable-Diffusion-v1.4:

- Download https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt from [CompVis/stable-diffusion-v-1-4-original · Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) and put it under the folder `models/ldm/stable-diffusion-v1/`.

- Download the folder `sd-v1-4` from https://drive.google.com/drive/folders/1sWq-htX9c3Xdajmo1BG-QvkbaeVtJqaq and put it under the folder `statistics/`.


## Generate Samples

For Latent-Diffusion on LSUN-Bedroom:

- Run `bash sample.sh lsun_beds256 <number-of-steps>`

- For example:

  ```shell
  bash sample.sh lsun_beds256 5
  ```

For Stable-Diffusion-v1.4:

- Run `bash sample.sh sd-v1-4 <number-of-steps> <guidance-scale> <prompt>`

- For example:

  ```shell
  bash sample.sh sd-v1-4 5 7.5 "A beautiful castle beside a waterfall in the woods, by Josef Thoma, matte painting, trending on artstation HQ"
  ```

The samples of different samplers will be generated under the folder `outputs/`. You can modify the script as you wish.

## Compute FID and MSE

TODO