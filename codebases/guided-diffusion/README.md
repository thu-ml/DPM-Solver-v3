# DPM-Solver-v3 (Guided-Diffusion)

## Preparation

- Download the pretrained models

  ```shell
  mkdir -p ddpm_ckpt/imagenet256
  wget -O ddpm_ckpt/imagenet256/256x256_diffusion.pt https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
  wget -O ddpm_ckpt/imagenet256/256x256_classifier.pt https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
  ```

- Download the folder `imagenet256_guided` from https://drive.google.com/drive/folders/1sWq-htX9c3Xdajmo1BG-QvkbaeVtJqaq and put it under the folder `statistics/`.

- Download the stats file for computing FID

  ```shell
  mkdir -p fid_stats
  wget -O fid_stats/VIRTUAL_imagenet256_labeled.npz https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
  ```

- Install the packages

  ```shell
  pip install PyYAML tqdm scipy pytorch_fid
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
  ```

## Generate Samples and Compute FIDs

Run `bash sample.sh`, and the samples of different samplers under different numbers of steps will be generated under the folder `samples/256x256_diffusion/`. After the samples are generated, their FIDs will be computed and stored in `output.txt`. You can modify the script as you wish.
