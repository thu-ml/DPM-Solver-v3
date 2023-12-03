# DPM-Solver-v3 (EDM)
## Preparation

To generate samples:

- Download the pretrained models

  ```shell
  mkdir -p pretrained
  wget -O pretrained/edm-cifar10-32x32-uncond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
  ```

- Download the folder `edm-cifar10-32x32-uncond-vp` from https://drive.google.com/drive/folders/1sWq-htX9c3Xdajmo1BG-QvkbaeVtJqaq and put it under the folder `statistics/`.

- Install the packages

  ```shell
  pip install absl-py
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
  ```

To compute FIDs:

- Download `cifar10_stats.npz` from https://drive.google.com/drive/folders/1bofxWSwcoVGRqsUnAGUbco1z5lwP0Rb6 and put it under the folder `assets/stats/`

- Install the packages

  ```shell
  pip install tqdm tensorflow==2.11.0 tensorflow_probability==0.19.0 tensorflow-gan
  ```

## Generate Samples

Run `bash sample.sh`, and the samples of different samplers under different numbers of steps will be generated under the folder `samples/edm-cifar10-32x32-uncond-vp/`. You can modify the script as you wish.

## Compute FIDs

Run `python compute_fid.py`, and the FIDs of the generated samples will be computed and stored to `output.txt`. You can modify the script as you wish.

