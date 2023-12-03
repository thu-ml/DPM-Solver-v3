# DPM-Solver-v3
This repo is the official code for the paper [DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics](https://openreview.net/forum?id=9fWKExmKa0) (NeurIPS 2023).

<h3><a href="https://ml.cs.tsinghua.edu.cn/dpmv3/">Project Page</a> | <a href="https://arxiv.org/pdf/2310.13268.pdf">Paper</a> | <a href="https://arxiv.org/abs/2310.13268">arXiv</a></h3>

DPM-Solver-v3 is a training-free ODE solver dedicated to fast sampling of diffusion models, equipped with precomputed *empirical model statistics (EMS)* to boost the convergence speed up to 40%. DPM-Solver-v3 brings especially notable and non-trivial quality improvement in few-step sampling (5~10 steps).

*Please refer to the paper and project page for detailed methods and results.*

## Code Examples

We integrate DPM-Solver-v3 into various codebases, and the previous state-of-the-art samplers [DPM-Solver++](https://github.com/LuChengTHU/dpm-solver) and [UniPC](https://github.com/wl-zhao/UniPC) are also included, enabling convenient benchmarking and comparisons.

We put the code examples in `codebases/`. The experiment results reported in the paper can be reproduced by them.

|       Name       |                     Original Repository                      |               Pretrained Models               |            Dataset            |                    Type                    |
| :--------------: | :----------------------------------------------------------: | :-------------------------------------------: | :---------------------------: | :----------------------------------------: |
|    score_sde     |        https://github.com/yang-song/score_sde_pytorch        | `cifar10_ddpmpp_deep_continuous-checkpoint_8` |           CIFAR-10            |             Uncond/Pixel-Space             |
|       edm        |                https://github.com/NVlabs/edm                 |         `edm-cifar10-32x32-uncond-vp`         |           CIFAR-10            |             Uncond/Pixel-Space             |
| guided-diffusion |          https://github.com/openai/guided-diffusion          |    `256x256_diffusion/256x256_classifier`     |         ImageNet-256          |              Cond/Pixel-Space              |
| stable-diffusion | https://github.com/CompVis/latent-diffusion<br />https://github.com/CompVis/stable-diffusion |      `lsun_beds256-model`<br />`sd-v1-4`      | LSUN-Bedroom<br />MS-COCO2014 | Uncond/Latent-Space<br />Cond/Latent-Space |

## Documentation

We provide the PyTorch implementation of DPM-Solver-v3 in a single file `dpm_solver_v3.py`. We suggest referring to the code examples for its practical usage in different settings.

To use DPM-Solver-v3, one can follow the steps below. Special thanks to [DPM-Solver](https://github.com/LuChengTHU/dpm-solver) for their unified model wrapper to support various diffusion models.

### 1. Define Noise Schedule

The *noise schedule* $\alpha_t,\sigma_t$ defines the forward transition kernel from time $0$ to time $t$:
$$
p(x_t|x_0)=\mathcal N(x_t;\alpha_tx_0,\sigma_t^2I)
$$
or equivalently
$$
x_t=\alpha_tx_0+\sigma_t\epsilon,\quad \epsilon\sim\mathcal N(0,I)
$$
We support two main class of noise schedules:

|                Name                 |    Python Class    |        Definition         |        Type         |
| :---------------------------------: | :----------------: | :-----------------------: | :-----------------: |
|      Variance Preserving (VP)       | `NoiseScheduleVP`  | $\alpha_t^2+\sigma_t^2=1$ | discrete/continuous |
| EDM (https://github.com/NVlabs/edm) | `NoiseScheduleEDM` |  $\alpha_t=1,\sigma_t=t$  |     continuous      |

#### 1.1. Discrete-time DPMs

##### VP

We support a picewise linear interpolation of $\log\alpha_{t}$  in the `NoiseScheduleVP` class to convert discrete noise schedules to continuous noise schedules.

We need either the $\beta_i$ array or the $\bar{\alpha}_i$ array (see [DDPM](https://arxiv.org/abs/2006.11239) for details) to define the noise schedule. The detailed relationship is:
$$
\bar{\alpha}_i = \prod (1 - \beta_k)
$$

$$
\alpha_{t_i} = \sqrt{\bar{\alpha}_i}
$$

Define the discrete-time noise schedule by the $\beta_i$ array:

```python
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
```

Or define the discrete-time noise schedule by the $\bar{\alpha}_i$ array:
```python
noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=alphas_cumprod)
```

#### 1.2. Continuous-time DPMs

##### VP

We support both linear schedule and cosine schedule for the continuous-time DPMs.

|    Name     |                          $\alpha_t$                          | Example Paper                                                |
| :---------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| VP (linear) |  $e^{-\frac{1}{4}(\beta_1-\beta_0)t^2-\frac{1}{2}\beta_0t}$  | [DDPM](https://arxiv.org/abs/2006.11239),[ScoreSDE](https://arxiv.org/abs/2011.13456) |
| VP (cosine) | $\frac{f(t)}{f(0)}$ ($f(t)=\cos\left(\frac{t+s}{1+s}\frac{\pi}{2}\right)$) | [improved-DDPM](https://arxiv.org/abs/2102.09672)            |

Define the continuous-time linear noise schedule with $\beta_0=0.1,\beta_1=20$:
```python
noise_schedule = NoiseScheduleVP(schedule='linear', continuous_beta_0=0.1, continuous_beta_1=20.)
```

Define the continuous-time cosine noise schedule with $s=0.008$:
```python
noise_schedule = NoiseScheduleVP(schedule='cosine')
```

##### EDM

```python
noise_schedule = NoiseScheduleEDM()
```

### 2. Define Model Wrapper

For a given diffusion `model` with an input of the time label
(may be discrete-time labels (i.e. 0 to 999) or continuous-time times (i.e. 0 to 1)), and the output type of the model may be "noise" or "x_start" or "v" or "score" (see `Model Types`), we wrap the model function to the following format:

```python
model_fn(x, t_continuous) -> noise
```

where `t_continuous` is the continuous time labels (i.e. 0 to 1), and the output type of the model is "noise", i.e. a noise prediction model. The wrapped continuous-time noise prediction model `model_fn`  is used for DPM-Solver-v3.

Note that DPM-Solver-v3 only needs the noise prediction model (the $\epsilon_\theta(x_t, t)$ model, also as known as the "mean" model), so for diffusion models which predict both "mean" and "variance" (such as [improved-DDPM](https://arxiv.org/abs/2102.09672)), you need to firstly define another function by yourself to only output the "mean".

#### Model Types
We support the following four types of diffusion models. You can set the model type by the argument `model_type` in the function `model_wrapper`.

| Model Type                                        | Training Objective                                           | Example Paper                                                      |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| "noise": noise prediction model $\epsilon_\theta$ | $E_{x_{0},\epsilon,t}\left[\omega_1(t)\|\|\epsilon_\theta(x_t,t)-\epsilon\|\|_2^2\right]$ | [DDPM](https://arxiv.org/abs/2006.11239), [Stable-Diffusion](https://github.com/CompVis/stable-diffusion) |
| "x_start": data prediction model $x_\theta$       | $E_{x_0,\epsilon,t}\left[\omega_2(t)\|\|x_\theta(x_t,t)-x_0\|\|_2^2\right]$ | [DALL·E 2](https://arxiv.org/abs/2204.06125)                 |
| "v": velocity prediction model $v_\theta$         | $E_{x_0,\epsilon,t}\left[\omega_3(t)\|\|v_\theta(x_t,t)-(\alpha_t\epsilon - \sigma_t x_0)\|\|_2^2\right]$ | [Imagen Video](https://arxiv.org/abs/2210.02303)             |
| "score": marginal score function $s_\theta$       | $E_{x_0,\epsilon,t}\left[\omega_4(t)\|\|\sigma_t s_\theta(x_t,t)+\epsilon\|\|_2^2\right]$ | [ScoreSDE](https://arxiv.org/abs/2011.13456)                 |

#### Sampling Types
We support the following three types of sampling by diffusion models. You can set the argument `guidance_type` in the function `model_wrapper`.

| Sampling Type                                     | Equation for Noise Prediction Model                          | Example Paper                                                |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| "uncond": unconditional sampling                  | $\tilde\epsilon_\theta(x_t,t)=\epsilon_\theta(x_t,t)$        | [DDPM](https://arxiv.org/abs/2006.11239)                     |
| "classifier": classifier guidance                 | $\tilde\epsilon_\theta(x_t,t,c)=\epsilon_\theta(x_t,t)-s\cdot\sigma_t\nabla_{x_t}\log q_\phi(x_t,t,c)$ | [ADM](https://arxiv.org/abs/2105.05233), [GLIDE](https://arxiv.org/abs/2112.10741) |
| "classifier-free": classifier-free guidance (CFG) | $\tilde\epsilon_\theta(x_t,t,c)=s\cdot \epsilon_\theta(x_t,t,c)+(1-s)\cdot\epsilon_\theta(x_t,t)$ | [DALL·E 2](https://arxiv.org/abs/2204.06125), [Imagen](https://arxiv.org/abs/2205.11487), [Stable-Diffusion](https://github.com/CompVis/stable-diffusion) |

#### 2.1. Sampling without Guidance (Unconditional)
The given `model` has the following format:
```python
model(x_t, t_input, **model_kwargs) -> noise | x_start | v | score
```

We wrap the model by:

```python
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type=model_type,  # "noise" or "x_start" or "v" or "score"
    model_kwargs=model_kwargs,  # additional inputs of the model
)
```
#### 2.2. Sampling with Classifier Guidance (Conditional)
The given `model` has the following format:
```python
model(x_t, t_input, **model_kwargs) -> noise | x_start | v | score
```
For DPMs with classifier guidance, we also combine the model output with the classifier gradient. We need to specify the classifier function and the guidance scale. The classifier function has the following format:
```python
classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
```
where `t_input` is the same time label as in the original diffusion model `model`, and `cond` is the condition (such as class labels).

We wrap the model by:

```python
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type=model_type,  # "noise" or "x_start" or "v" or "score"
    model_kwargs=model_kwargs,  # additional inputs of the model
    guidance_type="classifier",
    condition=condition,  # conditional input of the classifier
    guidance_scale=guidance_scale,  # classifier guidance scale
    classifier_fn=classifier,
    classifier_kwargs=classifier_kwargs,  # other inputs of the classifier function
)
```
#### 2.3. Sampling with Classifier-free Guidance (Conditional)
The given `model` has the following format:
```python
model(x_t, t_input, cond, **model_kwargs) -> noise | x_start | v | score
```
Note that for classifier-free guidance, the model needs another input `cond` (such as the text prompt). If `cond` is a special variable `unconditional_condition` (such as the empty text `""`), then the model output is the unconditional DPM output.

We wrap the model by:

```python
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type=model_type,  # "noise" or "x_start" or "v" or "score"
    model_kwargs=model_kwargs,  # additional inputs of the model
    guidance_type="classifier-free",
    condition=condition,  # conditional input
    unconditional_condition=unconditional_condition,  # special unconditional condition variable for the unconditional model
    guidance_scale=guidance_scale,  # classifier-free guidance scale
)
```
### 3. Define DPM-Solver-v3 and Sample

After defining `noise_schedule` and `model_fn`, we can further use them to define DPM-Solver-v3 and generate samples.

First we define the DPM-Solver-v3 instance `dpm_solver_v3`, and it will automatically handle some necessary preprocessing.

```python
dpm_solver_v3 = DPM_Solver_v3(
    statistics_dir="statistics/sd-v1-4/7.5_250_1024", 
    noise_schedule, 
    steps=10, 
    t_start=1.0, 
    t_end=1e-3, 
    skip_type="time_uniform", 
    degenerated=False,
)
```

- `statistics_dir`: the directory which stores the computed EMS (i.e. the three coefficients $l,s,b$ in the paper). The EMS used in the paper can be reached at https://drive.google.com/drive/folders/1sWq-htX9c3Xdajmo1BG-QvkbaeVtJqaq. For your own models, you need to follow *Appendix C.1.1* in the paper to compute their EMS.
- `steps`: the number of steps for samping. Since DPM-Solver-v3 uses multistep method, the total number of function evaluations (NFE) is equal to `steps`.
- `t_start` and `t_end`: we sample from time `t_start` to time `t_end`.
  - For discrete-time DPMs, we do not need to specify the `t_start` and `t_end`. The default setting is to sample from the discrete-time label $N-1$ to the discrete-time label $0$.
  - For continuous-time DPMs (VP), we sample from `t_start=1.0` (the default setting) to `t_end`. We recommend `t_end=1e-3` for `steps <= 15`, and `t_end=1e-4` for `steps > 15`. For continuous-time DPMs (EDM), we can follow the training setting of [EDM](https://github.com/NVlabs/edm) (i.e. `t_start=80.0` and `t_end=0.002`).

- `skip_type`: the timestep schedule for sampling (i.e. how we discretize the time from `t_start` to `t_end`). We support 4 types of `skip_type`:
  - `logSNR`: uniform logSNR for the time steps. **Recommended for low-resolutional images**.
  - `time_uniform`: uniform time for the time steps. **Recommended for high-resolutional images**.

  - `time_quadratic`: quadratic time for the time steps.

  - `edm`: geometric SNR for the time steps, proposed by [EDM](https://github.com/NVlabs/edm). In our experiments, we find it not as good as uniform logSNR or uniform time.

- `degenerated`: degenerate the EMS to suboptimal choice $l=1,s=0,b=0$, which corresponds to DPM-Solver++. See *Appendix A.2* in the paper for details.

Then we can use `dpm_solver_v3.sample` to quickly sample from DPMs. This function computes the ODE solution at time `t_end` by DPM-Solver-v3, given the initial `x_T` at time `t_start`.

```python
x_sample = dpm_solver_v3.sample(
    x_T,
    model_fn,
    order=3,
    p_pseudo=False,
    use_corrector=True,
    c_pseudo=True,
    lower_order_final=True,
    half=False,
)
```

- `order`: the order of DPM-Solver-v3. We recommend using `order=3` for unconditional sampling and `order=2` for conditional sampling.

- `p_pseudo`: whether to use pseudo-order predictor. Only enabled in few cases (5 steps on CIFAR-10).

- `use_corrector`: whether to use corrector. Corrector may make the samples sharper but distorted. We recommend turning it off when the number of steps is large.

- `c_pseudo`: whether to use pseudo-order corrector of `order+1`; otherwise, use corrector of `order`. Switching it on is better in most cases.

- `lower_order_final`: whether to use lower-order solver at the last steps. We recommend turning it on to make the sampling more stable at the expense of less details, otherwise the samples may be broken.

- `half`: only use corrector in the time region $t\leq 0.5$. May bring improvement when the guidance scale is large.


## EMS Computing by Yourself

TODO

## Acknowledgement

Special thanks to [DPM-Solver and DPM-Solver++](https://github.com/LuChengTHU/dpm-solver) for their unified model wrapper to support various diffusion models.

The predictor-corrector method is inspired by [UniPC](https://github.com/wl-zhao/UniPC).

We use the pretrained diffusion models and codebases provided by:

[ScoreSDE](https://github.com/yang-song/score_sde_pytorch), [EDM](https://github.com/NVlabs/edm), [Guided-Diffusion](https://github.com/openai/guided-diffusion), [Latent-Diffusion](https://github.com/CompVis/latent-diffusion), [Stable-Diffusion](https://github.com/CompVis/stable-diffusion)

## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{zheng2023dpm,
	title={DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics},
	author={Zheng, Kaiwen and Lu, Cheng and Chen, Jianfei and Zhu, Jun},
	booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
	year={2023}
}
```
