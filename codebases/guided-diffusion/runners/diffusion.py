import os
import logging
import time
import glob
from tkinter import E

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.distributed as dist

from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.guided_diffusion.unet import EncoderUNetModel as GuidedDiffusion_Classifier

import torchvision.utils as tvu


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, rank=None):
        self.args = args
        self.config = config
        assert not config.model.is_upsampling
        if rank is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = rank
            self.rank = rank
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def sample(self):
        if self.config.model.model_type == "guided_diffusion":
            model = GuidedDiffusion_Model(
                image_size=self.config.model.image_size,
                in_channels=self.config.model.in_channels,
                model_channels=self.config.model.model_channels,
                out_channels=self.config.model.out_channels,
                num_res_blocks=self.config.model.num_res_blocks,
                attention_resolutions=self.config.model.attention_resolutions,
                dropout=self.config.model.dropout,
                channel_mult=self.config.model.channel_mult,
                conv_resample=self.config.model.conv_resample,
                dims=self.config.model.dims,
                num_classes=self.config.model.num_classes,
                use_checkpoint=self.config.model.use_checkpoint,
                use_fp16=self.config.model.use_fp16,
                num_heads=self.config.model.num_heads,
                num_head_channels=self.config.model.num_head_channels,
                num_heads_upsample=self.config.model.num_heads_upsample,
                use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                resblock_updown=self.config.model.resblock_updown,
                use_new_attention_order=self.config.model.use_new_attention_order,
            )
        else:
            assert False, "Unknown model type."

        model = model.to(self.rank)
        map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}

        ckpt_dir = os.path.expanduser(self.config.model.ckpt_dir)
        states = torch.load(ckpt_dir, map_location=map_location)
        model.load_state_dict(states, strict=True)
        if self.config.model.use_fp16:
            model.convert_to_fp16()

        if self.config.sampling.cond_class:
            classifier = GuidedDiffusion_Classifier(
                image_size=self.config.classifier.image_size,
                in_channels=self.config.classifier.in_channels,
                model_channels=self.config.classifier.model_channels,
                out_channels=self.config.classifier.out_channels,
                num_res_blocks=self.config.classifier.num_res_blocks,
                attention_resolutions=self.config.classifier.attention_resolutions,
                channel_mult=self.config.classifier.channel_mult,
                use_fp16=self.config.classifier.use_fp16,
                num_head_channels=self.config.classifier.num_head_channels,
                use_scale_shift_norm=self.config.classifier.use_scale_shift_norm,
                resblock_updown=self.config.classifier.resblock_updown,
                pool=self.config.classifier.pool,
            )
            ckpt_dir = os.path.expanduser(self.config.classifier.ckpt_dir)
            states = torch.load(
                ckpt_dir,
                map_location=map_location,
            )
            classifier = classifier.to(self.rank)
            classifier.load_state_dict(states, strict=True)
            if self.config.classifier.use_fp16:
                classifier.convert_to_fp16()
        else:
            classifier = None

        model.eval()

        print("Model loaded.")

        if self.args.sample_type == "dpmsolver_v3":
            from samplers.dpm_solver_v3 import NoiseScheduleVP, DPM_Solver_v3

            stats_dir = self.args.statistics_dir

            assert stats_dir is not None, "No statistics file found."
            print("Use statistics", stats_dir)

            self.noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
            self.dpm_solver_v3 = DPM_Solver_v3(
                stats_dir,
                self.noise_schedule,
                steps=self.args.timesteps,
                skip_type=self.args.skip_type,
                degenerated=False,
            )
        print("Begin sampling")
        self.sample_fid(model, classifier=classifier)

    def sample_fid(self, model, classifier=None):
        config = self.config
        total_n_samples = config.sampling.fid_total_samples
        world_size = torch.cuda.device_count()
        if total_n_samples % (config.sampling.batch_size * world_size) != 0:
            raise ValueError(
                "Total samples for sampling must be divided exactly by config.sampling.batch_size and world size, but got {} and {} {}".format(
                    total_n_samples, config.sampling.batch_size, world_size
                )
            )
        if len(glob.glob(f"{self.args.image_folder}/*.png")) == total_n_samples:
            return
        else:
            n_rounds = total_n_samples // config.sampling.batch_size // world_size
        img_id = self.rank * total_n_samples // world_size

        with torch.no_grad():
            for _ in tqdm.tqdm(range(n_rounds), desc="Generating image samples for FID evaluation."):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x, classes = self.sample_image(x, model, classifier=classifier)
                x = inverse_data_transform(config, x)
                for i in range(x.shape[0]):
                    if classes is None:
                        path = os.path.join(self.args.image_folder, f"{img_id}.png")
                    else:
                        path = os.path.join(self.args.image_folder, f"{img_id}_{int(classes.cpu()[i])}.png")
                    tvu.save_image(x.cpu()[i], path)
                    img_id += 1

    def sample_image(self, x, model, last=True, classifier=None, base_samples=None):
        assert last
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        classifier_scale = self.config.sampling.classifier_scale if self.args.scale is None else self.args.scale
        if self.config.sampling.cond_class:
            if self.args.fixed_class is None:
                classes = torch.randint(low=0, high=self.config.data.num_classes, size=(x.shape[0],)).to(x.device)
            else:
                classes = torch.randint(
                    low=self.args.fixed_class, high=self.args.fixed_class + 1, size=(x.shape[0],)
                ).to(x.device)
        else:
            classes = None

        if classes is None:
            model_kwargs = {}
        else:
            model_kwargs = {"y": classes}

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out

            xs, _ = generalized_steps(
                x,
                seq,
                model_fn,
                self.betas,
                eta=self.args.eta,
                classifier=classifier,
                is_cond_classifier=self.config.sampling.cond_class,
                classifier_scale=classifier_scale,
                **model_kwargs,
            )
            x = xs[-1]
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out

            xs, _ = ddpm_steps(
                x,
                seq,
                model_fn,
                self.betas,
                classifier=classifier,
                is_cond_classifier=self.config.sampling.cond_class,
                classifier_scale=classifier_scale,
                **model_kwargs,
            )
            x = xs[-1]
        elif self.args.sample_type in ["dpmsolver", "dpmsolver++"]:
            from samplers.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]

            noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
            model_fn_continuous = model_wrapper(
                model_fn,
                noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            dpm_solver = DPM_Solver(
                model_fn_continuous,
                noise_schedule,
                algorithm_type=self.args.sample_type,
                correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
            )
            x = dpm_solver.sample(
                x,
                steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                order=self.args.order,
                skip_type=self.args.skip_type,
                lower_order_final=self.args.lower_order_final,
                denoise_to_zero=self.args.denoise,
            )
        elif self.args.sample_type == "unipc":
            from samplers.uni_pc import NoiseScheduleVP, model_wrapper, UniPC

            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]

            noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
            model_fn_continuous = model_wrapper(
                model_fn,
                noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            unipc = UniPC(
                model_fn_continuous,
                noise_schedule,
                algorithm_type="data_prediction",
                correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
            )
            x = unipc.sample(
                x,
                steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                order=self.args.order,
                skip_type=self.args.skip_type,
                lower_order_final=self.args.lower_order_final,
                denoise_to_zero=self.args.denoise,
            )
        elif self.args.sample_type == "dpmsolver_v3":
            from samplers.dpm_solver_v3 import model_wrapper

            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]

            model_fn_continuous = model_wrapper(
                model_fn,
                self.noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            x = self.dpm_solver_v3.sample(
                x,
                model_fn_continuous,
                order=self.args.order,
                p_pseudo=False,
                use_corrector=True,
                c_pseudo=True,
                lower_order_final=self.args.lower_order_final,
            )
        else:
            raise NotImplementedError
        return x, classes
