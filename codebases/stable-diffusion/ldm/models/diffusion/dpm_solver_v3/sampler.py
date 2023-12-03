"""SAMPLING ONLY."""

import torch

from .dpm_solver_v3 import NoiseScheduleVP, model_wrapper, DPM_Solver_v3


class DPMSolverv3Sampler:
    def __init__(self, ckp_path, stats_dir, model, steps, guidance_scale, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.alphas_cumprod = to_torch(model.alphas_cumprod)
        self.device = self.model.betas.device
        self.guidance_scale = guidance_scale

        self.ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

        assert stats_dir is not None, f"No statistics file found in {stats_base}."
        print("Use statistics", stats_dir)
        self.dpm_solver_v3 = DPM_Solver_v3(
            statistics_dir=stats_dir,
            noise_schedule=self.ns,
            steps=steps,
            t_start=None,
            t_end=None,
            skip_type="time_uniform",
            degenerated=False,
            device=self.device,
        )
        self.steps = steps

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        shape,
        conditioning=None,
        x_T=None,
        unconditional_conditioning=None,
        use_corrector=False,
        half=False,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        if x_T is None:
            img = torch.randn(size, device=self.device)
        else:
            img = x_T

        if conditioning is None:
            model_fn = model_wrapper(
                lambda x, t, c: self.model.apply_model(x, t, c),
                self.ns,
                model_type="noise",
                guidance_type="uncond",
            )
            ORDER = 3
        else:
            model_fn = model_wrapper(
                lambda x, t, c: self.model.apply_model(x, t, c),
                self.ns,
                model_type="noise",
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=self.guidance_scale,
            )
            ORDER = 2

        x = self.dpm_solver_v3.sample(
            img,
            model_fn,
            order=ORDER,
            p_pseudo=False,
            c_pseudo=True,
            lower_order_final=True,
            use_corrector=use_corrector,
            half=half,
        )

        return x.to(self.device), None
