import torch
import torch.nn.functional as F
import math
from .utils import expand_dims
import numpy as np


class Heun:
    def __init__(self, noise_schedule):
        self.noise_schedule = noise_schedule

    def model_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
            return t
        elif skip_type == "edm":
            rho = 7.0  # 7.0 is the value used in the paper

            sigma_min: float = t_0
            sigma_max: float = t_T
            ramp = np.linspace(0, 1, N + 1)
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            lambdas = torch.Tensor(-np.log(sigmas)).to(device)
            t = self.noise_schedule.inverse_lambda(lambdas)
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type)
            )

    def sample(
        self,
        model_fn,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        skip_type="time_uniform",
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        t_0 = t_end
        t_T = t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        device = x.device
        denoise_to_zero = (steps % 2) == 1
        steps //= 2
        with torch.no_grad():
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
            x_next = x
            for step in range(steps):
                t_cur, t_next = timesteps[step], timesteps[step + 1]
                x_cur = x_next

                # Euler step.
                d_cur = self.model_fn(x_cur, t_cur)
                x_next = x_cur + (t_next - t_cur) * d_cur

                # Apply 2nd order correction.
                d_prime = self.model_fn(x_next, t_next)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
                # print((t_cur, t_next))
            if denoise_to_zero:
                t_cur = timesteps[-1]
                x_cur = x_next

                # Euler step.
                d_cur = self.model_fn(x_cur, t_cur)
                x_next = x_cur + (0 - t_cur) * d_cur
        return x_next
