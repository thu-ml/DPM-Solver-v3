import torch
import math


class NoiseScheduleEDM:
    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return torch.zeros_like(t).to(torch.float64)

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.ones_like(t).to(torch.float64)

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return t.to(torch.float64)

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """

        return -torch.log(t).to(torch.float64)

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        return torch.exp(-lamb).to(torch.float64)


def model_wrapper(model, noise_schedule, class_labels=None):
    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = t_continuous
        output = model(x, t_input, cond)
        alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
        return (x - alpha_t[:, None, None, None] * output) / sigma_t[:, None, None, None]

    def model_fn(x, t_continuous):
        return noise_pred_fn(x, t_continuous, class_labels).to(torch.float64)

    return model_fn


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
