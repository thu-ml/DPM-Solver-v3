import torch
import torch.nn.functional as F
import math
import numpy as np
import os


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


class NoiseScheduleVP:
    def __init__(
        self,
        schedule="discrete",
        betas=None,
        alphas_cumprod=None,
        continuous_beta_0=0.1,
        continuous_beta_1=20.0,
    ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).

        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ["discrete", "linear", "cosine"]:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(
                    schedule
                )
            )

        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1))
            self.log_alpha_array = log_alphas.reshape(
                (
                    1,
                    -1,
                )
            )
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = (
                math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi)
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))
            self.schedule = schedule
            if schedule == "cosine":
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.0

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == "discrete":
            return interpolate_fn(
                t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)
            ).reshape((-1))
        elif self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == "cosine":
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == "linear":
            tmp = 2.0 * (self.beta_1 - self.beta_0) * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == "discrete":
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2.0 * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1]),
            )
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            t_fn = (
                lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0))
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            t = t_fn(log_alpha)
            return t


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.0,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.

    We support four types of the diffusion model by setting `model_type`:

        1. "noise": noise prediction model. (Trained by predicting noise).

        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).

        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].

            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
                arXiv preprint arXiv:2202.00512 (2022).
            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
                arXiv preprint arXiv:2210.02303 (2022).

        4. "score": marginal score function. (Trained by denoising score matching).
            Note that the score function and the noise prediction model follows a simple relationship:
            ```
                noise(x_t, t) = -sigma_t * score(x_t, t)
            ```

    We support three types of guided sampling by DPMs by setting `guidance_type`:
        1. "uncond": unconditional sampling by DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

        2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

            The input `classifier_fn` has the following format:
            ``
                classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
            ``

            [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

        3. "classifier-free": classifier-free guidance sampling by conditional DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            ``
            And if cond == `unconditional_condition`, the model output is the unconditional DPM output.

            [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
                arXiv preprint arXiv:2207.12598 (2022).


    The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)
    or continuous-time labels (i.e. epsilon to T).

    We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:
    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return noise_pred(model, x, t_input, **model_kwargs)
    ``
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    ===============================================================

    Args:
        model: A diffusion model with the corresponding format described above.
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        model_type: A `str`. The parameterization type of the diffusion model.
                    "noise" or "x_start" or "v" or "score".
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
        guidance_type: A `str`. The type of the guidance for sampling.
                    "uncond" or "classifier" or "classifier-free".
        condition: A pytorch tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
        unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.
                    Only used for "classifier-free" guidance type.
        guidance_scale: A `float`. The scale for the guided sampling.
        classifier_fn: A classifier function. Only used for the classifier guidance.
        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the continuous time as the inputs.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == "discrete":
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, None, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return (x - expand_dims(alpha_t, dims) * output) / expand_dims(sigma_t, dims)
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return expand_dims(alpha_t, dims) * output + expand_dims(sigma_t, dims) * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return -expand_dims(sigma_t, dims) * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, dims=cond_grad.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1.0 or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


def weighted_cumsumexp_trapezoid(a, x, b, cumsum=True):
    # ∫ b*e^a dx
    # Input: a,x,b: shape (N+1,...)
    # Output: y: shape (N+1,...)
    # y_0 = 0
    # y_n = sum_{i=1}^{n} 0.5*(x_{i}-x_{i-1})*(b_{i}*e^{a_{i}}+b_{i-1}*e^{a_{i-1}}) (n from 1 to N)

    assert x.shape[0] == a.shape[0] and x.ndim == a.ndim
    if b is not None:
        assert a.shape[0] == b.shape[0] and a.ndim == b.ndim

    a_max = np.amax(a, axis=0, keepdims=True)

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    out = 0.5 * (x[1:] - x[:-1]) * (tmp[1:] + tmp[:-1])
    if not cumsum:
        return np.sum(out, axis=0) * np.exp(a_max)
    out = np.cumsum(out, axis=0)
    out *= np.exp(a_max)
    return np.concatenate([np.zeros_like(out[[0]]), out], axis=0)


def weighted_cumsumexp_trapezoid_torch(a, x, b, cumsum=True):
    assert x.shape[0] == a.shape[0] and x.ndim == a.ndim
    if b is not None:
        assert a.shape[0] == b.shape[0] and a.ndim == b.ndim

    a_max = torch.amax(a, dim=0, keepdims=True)

    if b is not None:
        tmp = b * torch.exp(a - a_max)
    else:
        tmp = torch.exp(a - a_max)

    out = 0.5 * (x[1:] - x[:-1]) * (tmp[1:] + tmp[:-1])
    if not cumsum:
        return torch.sum(out, dim=0) * torch.exp(a_max)
    out = torch.cumsum(out, dim=0)
    out *= torch.exp(a_max)
    return torch.concat([torch.zeros_like(out[[0]]), out], dim=0)


def index_list(lst, index):
    new_lst = []
    for i in index:
        new_lst.append(lst[i])
    return new_lst


class DPM_Solver_v3:
    def __init__(
        self,
        statistics_dir,
        noise_schedule,
        steps=10,
        t_start=None,
        t_end=None,
        skip_type="time_uniform",
        degenerated=False,
        device="cuda",
    ):
        self.device = device
        self.model = None
        self.noise_schedule = noise_schedule
        self.steps = steps
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"

        l = np.load(os.path.join(statistics_dir, "l.npz"))["l"]
        sb = np.load(os.path.join(statistics_dir, "sb.npz"))
        s, b = sb["s"], sb["b"]
        if degenerated:
            l = np.ones_like(l)
            s = np.zeros_like(s)
            b = np.zeros_like(b)
        self.statistics_steps = l.shape[0] - 1
        ts = noise_schedule.marginal_lambda(
            self.get_time_steps("logSNR", t_T, t_0, self.statistics_steps, "cpu")
        ).numpy()[:, None, None, None]
        self.ts = torch.from_numpy(ts).cuda()
        self.lambda_T = self.ts[0].cpu().item()
        self.lambda_0 = self.ts[-1].cpu().item()
        z = np.zeros_like(l)
        o = np.ones_like(l)
        L = weighted_cumsumexp_trapezoid(z, ts, l)
        S = weighted_cumsumexp_trapezoid(z, ts, s)

        I = weighted_cumsumexp_trapezoid(L + S, ts, o)
        B = weighted_cumsumexp_trapezoid(-S, ts, b)
        C = weighted_cumsumexp_trapezoid(L + S, ts, B)
        self.l = torch.from_numpy(l).cuda()
        self.s = torch.from_numpy(s).cuda()
        self.b = torch.from_numpy(b).cuda()
        self.L = torch.from_numpy(L).cuda()
        self.S = torch.from_numpy(S).cuda()
        self.I = torch.from_numpy(I).cuda()
        self.B = torch.from_numpy(B).cuda()
        self.C = torch.from_numpy(C).cuda()

        # precompute timesteps
        if skip_type == "logSNR" or skip_type == "time_uniform" or skip_type == "time_quadratic":
            self.timesteps = self.get_time_steps(skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            self.indexes = self.convert_to_indexes(self.timesteps)
            self.timesteps = self.convert_to_timesteps(self.indexes, device)
        elif skip_type == "edm":
            self.indexes, self.timesteps = self.get_timesteps_edm(N=steps, device=device)
            self.timesteps = self.convert_to_timesteps(self.indexes, device)
        else:
            raise ValueError(f"Unsupported timestep strategy {skip_type}")

        print("Indexes", self.indexes)
        print("Time steps", self.timesteps)
        print("LogSNR steps", self.noise_schedule.marginal_lambda(self.timesteps))

        # store high-order exponential coefficients (lazy)
        self.exp_coeffs = {}

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def convert_to_indexes(self, timesteps):
        logSNR_steps = self.noise_schedule.marginal_lambda(timesteps)
        indexes = list(
            (self.statistics_steps * (logSNR_steps - self.lambda_T) / (self.lambda_0 - self.lambda_T))
            .round()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        return indexes

    def convert_to_timesteps(self, indexes, device):
        logSNR_steps = (
            self.lambda_T + (self.lambda_0 - self.lambda_T) * torch.Tensor(indexes).to(device) / self.statistics_steps
        )
        return self.noise_schedule.inverse_lambda(logSNR_steps)

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
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type)
            )

    def get_timesteps_edm(self, N, device):
        """Constructs the noise schedule of Karras et al. (2022)."""

        rho = 7.0  # 7.0 is the value used in the paper

        sigma_min: float = np.exp(-self.lambda_0)
        sigma_max: float = np.exp(-self.lambda_T)
        ramp = np.linspace(0, 1, N + 1)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        lambdas = torch.Tensor(-np.log(sigmas)).to(device)
        timesteps = self.noise_schedule.inverse_lambda(lambdas)

        indexes = list(
            (self.statistics_steps * (lambdas - self.lambda_T) / (self.lambda_0 - self.lambda_T))
            .round()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        return indexes, timesteps

    def get_g(self, f_t, i_s, i_t):
        return torch.exp(self.S[i_s] - self.S[i_t]) * f_t - torch.exp(self.S[i_s]) * (self.B[i_t] - self.B[i_s])

    def compute_exponential_coefficients_high_order(self, i_s, i_t, order=2):
        key = (i_s, i_t, order)
        if key in self.exp_coeffs.keys():
            coeffs = self.exp_coeffs[key]
        else:
            n = order - 1
            a = self.L[i_s : i_t + 1] + self.S[i_s : i_t + 1] - self.L[i_s] - self.S[i_s]
            x = self.ts[i_s : i_t + 1]
            b = (self.ts[i_s : i_t + 1] - self.ts[i_s]) ** n / math.factorial(n)
            coeffs = weighted_cumsumexp_trapezoid_torch(a, x, b, cumsum=False)
            self.exp_coeffs[key] = coeffs
        return coeffs

    def compute_high_order_derivatives(self, n, lambda_0n, g_0n, pseudo=False):
        # return g^(1), ..., g^(n)
        if pseudo:
            D = [[] for _ in range(n + 1)]
            D[0] = g_0n
            for i in range(1, n + 1):
                for j in range(n - i + 1):
                    D[i].append((D[i - 1][j] - D[i - 1][j + 1]) / (lambda_0n[j] - lambda_0n[i + j]))

            return [D[i][0] * math.factorial(i) for i in range(1, n + 1)]
        else:
            R = []
            for i in range(1, n + 1):
                R.append(torch.pow(lambda_0n[1:] - lambda_0n[0], i))
            R = torch.stack(R).t()
            B = (torch.stack(g_0n[1:]) - g_0n[0]).reshape(n, -1)
            shape = g_0n[0].shape
            solution = torch.linalg.inv(R) @ B
            solution = solution.reshape([n] + list(shape))
            return [solution[i - 1] * math.factorial(i) for i in range(1, n + 1)]

    def multistep_predictor_update(self, x_lst, eps_lst, time_lst, index_lst, t, i_t, order=1, pseudo=False):
        # x_lst: [..., x_s]
        # eps_lst: [..., eps_s]
        # time_lst: [..., time_s]
        ns = self.noise_schedule
        n = order - 1
        indexes = [-i - 1 for i in range(n + 1)]
        x_0n = index_list(x_lst, indexes)
        eps_0n = index_list(eps_lst, indexes)
        time_0n = torch.FloatTensor(index_list(time_lst, indexes)).cuda()
        index_0n = index_list(index_lst, indexes)
        lambda_0n = ns.marginal_lambda(time_0n)
        alpha_0n = ns.marginal_alpha(time_0n)
        sigma_0n = ns.marginal_std(time_0n)

        alpha_s, alpha_t = alpha_0n[0], ns.marginal_alpha(t)
        i_s = index_0n[0]
        x_s = x_0n[0]
        g_0n = []
        for i in range(n + 1):
            f_i = (sigma_0n[i] * eps_0n[i] - self.l[index_0n[i]] * x_0n[i]) / alpha_0n[i]
            g_i = self.get_g(f_i, index_0n[0], index_0n[i])
            g_0n.append(g_i)
        g_0 = g_0n[0]
        x_t = (
            alpha_t / alpha_s * torch.exp(self.L[i_s] - self.L[i_t]) * x_s
            - alpha_t * torch.exp(-self.L[i_t] - self.S[i_s]) * (self.I[i_t] - self.I[i_s]) * g_0
            - alpha_t
            * torch.exp(-self.L[i_t])
            * (self.C[i_t] - self.C[i_s] - self.B[i_s] * (self.I[i_t] - self.I[i_s]))
        )
        if order > 1:
            g_d = self.compute_high_order_derivatives(n, lambda_0n, g_0n, pseudo=pseudo)
            for i in range(order - 1):
                x_t = (
                    x_t
                    - alpha_t
                    * torch.exp(self.L[i_s] - self.L[i_t])
                    * self.compute_exponential_coefficients_high_order(i_s, i_t, order=i + 2)
                    * g_d[i]
                )
        return x_t

    def multistep_corrector_update(self, x_lst, eps_lst, time_lst, index_lst, order=1, pseudo=False):
        # x_lst: [..., x_s, x_t]
        # eps_lst: [..., eps_s, eps_t]
        # lambda_lst: [..., lambda_s, lambda_t]
        ns = self.noise_schedule
        n = order - 1
        indexes = [-i - 1 for i in range(n + 1)]
        indexes[0] = -2
        indexes[1] = -1
        x_0n = index_list(x_lst, indexes)
        eps_0n = index_list(eps_lst, indexes)
        time_0n = torch.FloatTensor(index_list(time_lst, indexes)).cuda()
        index_0n = index_list(index_lst, indexes)
        lambda_0n = ns.marginal_lambda(time_0n)
        alpha_0n = ns.marginal_alpha(time_0n)
        sigma_0n = ns.marginal_std(time_0n)

        alpha_s, alpha_t = alpha_0n[0], alpha_0n[1]
        i_s, i_t = index_0n[0], index_0n[1]
        x_s = x_0n[0]
        g_0n = []
        for i in range(n + 1):
            f_i = (sigma_0n[i] * eps_0n[i] - self.l[index_0n[i]] * x_0n[i]) / alpha_0n[i]
            g_i = self.get_g(f_i, index_0n[0], index_0n[i])
            g_0n.append(g_i)
        g_0 = g_0n[0]
        x_t_new = (
            alpha_t / alpha_s * torch.exp(self.L[i_s] - self.L[i_t]) * x_s
            - alpha_t * torch.exp(-self.L[i_t] - self.S[i_s]) * (self.I[i_t] - self.I[i_s]) * g_0
            - alpha_t
            * torch.exp(-self.L[i_t])
            * (self.C[i_t] - self.C[i_s] - self.B[i_s] * (self.I[i_t] - self.I[i_s]))
        )
        if order > 1:
            g_d = self.compute_high_order_derivatives(n, lambda_0n, g_0n, pseudo=pseudo)
            for i in range(order - 1):
                x_t_new = (
                    x_t_new
                    - alpha_t
                    * torch.exp(self.L[i_s] - self.L[i_t])
                    * self.compute_exponential_coefficients_high_order(i_s, i_t, order=i + 2)
                    * g_d[i]
                )
        return x_t_new

    def sample(
        self,
        x,
        model_fn,
        order,
        p_pseudo,
        use_corrector,
        c_pseudo,
        lower_order_final,
        half=False,
        return_intermediate=False,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        steps = self.steps
        cached_x = []
        cached_model_output = []
        cached_time = []
        cached_index = []
        indexes, timesteps = self.indexes, self.timesteps
        step_p_order = 0

        for step in range(1, steps + 1):
            cached_x.append(x)
            cached_model_output.append(self.noise_prediction_fn(x, timesteps[step - 1]))
            cached_time.append(timesteps[step - 1])
            cached_index.append(indexes[step - 1])
            if use_corrector and (timesteps[step - 1] > 0.5 or not half):
                step_c_order = step_p_order + c_pseudo
                if step_c_order > 1:
                    x_new = self.multistep_corrector_update(
                        cached_x, cached_model_output, cached_time, cached_index, order=step_c_order, pseudo=c_pseudo
                    )
                    sigma_t = self.noise_schedule.marginal_std(cached_time[-1])
                    l_t = self.l[cached_index[-1]]
                    N_old = sigma_t * cached_model_output[-1] - l_t * cached_x[-1]
                    cached_x[-1] = x_new
                    cached_model_output[-1] = (N_old + l_t * cached_x[-1]) / sigma_t
            if step < order:
                step_p_order = step
            else:
                step_p_order = order
            if lower_order_final:
                step_p_order = min(step_p_order, steps + 1 - step)
            t = timesteps[step]
            i_t = indexes[step]

            x = self.multistep_predictor_update(
                cached_x, cached_model_output, cached_time, cached_index, t, i_t, order=step_p_order, pseudo=p_pseudo
            )

        if return_intermediate:
            return x, cached_x
        else:
            return x


#############################################################
# other utility functions
#############################################################


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


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
