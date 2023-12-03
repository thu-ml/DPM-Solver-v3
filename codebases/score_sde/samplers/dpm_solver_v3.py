import torch
import torch.nn.functional as F
import math
import numpy as np
import os


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
        model_fn,
        noise_schedule,
        steps=10,
        t_start=None,
        t_end=None,
        skip_type="logSNR",
        degenerated=False,
        device="cuda",
    ):
        # precompute
        self.device = device
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
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

    @torch.no_grad()
    def sample(self, x, order, p_pseudo, use_corrector, c_pseudo, lower_order_final, return_intermediate=False):
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
            if use_corrector:
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
