# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import torch
import io
import time
import numpy as np

# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
from torchvision.utils import make_grid, save_image
from utils import restore_checkpoint
from models.utils import get_noise_fn
from samplers.dpm_solver_v3 import DPM_Solver_v3
from samplers.utils import NoiseScheduleVP
import functools

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("ckp_path", None, "Checkpoint path.")
flags.DEFINE_string("statistics_dir", None, "Statistics path for DPM-Solver-v3.")
flags.DEFINE_string("eval_folder", "samples", "The folder name for storing evaluation results")
flags.DEFINE_string("sample_folder", "sample", "The folder name for storing samples")
flags.mark_flags_as_required(["ckp_path", "config"])


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def main(argv):
    sample(FLAGS.config, FLAGS.ckp_path, FLAGS.statistics_dir, FLAGS.eval_folder, FLAGS.sample_folder)


def sample(config, ckp_path, statistics_dir, eval_folder="samples", sample_dir="sample"):
    # Fix the seed for z = sde.prior_sampling(shape).to(device) in deterministic sampling
    torch.manual_seed(config.seed)
    eval_dir = os.path.join(eval_folder, ckp_path.split("/")[-1].split(".")[-2])
    os.makedirs(eval_dir, exist_ok=True)

    # Create data normalizer and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unsupported.")

    sampling_shape = (config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)

    state = restore_checkpoint(ckp_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    if config.sampling.method == "dpm_solver_v3":
        assert statistics_dir is not None, "No appropriate statistics found."
        print("Use statistics", statistics_dir)

        noise_pred_fn = get_noise_fn(sde, score_model, train=False, continuous=True)
        ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)
        dpm_solver_v3 = DPM_Solver_v3(
            statistics_dir,
            noise_pred_fn,
            ns,
            steps=config.sampling.steps,
            t_start=sde.T,
            t_end=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            degenerated=config.sampling.degenerated,
            device=config.device,
        )

        def dpm_solver_v3_sampler():
            with torch.no_grad():
                x = sde.prior_sampling(sampling_shape).to(config.device)
                x = dpm_solver_v3.sample(
                    x,
                    order=config.sampling.order,
                    p_pseudo=config.sampling.predictor_pseudo,
                    use_corrector=config.sampling.use_corrector,
                    c_pseudo=config.sampling.corrector_pseudo,
                    lower_order_final=config.sampling.lower_order_final,
                )
            return inverse_scaler(x), config.sampling.steps

        sampling_fn = dpm_solver_v3_sampler
    else:
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler)
        sampling_fn = functools.partial(sampling_fn, score_model)

    this_sample_dir = os.path.join(eval_dir, sample_dir)
    os.makedirs(this_sample_dir, exist_ok=True)
    logging.info(this_sample_dir)
    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
    for r in range(num_sampling_rounds):
        samples_raw, n = sampling_fn()
        logging.info("sampling -- round: %d (NFE %d)" % (r, n))
        samples = np.clip(samples_raw.permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        np.savez_compressed(os.path.join(this_sample_dir, f"samples_{r}.npz"), samples=samples)

        if r == 0:
            nrow = int(np.sqrt(samples_raw.shape[0]))
            image_grid = make_grid(samples_raw, nrow, padding=2)
            with open(os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                save_image(image_grid, fout)


if __name__ == "__main__":
    app.run(main)
