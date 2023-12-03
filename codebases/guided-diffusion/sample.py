import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.multiprocessing as mp

from runners.diffusion import Diffusion
from evaluate.fid_score import calculate_fid_given_paths

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--exp", type=str, default="exp", help="Experiment name")
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="dpmsolver++",
        help="Sampling approach ('generalized'(DDIM) or 'ddpm_noisy'(DDPM) or 'dpmsolver' or 'dpmsolver++' or 'unipc' or 'dpmsolver_v3')",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="time_uniform",
        help="Timestep schedule for sampling ('uniform' or 'quadratic' for DDIM/DDPM; 'logSNR' or 'time_uniform' or 'time_quadratic' for DPM-Solver)",
    )
    parser.add_argument("--timesteps", type=int, default=10, help="Number of steps for sampling")
    parser.add_argument("--order", type=int, default=2, help="Order of dpm-solver")
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--fixed_class", type=int, default=None, help="Fixed class label for conditional sampling")
    parser.add_argument("--scale", type=float, default=0.0, help="Guidance scale")
    parser.add_argument("--denoise", action="store_true", default=False, help="Denoise at the last step")
    parser.add_argument(
        "--lower_order_final", action="store_true", default=False, help="Use first-order at the last step"
    )
    parser.add_argument("--thresholding", action="store_true", default=False, help="Use dynamic thresholding")
    parser.add_argument("--statistics_dir", type=str, default=None, help="Statistics path for DPM-Solver-v3.")

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.exp = os.path.join("samples", new_config.model.ckpt_dir.split("/")[-1].split(".")[-2], args.exp)
    os.makedirs(os.path.join(args.exp, args.image_folder), exist_ok=True)
    args.image_folder = os.path.join(args.exp, args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Exp instance id = {}".format(os.getpid()))

    world_size = torch.cuda.device_count()
    if not os.path.exists(os.path.join(args.exp, "fid.npy")):
        mp.spawn(sample, args=(world_size, args, config), nprocs=world_size, join=True)
        print("Begin to compute FID...")
        fid = calculate_fid_given_paths(
            (config.sampling.fid_stats_dir, args.image_folder),
            batch_size=config.sampling.fid_batch_size,
            device="cuda",
            dims=2048,
            num_workers=8,
        )
        print("FID: {}".format(fid))
        np.save(os.path.join(args.exp, "fid"), fid)
    else:
        fid = np.load(os.path.join(args.exp, "fid.npy"))
        print(args.exp, fid)
        print(args.exp, fid, file=open("output.txt", mode="a"))


def sample(rank, world_size, args, config):
    # set random seed
    torch.cuda.set_device(rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    try:
        runner = Diffusion(args, config, rank=rank)
        runner.sample()

    except Exception:
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    sys.exit(main())
