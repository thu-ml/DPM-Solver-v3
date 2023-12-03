import tensorflow_gan as tfgan
import tensorflow as tf
import numpy as np
import os

from evaluation import *
import gc
from tqdm import tqdm


inception_model = get_inception_model(inceptionv3=False)
BATCH_SIZE = 1000


def load_cifar10_stats():
    """Load the pre-computed dataset statistics."""
    filename = "assets/stats/cifar10_stats.npz"

    with tf.io.gfile.GFile(filename, "rb") as fin:
        stats = np.load(fin)
        return stats


def compute_fid(path):
    images = []
    for file in os.listdir(path):
        if file.endswith(".npz"):
            with tf.io.gfile.GFile(os.path.join(path, file), "rb") as fin:
                sample = np.load(fin)
        images.append(sample["samples"])
    samples = np.concatenate(images, axis=0)
    all_pools = []
    N = samples.shape[0]
    assert N >= 50000, "At least 50k samples are required to compute FID."
    for i in tqdm(range(N // BATCH_SIZE)):
        gc.collect()
        latents = run_inception_distributed(
            samples[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, ...], inception_model, inceptionv3=False
        )
        gc.collect()
        all_pools.append(latents["pool_3"])
    all_pools = np.concatenate(all_pools, axis=0)[:50000, ...]
    data_stats = load_cifar10_stats()
    data_pools = data_stats["pool_3"]

    fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, all_pools)
    return fid

for name in ["DPM-Solver++", "UniPC_bh1", "UniPC_bh2", "DPM-Solver-v3"]:
    fids = []
    for step in [5, 6, 8, 10, 12, 15, 20, 25]:
        path = f"samples/checkpoint_8/{name}_{step}"
        fid = compute_fid(path)
        fids.append(float(fid))
    print(name, fids, file=open("output.txt", "a"))
