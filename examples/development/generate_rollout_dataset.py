import argparse
from distutils.util import strtobool
import json
import os
from pathlib import Path
import pickle
import h5py
import tqdm
import cv2

import tensorflow as tf
import pandas as pd
import numpy as np

from softlearning.environments.utils import (
    get_environment_from_params, get_environment)
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollout
from softlearning.utils.video import save_video

from .meta_envs import get_metaenv
from softlearning.environments.adapters.gym_adapter import GymAdapter


DEFAULT_RENDER_KWARGS = {
    'mode': 'rgb_array',
    'width': 128,
    'height': 128,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-kwargs', '-r',
                        type=json.loads,
                        default='{}',
                        help="Kwargs for rollouts renderer.")
    parser.add_argument('--data-save-path',
                        type=Path,
                        default=None)
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=False,
                        default=False,
                        help="Evaluate policy deterministically.")
    parser.add_argument('--metaenv_name',
                        type=str,
                        help='Name of meta-environment.')

    args = parser.parse_args()

    return args


def load_checkpoint(checkpoint_path, session=None):
    session = session or tf.keras.backend.get_session()
    checkpoint_path = checkpoint_path.rstrip('/')
    trial_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(trial_path, 'params.pkl')
    with open(variant_path, 'rb') as f:
        variant = pickle.load(f)

    metadata_path = os.path.join(checkpoint_path, ".tune_metadata")
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = None

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    progress_path = os.path.join(trial_path, 'progress.csv')
    progress = pd.read_csv(progress_path)

    return picklable, variant, progress, metadata


def load_policy_and_environment(picklable, variant, metaenv_name):
    environment = GymAdapter(domain=None, task=None, env=get_metaenv(metaenv_name))

    policy = get_policy_from_variant(variant, environment)
    policy.set_weights(picklable['policy_weights'])

    return policy, environment


def resize_video(images, dim=64):
    ret = np.zeros((images.shape[0], dim, dim, 3))

    for i in range(images.shape[0]):
        ret[i] = cv2.resize(images[i], dsize=(dim, dim), interpolation=cv2.INTER_CUBIC)

    return ret.astype(np.uint8)


def simulate_policy(checkpoint_path,
                    deterministic,
                    num_rollouts,
                    max_path_length,
                    render_kwargs,
                    data_save_path=None,
                    metaenv_name='',
                    image_resize_dim=64):
    checkpoint_path = checkpoint_path.rstrip('/')
    picklable, variant, progress, metadata = load_checkpoint(checkpoint_path)
    policy, environment = load_policy_and_environment(picklable, variant, metaenv_name)
    print("Loading done")
    render_kwargs = {**DEFAULT_RENDER_KWARGS, **render_kwargs}
    render_kwargs['mode'] = 'rgb_array'

    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    with policy.set_deterministic(deterministic):
        for i in tqdm.tqdm(range(num_rollouts)):
            path = rollout(environment,
                             policy,
                             path_length=max_path_length,
                             render_kwargs=render_kwargs)

            # save rollout to file
            f = h5py.File(os.path.join(data_save_path, "rollout_{}.h5".format(i)), "w")
            f.create_dataset("traj_per_file", data=np.array([1]))

            traj_data = f.create_group("traj0")       # store trajectory info in traj0 group
            traj_data.create_dataset("states", data=path['observations']['observations'])
            traj_data.create_dataset("images", data=resize_video(path['images'], dim=image_resize_dim))
            traj_data.create_dataset("actions", data=path['actions'])

            if np.sum(path['terminals']) == 0:
                # episode didn't end
                path['terminals'][-1, 0] = True

            is_terminal_idxs = np.nonzero(path['terminals'][:, 0])[0]    # build pad-mask that indicates how long sequence is
            pad_mask = np.zeros((path['terminals'].shape[0],))
            pad_mask[:is_terminal_idxs[0]] = 1.
            traj_data.create_dataset("pad_mask", data=pad_mask)

            f.close()

    print("Dataset generation done")


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    args = parse_args()
    print("Let's start")
    simulate_policy(**vars(args))
