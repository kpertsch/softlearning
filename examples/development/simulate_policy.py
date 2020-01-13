import argparse
from distutils.util import strtobool
import json
import os
from pathlib import Path
import pickle

import tensorflow as tf
import pandas as pd
import numpy as np

from softlearning.environments.utils import (
    get_environment_from_params, get_environment)
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from softlearning.utils.video import save_video

from examples.development.meta_envs import get_metaenv
from softlearning.environments.adapters.gym_adapter import GymAdapter


DEFAULT_RENDER_KWARGS = {
    'mode': 'rgb_array',
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
    parser.add_argument('--video-save-path',
                        type=Path,
                        default=None)
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
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
    # environment_params = (
    #     variant['environment_params']['training']
    #     if 'evaluation' in variant['environment_params']
    #     else variant['environment_params']['training'])

    environment = GymAdapter(domain=None, task=None, env=get_metaenv(metaenv_name))
    #environment = get_environment_from_params(environment_params)

    policy = get_policy_from_variant(variant, environment)
    policy.set_weights(picklable['policy_weights'])

    return policy, environment


def get_raw_observations(paths, z_dim):
    observations = []
    observatoin_dim = paths[0]['observations']['observations'].shape[1]
    for i in range(len((paths))):
        observations.append(paths[i]['observations']['observations'][:,:observatoin_dim-z_dim])

    return  observations


def find_entropy(observations):
    eef_observations = observations[:,:3]
    H, edges = np.histogramdd(eef_observations, bins = 50)
    H = H/np.sum(H)
    eps = 1e-7
    entropy = -1*np.sum(H*np.log(H+eps))
    return entropy

def calculate_diversity(raw_observations):
    observations = np.array(raw_observations)
    observations = observations.reshape(-1, observations.shape[-1])
    diversity = find_entropy(observations)
    return diversity

def simulate_policy(checkpoint_path,
                    deterministic,
                    num_rollouts,
                    max_path_length,
                    render_kwargs,
                    video_save_path=None,
                    evaluation_environment_params=None,
                    metaenv_name=''):
    checkpoint_path = checkpoint_path.rstrip('/')
    picklable, variant, progress, metadata = load_checkpoint(checkpoint_path)

    z_dim = variant['policy_params']['z_dim']
    z_type = variant['policy_params']['z_type']

    policy, environment = load_policy_and_environment(picklable, variant, metaenv_name)
    print("Loading done")
    render_kwargs = {**DEFAULT_RENDER_KWARGS, **render_kwargs}
    render_kwargs['mode'] = 'rgb_array'

    with policy.set_deterministic(deterministic):
        paths = rollouts(num_rollouts,
                         environment,
                         policy,
                         z_dim = z_dim,
                         z_type = z_type,
                         path_length=max_path_length,
                         render_kwargs=render_kwargs)
    print("Rollout done")

    raw_observations = get_raw_observations(paths, z_dim)
    diversity = calculate_diversity(raw_observations)
    print('diversity measure: {}'.format(diversity))

    if video_save_path and render_kwargs.get('mode') == 'rgb_array':
        fps = 1 // getattr(environment, 'dt', 1/30)
        if video_save_path is None:
            video_save_path = os.path.expanduser('/tmp/simulate_policy/')
        
        # save individual rollout videos
        #for i, path in enumerate(paths):
        #    video_save_path_full = os.path.join(video_save_path, f'episode_{i}.mp4')
        #    save_video(path['images'], video_save_path_full, fps=fps)
        
        # make combined videos for easier comparison
        path_images = [path['images'] for path in paths]
        mean_img = np.asarray(np.sum(np.array(path_images), axis=0) / len(path_images), dtype=path_images[0].dtype)
        comb_videos = np.concatenate(path_images + [mean_img], axis=2)
        save_video(comb_videos, os.path.join(video_save_path, 'combined.mp4'), fps=fps)
        

    return paths


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    args = parse_args()
    print("Let's start")
    simulate_policy(**vars(args))
