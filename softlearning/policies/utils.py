from collections import OrderedDict
from copy import deepcopy
from tensorflow import TensorShape

from softlearning.preprocessors.utils import get_preprocessor_from_params
from softlearning.environments.adapters.gym_adapter import GymAdapter
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv


def get_gaussian_policy(*args, **kwargs):
    from .gaussian_policy import FeedforwardGaussianPolicy

    policy = FeedforwardGaussianPolicy(*args, **kwargs)

    return policy


def get_uniform_policy(*args, **kwargs):
    from .uniform_policy import ContinuousUniformPolicy

    policy = ContinuousUniformPolicy(*args, **kwargs)

    return policy


POLICY_FUNCTIONS = {
    'GaussianPolicy': get_gaussian_policy,
    'UniformPolicy': get_uniform_policy,
}


def get_policy(policy_type, *args, **kwargs):
    return POLICY_FUNCTIONS[policy_type](*args, **kwargs)


def get_policy_from_params(policy_params, env, *args, **kwargs):
    policy_type = policy_params['type']
    policy_kwargs = deepcopy(policy_params.get('kwargs', {}))

    observation_preprocessors_params = policy_kwargs.pop(
        'observation_preprocessors_params', {})
    observation_keys = policy_kwargs.pop(
        'observation_keys', None) or env.observation_keys

    observation_shapes = OrderedDict((
        (key, value) for key, value in env.observation_shape.items()
        if key in observation_keys
    ))

    # HACK: set observation shape to 6
    if env.__class__ == GymAdapter and env.unwrapped.__class__ == MultiClassMultiTaskEnv:
        env_class_init = env.unwrapped.active_env.__class__.__init__
        obs_type_param_ix = env_class_init.__code__.co_varnames.index('obs_type') - 1
        default_obs_type = env_class_init.__defaults__[obs_type_param_ix]
        if default_obs_type == 'plain':
            observation_len = 6
        elif default_obs_type == 'with_goal':
            observation_len = 9
        elif default_obs_type == 'with_goal_init_obs':
            observation_len = 15
        observation_shapes = OrderedDict({'observations': TensorShape([observation_len])})

    observation_preprocessors = OrderedDict()
    for name, observation_shape in observation_shapes.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            observation_preprocessors[name] = None
            continue

        observation_preprocessors[name] = get_preprocessor_from_params(
            env, preprocessor_params)

    action_range = (env.action_space.low, env.action_space.high)

    policy = POLICY_FUNCTIONS[policy_type](
        input_shapes=observation_shapes,
        output_shape=env.action_shape,
        action_range=action_range,
        observation_keys=observation_keys,
        *args,
        preprocessors=observation_preprocessors,
        **policy_kwargs,
        **kwargs)

    return policy


def get_policy_from_variant(variant, *args, **kwargs):
    policy_params = variant['policy_params']
    return get_policy_from_params(policy_params, *args, **kwargs)
