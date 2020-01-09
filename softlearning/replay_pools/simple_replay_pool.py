from gym.spaces import Dict

from .flexible_replay_pool import FlexibleReplayPool, Field


class SimpleReplayPool(FlexibleReplayPool):
    def __init__(self,
                 z_dim,
                 z_type,
                 environment,
                 *args,
                 extra_fields=None,
                 **kwargs):
        extra_fields = extra_fields or {}
        observation_space = environment.observation_space
        action_space = environment.action_space
        assert isinstance(observation_space, Dict), observation_space

        self._environment = environment
        self._observation_space = observation_space
        self._action_space = action_space

        self.z_dim = z_dim
        self.z_type = z_type
        fields = {
            'observations': {
                name: Field(
                    name=name,
                    dtype=observation_space.dtype,
                    shape=(observation_space.shape[0]+self.z_dim,))
                for name, observation_space
                in observation_space.spaces.items()
            },
            'next_observations': {
                name: Field(
                    name=name,
                    dtype=observation_space.dtype,
                    shape=(observation_space.shape[0]+self.z_dim,))
                for name, observation_space
                in observation_space.spaces.items()
            },
            'actions': Field(
                name='actions',
                dtype=action_space.dtype,
                shape=environment.action_shape),
            'rewards': Field(
                name='rewards',
                dtype='float32',
                shape=(1, )),
            # terminals[i] = a terminal was received at time i
            'terminals': Field(
                name='terminals',
                dtype='bool',
                shape=(1, )),
            **extra_fields
        }

        super(SimpleReplayPool, self).__init__(
            *args, fields=fields, **kwargs)
