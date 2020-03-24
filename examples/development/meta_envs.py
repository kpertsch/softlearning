from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT


cls_dict = {**HARD_MODE_CLS_DICT['train'], **HARD_MODE_CLS_DICT['test']}
env_list = [k for k in cls_dict.keys()]

task_args = {**HARD_MODE_ARGS_KWARGS['train'], **HARD_MODE_ARGS_KWARGS['test']}
for key in task_args:
    task_args[key]['obs_type'] = 'with_goal'

MULTITASK_ENV = MultiClassMultiTaskEnv(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=task_args,
            sample_goals=True,
            obs_type='with_goal',
            sample_all=True)

def get_metaenv(name):
    # change "env_name" format to "env-name-v1" format
    name = name.replace('_', '-') + '-v1'
    task_id = env_list.index(name)
    sampled_goal = MULTITASK_ENV._task_envs[task_id].sample_goals_(1)[0]
    MULTITASK_ENV.set_task(dict(task=task_id, goal=sampled_goal))
    return MULTITASK_ENV
