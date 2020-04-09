from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT


cls_dict = {**HARD_MODE_CLS_DICT['train'], **HARD_MODE_CLS_DICT['test']}
env_list = [k for k in cls_dict.keys()]

env_dict = {
    'assembly_peg': 'assembly-v1',
    'bin_picking': 'bin-picking-v1',
    'box_close': 'box-close-v1',
    'button_press': 'button-press-v1',
    'button_press_topdown': 'button-press_topdown-v1',
    'dial_turn': 'dial-turn-v1',
    'door': 'door-open-v1',
    'door_close': 'door-close-v1',
    'drawer_close': 'drawer-close-v1',
    'drawer_open': 'drawer-open-v1',
    'hammer': 'hammer-v1',
    'hand_insert': 'hand-insert-v1',
    'lever_pull': 'lever-pull-v1',
    'peg_insertion_side': 'peg-insert-side-v1',
    'reach_push_pick_place': 'pick-place-v1',
    'shelf_place': 'shelf-place-v1',
    'stick_pull': 'stick-pull-v1',
    'stick_push': 'stick-push-v1',
    'sweep': 'sweep-v1',
    'sweep_into_goal': 'sweep-into-v1',
    'window_close': 'window-close-v1',
    'window_open': 'window-open-v1',
    'coffee_button': 'coffee-button-v1',
    'coffee_push': 'coffee-push-v1',
    'coffee_pull': 'coffee-pull-v1',
    'faucet_open': 'faucet-open-v1',
    'faucet_close': 'faucet-close-v1',
    'peg_unplug_side': 'peg-unplug-side-v1',
    'soccer': 'soccer-v1',
    'basketball': 'basket-ball-v1',
    'reach_push_pick_place_wall': 'pick-place-wall-v1',
    'push_back': 'push-back-v1',
    'pick_out_of_hole': 'pick-out-of-hole-v1',
    'shelf_remove': 'shelf-remove-v1', # not in metaworld
    'disassemble_peg': 'diassemble-v1',
    'door_lock': 'door-lock-v1',
    'door_unlock': 'door-unlock-v1',
    'sweep_tool': 'sweep-tool-v1', # not in metaworld
    'button_press_wall': 'button-press-wall-v1',
    'button_press_topdown_wall': 'button-press-topdown-wall-v1',
    'handle_press_side': 'handle-press-side-v1',
    'handle_pull_side': 'handle-pull-side-v1',
    'plate_slide': 'plate-slide-v1',
    'plate_slide_back': 'plate-slide-back-v1',
    'plate_slide_side': 'plate-slide-side-v1',
    'plate_slide_back_side': 'plate-slide-back-side-v1',
    'reach_push_pick_place_reach': 'reach-v1',
    'reach_push_pick_place_push' : 'push-v1',
    'reach_push_pick_place_wall_reach' : 'reach-wall-v1',
    'reach_push_pick_place_wall_push' : 'push-wall-v1',
    'handle_press' : 'handle-press-v1',
    'handle_pull' : 'handle-pull-v1'
}

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
    task_id = env_list.index(env_dict[name])
    sampled_goal = MULTITASK_ENV._task_envs[task_id].sample_goals_(1)[0]
    MULTITASK_ENV.set_task(dict(task=task_id, goal=sampled_goal))
    return MULTITASK_ENV
