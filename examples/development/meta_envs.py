from metaworld.envs.mujoco.sawyer_xyz import \
    SawyerNutAssemblyEnv, SawyerBinPickingEnv, SawyerBoxCloseEnv, SawyerButtonPressEnv,\
    SawyerButtonPressTopdownEnv, SawyerDialTurnEnv, SawyerDoorEnv, SawyerDoorCloseEnv, SawyerDrawerCloseEnv,\
    SawyerDrawerOpenEnv, SawyerHammerEnv, SawyerHandInsertEnv, SawyerLeverPullEnv, SawyerPegInsertionSideEnv,\
    SawyerReachPushPickPlaceEnv, SawyerShelfPlaceEnv, SawyerStickPullEnv, SawyerStickPushEnv, SawyerSweepEnv,\
    SawyerSweepIntoGoalEnv, SawyerWindowCloseEnv, SawyerWindowOpenEnv, SawyerCoffeeButtonEnv, SawyerCoffeePushEnv,\
    SawyerCoffeePullEnv, SawyerFaucetOpenEnv, SawyerFaucetCloseEnv, SawyerPegUnplugSideEnv, SawyerSoccerEnv,\
    SawyerBasketballEnv, SawyerReachPushPickPlaceWallEnv, SawyerPushBackEnv, SawyerPickOutOfHoleEnv,\
    SawyerShelfRemoveEnv, SawyerNutDisassembleEnv, SawyerDoorLockEnv, SawyerDoorUnlockEnv, SawyerSweepToolEnv,\
    SawyerButtonPressWallEnv, SawyerButtonPressTopdownWallEnv, SawyerHandlePressSideEnv, SawyerHandlePullSideEnv,\
    SawyerPlateSlideEnv, SawyerPlateSlideBackEnv, SawyerPlateSlideSideEnv, SawyerPlateSlideBackSideEnv, \
    SawyerReachPushPickPlaceSawyerReachEnv, SawyerReachPushPickPlaceSawyerPushEnv, \
    SawyerReachPushPickPlaceWallSawyerPushEnv, SawyerReachPushPickPlaceWallSawyerReachEnv


META_ENVS = {
    'assembly_peg': SawyerNutAssemblyEnv,
    'bin_picking': SawyerBinPickingEnv,
    'box_close': SawyerBoxCloseEnv,
    'button_press': SawyerButtonPressEnv,
    'button_press_topdown': SawyerButtonPressTopdownEnv,
    'dial_turn': SawyerDialTurnEnv,
    'door': SawyerDoorEnv,
    'door_close': SawyerDoorCloseEnv,
    'drawer_close': SawyerDrawerCloseEnv,
    'drawer_open': SawyerDrawerOpenEnv,
    'hammer': SawyerHammerEnv,
    'hand_insert': SawyerHandInsertEnv,
    'lever_pull': SawyerLeverPullEnv,
    'peg_insertion_side': SawyerPegInsertionSideEnv,
    'reach_push_pick_place': SawyerReachPushPickPlaceEnv,
    'shelf_place': SawyerShelfPlaceEnv,
    'stick_pull': SawyerStickPullEnv,
    'stick_push': SawyerStickPushEnv,
    'sweep': SawyerSweepEnv,
    'sweep_into_goal': SawyerSweepIntoGoalEnv,
    'window_close': SawyerWindowCloseEnv,
    'window_open': SawyerWindowOpenEnv,
    'coffee_button': SawyerCoffeeButtonEnv,
    'coffee_push': SawyerCoffeePushEnv,
    'coffee_pull': SawyerCoffeePullEnv,
    'faucet_open': SawyerFaucetOpenEnv,
    'faucet_close': SawyerFaucetCloseEnv,
    'peg_unplug_side': SawyerPegUnplugSideEnv,
    'soccer': SawyerSoccerEnv,
    'basketball': SawyerBasketballEnv,
    'reach_push_pick_place_wall': SawyerReachPushPickPlaceWallEnv,
    'push_back': SawyerPushBackEnv,
    'pick_out_of_hole': SawyerPickOutOfHoleEnv,
    'shelf_remove': SawyerShelfRemoveEnv,
    'disassemble_peg': SawyerNutDisassembleEnv,
    'door_lock': SawyerDoorLockEnv,
    'door_unlock': SawyerDoorUnlockEnv,
    'sweep_tool': SawyerSweepToolEnv,
    'button_press_wall': SawyerButtonPressWallEnv,
    'button_press_topdown_wall': SawyerButtonPressTopdownWallEnv,
    'handle_press_side': SawyerHandlePressSideEnv,
    'handle_pull_side': SawyerHandlePullSideEnv,
    'plate_slide': SawyerPlateSlideEnv,
    'plate_slide_back': SawyerPlateSlideBackEnv,
    'plate_slide_side': SawyerPlateSlideSideEnv,
    'plate_slide_back_side': SawyerPlateSlideBackSideEnv,
    'reach_push_pick_place_reach': SawyerReachPushPickPlaceSawyerReachEnv,
    'reach_push_pick_place_push' : SawyerReachPushPickPlaceSawyerPushEnv,
    'reach_push_pick_place_wall_reach' : SawyerReachPushPickPlaceWallSawyerReachEnv,
    'reach_push_pick_place_wall_push' : SawyerReachPushPickPlaceWallSawyerPushEnv,
}


def get_metaenv(name):
    return META_ENVS[name]()
