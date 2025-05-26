from typing import Callable

import tyro
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from tools import MyDummyVecEnv, get_vecnormalize_path
from zk_cmd_env.zk_env import ZK_Env
from config import Env_Args, Run_Args
from callback import get_callback_list
import wandb
import logging


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    continue_training = False
    run_args, env_args = tyro.cli(Run_Args), tyro.cli(Env_Args)
    run_args.num_envs = 1
    if run_args.sbx:
        from sbx import PPO
        logging.info(f"Training with SBX")
    else:
        from stable_baselines3 import PPO
        logging.info(f"Training with SB3")
    x="Training with obs_scale" if env_args.obs_scale else "Training with no obs_scale"
    logging.info(x)
    y= "Training with vec_normalize" if run_args.vec_normalize else "Training with no vec_normalize"
    logging.info(y)


    with make_vec_env(
            env_id=ZK_Env,
            n_envs=run_args.num_envs,
            env_kwargs={"env_args": env_args, "render_mode":0},
            vec_env_cls=MyDummyVecEnv,
    ) as vec_env:

        wandb.login(key=run_args.wandb_apikey)
        with wandb.init(
            entity=run_args.wandb_entity,
            project=run_args.wandb_project_name,
            name=run_args.run_name,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            notes=run_args.wandb_notes,
            # mode="disabled"
            # save_code=True,
            # settings=wandb.Settings(code_dir=".")
        ) as run:

            # get the callback list
            callback = get_callback_list(run_args)
            model_path = run_args.model_path
            save_freq = run_args.save_freq


            def linear_schedule(initial_value: float) -> Callable[[float], float]:
                """
                Linear learning rate schedule.

                :param initial_value: Initial learning rate.
                :return: schedule that computes
                  current learning rate depending on remaining progress
                """

                def func(progress_remaining: float) -> float:
                    """
                    Progress will decrease from 1 (beginning) to 0.

                    :param progress_remaining:
                    :return: current learning rate
                    """
                    return progress_remaining * initial_value

                return func
            if continue_training:
                logging.info(f"Continue training")
                checkpoint = "04_29_15_42_time_2048000_steps.zip" # 不用zip后缀
                if run_args.vec_normalize:
                    vecnormalize_path = get_vecnormalize_path(checkpoint)
                    vec_env = VecNormalize.load(f"{model_path}/{vecnormalize_path}", vec_env)
                model = PPO.load(f"{model_path}/{checkpoint}", env=vec_env)
                model.learn(total_timesteps=save_freq * 1, reset_num_timesteps=False, callback=callback)
            else:
                logging.info(f"Training from scratch")
                vec_env = VecNormalize(vec_env) if run_args.vec_normalize else vec_env
                # vec_env = VecFrameStack(vec_env, 4)
                # policy_kwargs = dict(net_arch=[128, 128])
                model = PPO("MlpPolicy", vec_env, learning_rate=linear_schedule(0.0003), tensorboard_log="runs", verbose=1) # verbose print loss etc. in terminal. 指定tensorboard_log wandb才会同步损失等信息
                model.learn(total_timesteps=save_freq * 3, callback=callback)

