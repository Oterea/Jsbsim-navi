import tyro
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from tools import MyDummyVecEnv, get_vecnormalize_path, get_artifact
import wandb
from zk_cmd_env.zk_env import ZK_Env
from config import Env_Args, Run_Args
from stable_baselines3.common.evaluation import evaluate_policy
import logging
def callback(locals_, global_):
    # print("============")
    # for key, value in locals_.items():
    #     print(f"Key: {key}, Value: {value}")

    episode_count_targets = locals_['episode_count_targets'][0]
    success_count = 0
    if locals_["dones"][0]:
        infos = locals_["infos"][0]
        del infos["terminal_observation"]
        if "is_reach_waypoint" in infos["done_info"]:
            success_count += 1
        print(f"{infos}")
        if locals_["episode_counts"][0]+1 == episode_count_targets:
            print(f"Success Rate: {success_count/episode_count_targets*100:.2%}")


if __name__ == "__main__":

    run_args, env_args = tyro.cli(Run_Args), tyro.cli(Env_Args)
    run_args.num_envs = 1
    if run_args.sbx:
        from sbx import PPO
        logging.info(f"Evaluating with SBX")
    else:
        from stable_baselines3 import PPO
        logging.info(f"Evaluating with SB3")
    with (make_vec_env(
            env_id=ZK_Env,
            n_envs=run_args.num_envs,
            env_kwargs={"env_args": env_args, "render_mode":2},
            vec_env_cls=MyDummyVecEnv
    ) as vec_env):
        # 下载模型或者加载本地模型文件
        artifact_model_path = """
04_29_23_01_time_6144000_steps.zip
        """
        if artifact_model_path.startswith("wandb-artifact"):
            artifact_name, file_name = get_artifact(artifact_model_path)
            print(artifact_name, file_name)
            model_path = (wandb.Api(api_key=run_args.wandb_apikey)
                            .artifact(f'{artifact_name}', type='model')
                            .get_entry(f"{file_name}")
                            .download(f"wandb/artifacts"))
        else:
            model_path = f"{run_args.model_path}/{artifact_model_path.strip()}"

        print(model_path)

        model = PPO.load(f"{model_path}", env=vec_env)

        # Eval trained agent
        mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=100, callback=callback)

    # Enjoy trained agent
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    #
        # for i in range(20480):
        #     action, _states = model.predict(obs)
    #     # print(action)
    #     obs, rewards, dones, info = vec_env.step(action)
            # print(obs)

