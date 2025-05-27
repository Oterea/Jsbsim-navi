import tyro
from stable_baselines3.common.env_checker import check_env


from zk_cmd_env.zk_env import ZK_Env
from config import Env_Args

if __name__ == "__main__":
    env = ZK_Env(env_args=tyro.cli(Env_Args))
    check_env(env)