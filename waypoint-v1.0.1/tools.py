import re

def get_vecnormalize_path(checkpoint):
    # 以 "time_" 为分割点
    parts = checkpoint.removesuffix(".zip").split("time_", 1)
    # 分成两部分
    time_prefix, step_suffix = parts[0] + "time", parts[1]  # 保留 "time"
    return f"{time_prefix}_vecnormalize_{step_suffix}.pkl"


def get_artifact(input_str):
    # 定义正则表达式来匹配两个部分
    input_str = input_str.strip()
    # 定义正则表达式，确保正确拆分
    pattern = r"wandb-artifact:///(.+?/[^/]+):(v\d+)/(.+)"
    match = re.match(pattern, input_str)

    if match:
        first_part = f"{match.group(1)}:{match.group(2)}"  # 拼接第一部分，包含 ":v0"
        second_part = match.group(3)  # 第二部分
        return first_part, second_part
    else:
        return None, None


from stable_baselines3.common.vec_env import DummyVecEnv


class MyDummyVecEnv(DummyVecEnv):
    def __enter__(self):
        return self

    def __exit__(self,exc_type, exc_value, traceback):
        self.close()


