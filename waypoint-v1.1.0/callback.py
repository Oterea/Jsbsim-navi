from stable_baselines3.common.callbacks import BaseCallback
import wandb
import os
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import numpy as np

class WandbCallback(BaseCallback):
    def __init__(self, run_args, verbose: int = 0):
        super().__init__(verbose)
        self.run_args = run_args
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: # BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: # stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.artifact = wandb.Artifact(name=self.run_args.run_name, type="model")
        wandb.run.log_code(
            ".",
            name=f"{self.run_args.version}-{self.run_args.run_name}",
            include_fn=lambda path, root: path.endswith(".py") or path.startswith("readme"),
            exclude_fn=lambda path, root: os.path.relpath(path, root).startswith(
                ("runs/", "wandb/", "checkpoints/")
            ),
        )

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # add checkpoint=====================
        if self.n_calls % self.run_args.save_freq == 0:
            self.artifact.add_dir(self.run_args.model_path)
        # logging=====================
        # for key, value in self.locals.items():
        #     print(f"Key: {key}, Value: {value}")
        #
        # print("======================================")

        # actions = self.locals["actions"]

        # print(f"actions---{actions}")
        # clipped_actions = self.locals["clipped_actions"]
        # print(f"clipped_actions---{clipped_actions}")
        # print(f"newobs {self.locals['new_obs']}")
        # print(f"obs_tensor {self.locals['obs_tensor']}")
        # TODO 收集数据，然后周期log
        if self.locals["dones"][0]:
            infos = self.locals["infos"][0]
            wandb.log({"episode/return": infos["episode"]["r"], "episode/length": infos["episode"]["l"]})
            wandb.log(infos["reward_info"])
            # print(f"final new_obs{self.locals['new_obs']}")
            # print(f"final obs_tensor{self.locals['obs_tensor']}")
            print(f"{infos}")

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        wandb.run.log_artifact(self.artifact)




class SuccessRateCallback(BaseCallback):
    def __init__(self, stats_window_size: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.stats_window_size = stats_window_size
        self.success_history = []

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            infos = self.locals["infos"][0]
            self.success_history.append(int(infos["is_success"]))
            # 裁剪长度，保持 stats_window_size
            if len(self.success_history) > self.stats_window_size:
                self.success_history = self.success_history[-self.stats_window_size:]
            success_rate = np.mean(self.success_history)
            self.logger.record("success_rate", success_rate)


        return True


def get_callback_list(run_args):
    checkpoint_callback = CheckpointCallback(
        save_freq=run_args.save_freq,
        save_path=run_args.model_path,
        name_prefix=run_args.run_name,
        save_vecnormalize=run_args.vec_normalize
    )
    wandb_callback = WandbCallback(run_args)
    # success_rate_callback = SuccessRateCallback()
    # Create the callback list
    callback_list = CallbackList([checkpoint_callback, wandb_callback])

    return callback_list

