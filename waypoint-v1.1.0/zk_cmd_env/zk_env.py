import numpy as np
from typing import Optional
import gymnasium as gym
from .connection import Connection
from .start_goal import Start, Goal
from .state import State


class ZK_Env(gym.Env):
    def __init__(self, env_args, render_mode=None):
        np.set_printoptions(precision=6, suppress=True, linewidth=80)
        # 顺序不能交换
        self.args = env_args
        self._set_render_mode(render_mode)
        self.connection = Connection(self.args)
        self._set_s_a_space()
        # 是否环境第一次启动
        self.initial = True
        # 提取我方单个飞机的原始观测数据,一个字典{}， reset in reset()
        # need to reset in reset_var()
        self.is_done = False
        self.step_count = 0
        self.max_steps = 500
        self.reward_info = {}

    def _set_render_mode(self, render_mode):
        # 演示模式，无加速
        if render_mode == 1:
            self.args.play_mode = 1
        # 训练调试模式，可加速查看
        if render_mode == 2:
            self.args.render = 1

    def _set_s_a_space(self):
        """
        define observation space and action space
        """

        # 使用 Box 定义连续的动作空间
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.args.action_dim,))

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.args.obs_dim,))

    def close(self):
        self.connection.close()

    def _reset_var(self):

        # 回合是否结束重置
        self.is_done = False
        # 回合步数重置
        self.step_count = 0
        # 重置
        self.reward_info = {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._reset_var()
        # 初始化
        self.goal = Goal()
        self.start = Start()
        self.state = State()
        init_data = self.start.get_init_data(self.initial, render=self.args.render)
        self.initial = False
        self.connection.send_condition(init_data)
        # 返回 red obs {"":"", .....}
        obs = self.compute_observation(self.connection.accept_from_socket())
        return obs, {}

    def step(self, action_index):
        self.connection.send_condition(self.compute_action(action_index))
        next_obs = self.compute_observation(self.connection.accept_from_socket())
        done_info, terminated, truncated = self._is_done()
        self.is_done = len(done_info) != 0
        #reward 都需要根据is_done来判断

        # 根据上一次的obs和这一次的obs信息计算reward
        reward = self._compute_reward(done_info=done_info)
        self.step_count += 1
        info = {"done_info": done_info, "reward_info": self.reward_info}

        return next_obs, reward, terminated, truncated, info

    def _is_done(self):
        done_info = {}

        truncated, terminated = False, False

        if self.step_count > self.max_steps:
            done_info["is_max_steps_reach"] = True
            truncated |= True

        if self.state.is_out_of_bounds():
            done_info["is_out_of_bounds"] = True
            terminated |= True

        if self.state.is_reach_goal():
            done_info["is_reach_goal"] = True
            truncated |= True

        return done_info, terminated, truncated

    # ============================================================ reward related ============================================================
    def _compute_reward(self, done_info=None):
        """
        Returns:
            calculated reward
        """

        def process_reward(name, value, reward_fn, debug_fn=None):
            if debug_fn is not None:
                debug_fn(value)
            r = reward_fn(value)
            r = round(r, 6)
            self.reward_info.setdefault(name, {"r": 0, "t_r": 0})
            self.reward_info[name]["r"] = r
            self.reward_info[name]["t_r"] += r
            return r

        # step reward
        total_reward = 0
        # =======================稀疏奖励=======================
        if "is_max_steps_reach" in done_info:
            total_reward = 0
        if "is_out_of_bounds" in done_info:
            total_reward = -100

        # ====================== 稠密奖励 ======================
        # 预先定义各个 reward function
        distance_delta_fn = lambda x: max(0.05 * self.state.distance_delta, 0.0)
        distance_fn = lambda x: 200 / x

        bearing_err_fn = lambda x: np.exp(-1 * abs(x))

        bearing_err_abs_delta_fn = lambda x: np.clip(x, -2, 2)

        altitude_err_fn = lambda x: np.exp(-abs(x) / 1000)

        altitude_err_abs_delta_fn = lambda x: np.clip(x, -50, 50) * 0.01

        roll_fn = lambda x: (-1 if abs(x) > 0.5 * np.pi else np.exp(-3 * abs(x)))
        pitch_fn = lambda x: (-1 if abs(x) > 1 else np.exp(-2 * abs(x)))
        mach_err_fn = lambda x: ((max(-1, x * 10) if x < 0 else np.exp(-10 * x)))

        reward_items = [
            ("distance_delta", self.state.distance_delta, distance_delta_fn),
            ("distance", self.state.new_distance, distance_fn),

        ]

        for name, value, reward_fn in reward_items:
            total_reward += process_reward(name, value, reward_fn)

        if "is_reach_goal" in done_info:
            total_reward = 100

        return total_reward

    # ============================================================ obs related ============================================================

    def compute_observation(self, full_state):
        control_side = self.args.control_side
        obs = full_state[control_side][f'{control_side}_0']

        GEO_METER = 111319.49
        FOOT_METER = 0.3048

        cur_state = [
            obs['position/lat-geod-deg'] * GEO_METER,
            obs['position/long-gc-deg'] * GEO_METER,
            obs['position/h-sl-ft'] * FOOT_METER,

            obs['velocities/u-fps'] * FOOT_METER,
            obs['velocities/v-fps'] * FOOT_METER,
            obs['velocities/w-fps'] * FOOT_METER,

            # np.radians(np.clip(obs['aero/alpha-deg'], -30, 30)),
            # np.radians(np.clip(obs['aero/beta-deg'], -30, 30)),

            obs['attitude/roll-rad'],
            obs['attitude/pitch-rad'],
            np.radians(obs['attitude/psi-deg']),

            obs['velocities/p-rad_sec'],
            obs['velocities/q-rad_sec'],
            obs['velocities/r-rad_sec'],

        ]
        # print(f"{obs['simulation/dt']}")
        # print(f"{obs['simulation/sim-time-sec']}")
        observation = self.state.step_state(cur_state=cur_state, goal=self.goal)
        return observation

    # ============================================================ action related ============================================================
    def compute_action(self, origin_action):
        origin_action = np.array(origin_action, dtype=np.float64)

        ca = origin_action[0]
        cr = origin_action[1]
        ce = origin_action[2]
        ct = (origin_action[3] + 1) * 0.5

        control_side = self.args.control_side
        action = dict()
        action[control_side] = {
            f'{control_side}_0': {
                'mode': 0,
                # 副翼
                "fcs/aileron-cmd-norm": ca,  # 负数 左下压
                # 方向舵
                "fcs/rudder-cmd-norm": cr,  #
                # 升降舵
                "fcs/elevator-cmd-norm": ce,  # 负数 上升
                # 油门
                "fcs/throttle-cmd-norm": ct,
            }}

        # print(f"ca {ca} -- cr {cr} -- ce {ce} -- ct {ct} ")
        return action
