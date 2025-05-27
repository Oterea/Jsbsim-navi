import torch
import sys
import os
import numpy as np
def cuda():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())

def system():
    print(sys.platform)
    # 返回更详细的系统信息，如 'win32', 'linux', 'darwin'
def jax():
    import jax
    import jax.numpy as jnp

    print("JAX 设备列表:", jax.devices())

    # 让 JAX 在当前设备上执行一次计算
    x = jnp.ones((1000, 1000))  # 创建一个大数组
    y = jnp.dot(x, x)  # 计算矩阵乘法
    print("计算完成！")

def version():
    from pathlib import Path
    import sys

    # 获取主文件所在目录的上一级目录名
    grandparent_dir = Path(sys.argv[0]).resolve().parent.name

    print(grandparent_dir)
def home():
    home = rf'{os.environ.get("USERPROFILE")}\Desktop\MM\windows\ZK.exe'
    print(home)

def cwd():
    print(os.getcwd())

def concat():
    b= [4, 5, 6]
    a = torch.concat([1, 2, 3, b], 1)
    print(a)
def array():
    a= 5
    mach = np.array(a)
    print(mach.shape)

def initdata():
    # Generate a new goal
    distance = 20000
    bearing = np.pi / 2
    altitude = 7000
    position = np.array([0, 0, 0])
    position[:2] = np.cos(bearing), np.sin(bearing)
    position[:2] *= distance
    position[2] = altitude
    print(position)
    long = position[1] / 111319.49
    lat = position[0] / 111319.49
    yaw = np.degrees(bearing + np.pi)
    print(f"long{long}")
    print(f"lat{lat}")
    print(f"yaw{yaw}")


def postprocess_obs():
        # 我方的观测  14 维度 obs
        observations = _state_process()
        np.set_printoptions(precision=6, suppress=True, linewidth=80)
        # Unpack
        position = observations[:3]
        mach = observations[3:4]
        alpha_beta = observations[4:6]
        angular_rates = observations[6:9]
        phi_theta = observations[9:11]
        psi = observations[11:12]
        goal = observations[12:]

        # Transform position
        displacement = goal - position
        distance = np.sqrt(np.sum(displacement[:2] ** 2, keepdims=True))
        dz = displacement[2:3]
        altitude = position[2:3]
        abs_bearing = np.arctan2(displacement[1:2], displacement[0:1])
        rel_bearing = abs_bearing - psi
        print(f'abs_bearing{np.degrees(abs_bearing)}')
        print(f'psi{np.degrees(psi)}')
        print(f'rel_bearing{np.degrees(rel_bearing)}')
        # Angles to Sine/Cosine pairs
        cab, sab = np.cos(alpha_beta), np.sin(alpha_beta)
        cpt, spt = np.cos(phi_theta), np.sin(phi_theta)
        cr, sr = np.cos(rel_bearing), np.sin(rel_bearing)

        aux_obs = np.concatenate([distance, dz, altitude, mach, angular_rates, cab, sab, cpt, spt, cr, sr])

        low = np.array([0,    -4000, 5000, 0, -10, -10, -10 ])
        high = np.array([50000, 4000, 9000, 2,  10,  10,  10 ])
        obs_scale = 2 * (aux_obs[:7] - low) / (high - low) - 1
        obs = np.concatenate([obs_scale, aux_obs[7:]])

        return obs, aux_obs

def _state_process():

        # TODO 作为self变量，自动计算维度
        GEO_METER = 111319.49
        FOOT_METER = 0.3048

        state = [
            0.18 * GEO_METER,
            0.00001 * GEO_METER,
            7000,
            0,
            np.radians(np.clip(10, -30, 30)),
            np.radians(np.clip(10, -30, 30)),
            0,
            0,
            0,
            0,
            0,
            np.radians(315),
            0 * GEO_METER,
            0 * GEO_METER,
            7000
        ]
        return  np.array(state)

def qipian():
    position = np.array([4, 8, 10])
    lat, long = position[:2] / 2
    print(lat, long)

def random_num():
    a = np.random.default_rng().random()
    b= np.random.default_rng().uniform(5000,10000)
    print(b)

def distance():
    a = np.array([30000, 2000])
    b = np.linalg.norm(a)
    print(b)

def random():
    rng = np.random.default_rng()
    a= (rng.random() * 2 - 1) * 90
    print(a)


def goal():
    from zk_cmd_env.start_goal import Goal
    goal = Goal()
    for i in range(10):
        goal.update_goal()
        print(goal.position)

if __name__ == '__main__':
    goal()




