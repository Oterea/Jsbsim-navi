import numpy as np


class Start:
    def __init__(self, random=False):

        self.position = np.zeros(3)
        self.rng = np.random.default_rng()
        self.position[:2] = np.array([0, 15000])
        self.position[2] = 3000  # 起点海拔为最高+最低 除以2， 7500
        self.yaw = 270 + 30
        self.velocity_u = 400

    def get_init_data(self, initial, render=0):
        GEO_METER = 111319.49
        METER_FOOT = 3.28084

        lat, long = self.position[:2] / GEO_METER
        altitude_ft = self.position[2] * METER_FOOT

        # 固定位置范围
        initial_data = {
            # 0.2703
            "red": {
                "red_0": {
                    "ic/h-sl-ft": altitude_ft,
                    "ic/terrain-elevation-ft": 1e-08,
                    "ic/long-gc-deg": long,
                    "ic/lat-geod-deg": lat,
                    "ic/u-fps": self.velocity_u,
                    "ic/v-fps": 0,
                    "ic/w-fps": 0,
                    "ic/p-rad_sec": 0,
                    "ic/q-rad_sec": 0,
                    "ic/r-rad_sec": 0,
                    "ic/roc-fpm": 0,
                    "ic/psi-true-deg": self.yaw,
                    "ic/phi-deg": 0,
                    "ic/theta-deg": 0
                    # , "model": 16, "mode": 1.0,"target_longdeg":0.3,"target_latdeg":0.3,"target_altitude_ft":28000.0
                }
            }
        }
        if initial:
            initial_data.update({"flag": {"init": {"save": 0, "SplitScreen": 0, "render": render}}})
        else:
            initial_data.update({"flag": {"reset": {"save": 0, "SplitScreen": 0}}})

        return initial_data


class Goal:
    def __init__(self, random=False):
        # 终点位置
        self.position = np.array([0, 0, 5000])

