import numpy as np


class State:
    def __init__(self):
        self.old_distance = np.inf
        self.new_distance = np.inf
        self.distance_delta = 0
        self.goal_reach_threshold = 200

        # bounds
        self.altitude = np.inf

    def reset_state(self):
        pass

    def step_state(self, cur_state, goal):
        cur_state = np.array(cur_state)

        # target_deltas for observation
        target_deltas = goal.position - cur_state[:3]
        # distance for reward
        self.old_distance = self.new_distance
        self.new_distance = np.linalg.norm(target_deltas[:3])
        self.distance_delta = self.old_distance - self.new_distance

        # bounds
        self.altitude = cur_state[2]

        scale_state = np.concatenate([cur_state[:3] * 1e-4, cur_state[3:6] * 1e-2, cur_state[6:]])

        return np.concatenate([scale_state, target_deltas * 1e-2])

    def is_reach_goal(self):
        if self.new_distance < self.goal_reach_threshold:
            print('Goal reached!')
            return True

        return False

    def is_out_of_bounds(self):
        if not (1000 < self.altitude < 7000):
            return True
        return False
