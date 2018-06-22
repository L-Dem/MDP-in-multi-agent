import numpy as np
import gym
from gym import spaces
import math


def init_network(target_x, target_y):
    s_batch = []
    a_batch = []
    r_batch = []
    for i in range(1, 50):
        for j in range(1, 50):
            s_batch.append([i, j])
            a_batch.append([target_x - i, target_y - j])
            distance = 100/(math.sqrt(pow(target_x - i, 2) + pow(target_y - j, 2)) + 1)
            r_batch.append(distance)
    return s_batch, a_batch, r_batch


if __name__ == '__main__':
    s = [1, 1, 1, 1]
    s1 = [[[1, 1], [1, 1]]]
    print(s1)
    print(s1)
