from gym import spaces
import numpy as np
from gym import logger
import random
# from gym.utils import seeding
import math

TARGET_X = 225
TARGET_Y = 225
REWARD = 200
STATE_WIDTH = 450


class Agent(object):  # single agent

    def __init__(self, my_id):  # rewrite
        self.id = my_id
        self.xth = 0
        self.counts = 0
        self.in_X_min = 0
        self.in_X_max = 450
        self.in_Y_min = 0
        self.in_Y_max = 450
        self.positionX = random.uniform(self.in_X_min, self.in_X_max)  # initial position
        self.positionY = random.uniform(self.in_Y_min, self.in_Y_max)
        self.stateWidth = STATE_WIDTH  # window width
        self.r_ob = 10  # obstacle radius
        self.r_arr = 30  # distance to the neighbour which is arrived (target)
        self.r_com = 100  # communicate radius
        self.targetX = TARGET_X  # target point
        self.targetY = TARGET_Y
        self.x_tar = 0  # relative distance to target
        self.y_tar = 0
        self.x_obs = 0  # relative distance to obs
        self.y_obs = 0
        self.neighbour = []
        low = -10.0
        high = 10.0
        # self.action_space = spaces.Discrete(5)  # 0, 1, 2，3，4: 不动\上\下\左\右
        self.action_space = spaces.Box(low=np.array([low, low]), high=np.array([high, high]))  # x, y speed
        self.x_speed = []
        self.y_speed = []
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]),
                                            high=np.array([self.stateWidth, self.stateWidth, self.r_ob, self.r_ob]))
        '''set obstacle'''
        self.obstacleNum = 4  # static obstacle
        self.obstacle = self.create_obs()
        self.obstacleMargin = self.obstacle_margin()
        self.find_tar = False  # arrive target point or not
        self.find_obs = False  # arrive obstacle or not
        self.arrived = False  # arrive other target agent or not
        self.energy = 0
        self.state = None
        self.step_beyond_done = 0
        self.done = False
        self.distance = 10000

    '''find nearest distance to obstacle'''
    def find_nearest_obs(self):  #
        dis = 10000
        nearest_x = 0
        nearest_y = 0
        x = self.positionX
        y = self.positionY
        count1 = 0
        count2 = 0
        for i in self.obstacleMargin:
            dis1 = math.sqrt(pow(x - i[0], 2) + pow(y - i[1], 2))
            if dis1 < dis:
                dis = dis1
                nearest_x = x - i[0]  # coordinate to direction
                nearest_y = y - i[1]
                count1 = i[0]
                count2 = i[1]
            else:
                dis = dis
                nearest_y = nearest_y
                nearest_x = nearest_x
                count1 = count1
                count2 = count2
        test_need_states = math.sqrt(math.pow(nearest_x, 2) + math.pow(nearest_y, 2))
        if self.r_ob < test_need_states < (3*self.r_ob):
            self.x_obs = nearest_x
            self.y_obs = nearest_y
        else:
            self.x_obs = 0
            self.y_obs = 0
        return dis

    '''calculate  neighbour(which has energy)'s position, find out which agent is touched'''
    def find_nearest_arr(self):
        arrived_target = []
        neighbour = self.neighbour
        dis = 10000
        nearest_x = 0
        nearest_y = 0
        x = self.positionX
        y = self.positionY
        agent = None
        has_target = False  # has arrived neighbour
        for i in neighbour:
            if i.energy != 0:  # ordinary agent
                arrived_target.append([2.0, 2, i.positionX, i.positionY])
                has_target = True
                dis1 = math.sqrt(pow(x - i.positionX, 2) + pow(y - i.positionY, 2))
                if dis1 < dis:
                    dis = dis1
                    nearest_x = x - i.positionX  # coordinate to direction
                    nearest_y = y - i.positionY
                    agent = i
                else:
                    dis = dis
                    nearest_y = nearest_y
                    nearest_x = nearest_x
                    agent = agent
        return has_target, dis, agent
        # return dis, nearest_x, nearest_y
        # return has_target, arrived_target

    '''create obstacle, every agent knows all the obstacles'''
    def create_obs(self):
        obstacle = []
        x = self.neighbour
        for i in x:
            if i.energy == 0 and i.id != self.id:  # ordinary agent
                obstacle.append([2., 2., i.positionX, i.positionY])
        obstacle1 = np.array([250., 10., 50., 290.])  # width \height \position_x\ position_y
        # obstacle2 = np.array([10., 250., 290., 50.])
        # obstacle3 = np.array([100., 20., 150., 340.])
        # obstacle4 = np.array([20., 100., 340., 150.])
        # margin 方框
        obstacle5 = np.array([450., 1, 0, 0])
        obstacle6 = np.array([1, 450., 0, 0])
        obstacle7 = np.array([1, 450., 450, 0])
        obstacle8 = np.array([450., 1, 0, 450])

        obstacle.append(obstacle1)
        # obstacle.append(obstacle2)
        # obstacle.append(obstacle3)
        # obstacle.append(obstacle4)
        obstacle.append(obstacle5)
        obstacle.append(obstacle6)
        obstacle.append(obstacle7)
        obstacle.append(obstacle8)

        return obstacle

    '''calculate obstacle margin'''
    def obstacle_margin(self):
        obstacle_margin = []
        for obs in self.obstacle:
            width = 0
            while width < obs[0]:
                x = width + obs[2]
                y1 = obs[3] + obs[1]
                y2 = obs[3]
                obstacle_margin.append([x, y1])
                obstacle_margin.append([x, y2])
                width = width + 1
            height = 0
            while height < obs[1]:
                y = height + obs[3]
                x1 = obs[2] + obs[0]
                x2 = obs[2]
                obstacle_margin.append([x1, y])
                obstacle_margin.append([x2, y])
                height = height + 1
        return obstacle_margin

    '''find neighbour'''
    def find_neighbour(self, agents):
        neighbour = []
        x = self.positionX
        y = self.positionY
        for i in agents:
            dis = math.sqrt(pow(x - i.positionX, 2) + pow(y - i.positionY, 2))
            if dis < self.r_com and i.id != self.id:
                neighbour.append(i)
        return neighbour

    '''calculate single agent reward'''
    def get_reward(self, all_has_target, agent):
        reward = 0
        self.distance = math.sqrt(pow(self.targetX - self.positionX, 2) + pow(self.targetY - self.positionY, 2))
        reward_new = REWARD / self.distance
        if reward_new > REWARD/2:
            reward_new = REWARD/2
        if not self.done:
            if all_has_target is True:
                if self.find_tar is True:  # arrive target
                    self.energy = REWARD
                    self.done = True
                    reward = REWARD * pow(0.4, 0)  # change state and rewrite reward for other agents
                elif self.find_obs is True:  # stopped by obstacle
                    self.done = True
                    reward = -REWARD
                else:
                    reward = -REWARD/200 + reward_new
            elif all_has_target is False:
                if self.arrived is True:  # need to know the agent been touched (calculated in multi_env)
                    self.done = True
                    reward = agent.energy * 0.4
                    self.energy = reward
                    agent.energy = agent.energy * 0.6
                elif self.find_obs is True:  # stopped by obstacle
                    self.done = True
                    reward = -REWARD
                else:
                    reward = -REWARD/200 + reward_new
        else:
            reward = 0
            if self.step_beyond_done == 0:
                logger.warn("agent" + str(self.id) + "has done.")
                logger.warn("agent" + str(self.id) + "is arrived?" + str(self.arrived))
                logger.warn("agent" + str(self.id) + "has obstacle?" + str(self.find_obs))
                logger.warn("agent" + str(self.id) + "has target?" + str(self.find_tar))
        return reward

    '''calculate single agent reward'''
    def get_reward_1(self, has_target, agent):
        reward = 0
        self.distance = math.sqrt(pow(self.targetX - self.positionX, 2) + pow(self.targetY - self.positionY, 2))
        reward_new = 10 / self.distance
        if not self.done:
            if has_target is False:
                if self.find_tar is True:  # arrive target
                    self.energy = 100
                    self.done = True
                    reward = 100 * pow(0.4, 0)  # change state and rewrite reward for other agents
                elif self.find_obs is True:  # stopped by obstacle
                    self.done = True
                    reward = -100
                else:

                    if reward_new > 50:
                        reward_new = 50
                    reward = -1 + reward_new
            elif has_target is True:
                if self.arrived is True:  # need to know the agent been touched (calculated in multi_env)
                    self.done = True
                    reward = agent.energy * 0.4
                    agent.energy = agent.energy * 0.6
                elif self.find_obs is True:  # stopped by obstacle
                    self.done = True
                    reward = -100
                else:
                    reward = -1 + reward_new
        else:
            reward = 0
            if self.step_beyond_done == 0:
                logger.warn("agent" + str(self.id) + "has done.")
                logger.warn("agent" + str(self.id) + "is arrived?" + str(self.arrived))
                logger.warn("agent" + str(self.id) + "has obstacle?" + str(self.find_obs))
                logger.warn("agent" + str(self.id) + "has target?" + str(self.find_tar))
        return reward

    '''calculate agent to the agent which has energy'''
    def distance_to(self):  # return
        x = self.positionX
        y = self.positionY
        arrived_neighbour = []
        dis = 10000
        agent = None
        for i in self.neighbour:
            if i.arrived is True or i.find_tar is True:
                arrived_neighbour.append(i)
                dis1 = math.sqrt(pow(x - i.positionX, 2) + pow(y - i.positionY, 2))
                if dis1 < dis:
                    dis = dis1
                    agent = i
                else:
                    dis = dis
                    agent = agent
        return dis, agent

    '''find out agent is arrived or not and calculate the steps beyond the arrived time'''
    def steps_beyond(self):
        if self.done is False:
            self.step_beyond_done = self.step_beyond_done
        else:
            self.step_beyond_done = self.step_beyond_done + 1

    '''init '''
