""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with Tensorflow

Authors: 
Patrick Emami
Shusen Wang
"""


import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
# from neural_network import NeuralNetworks
from neural_network_share_weight import NeuralNetworks
from replay_buffer import ReplayBuffer
import os
import copy
import math
import agent
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000  # 50000
# Max episode length
MAX_EP_STEPS = 200
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001  # 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001  # 0.001
# Discount factor
GAMMA = 0.99  # 0.99
# Soft target update param
TAU = 0.001  # 0.001
# number of weak agent

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'MultiCar-v0'   # Pendulum-v0
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
# File for saving reward and qmax
RESULTS_FILE = './results/rewards.npz'
# Directory for storing model
MODEL_DIR = './results/model'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 1000  # 1000
MINIBATCH_SIZE = 100  # 128


class Actor:
    
    def __init__(self, sess, network, learning_rate):
        self.sess = sess
        self.learning_rate = learning_rate
        _, self.a_dim, _ = network.get_const()
        
        self.inputs = network.get_input_state(is_target=False)
        self.out = network.get_actor_out(is_target=False)
        self.params = network.get_actor_params(is_target=False)
        
        # This gradient will be provided by the critic network
        self.critic_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients
        self.policy_gradient = tf.gradients(tf.multiply(self.out, -self.critic_gradient), self.params)
        
        # Optimization Op        
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.policy_gradient, self.params))
        
    def train(self, state, c_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: state,
            self.critic_gradient: c_gradient
        })

    def predict(self, state):
        return self.sess.run(self.out, feed_dict={
            self.inputs: state
        })
    

class ActorTarget:
    
    def __init__(self, sess, network, tau):
        self.sess = sess
        self.tau = tau
        
        self.inputs = network.get_input_state(is_target=True)
        self.out = network.get_actor_out(is_target=True)
        self.params = network.get_actor_params(is_target=True)
        param_num = len(self.params)
        self.params_other = network.get_actor_params(is_target=False)
        assert(param_num == len(self.params_other))
        
        # update target network
        self.update_params = \
            [self.params[i].assign(tf.multiply(self.params_other[i], self.tau) +
                                   tf.multiply(self.params[i], 1. - self.tau))
                for i in range(param_num)]
    
    def train(self):
        self.sess.run(self.update_params)

    def predict(self, state):
        return self.sess.run(self.out, feed_dict={self.inputs: state})
        
        
class Critic:
    def __init__(self, sess, network, learning_rate):
        self.sess = sess
        self.learning_rate = learning_rate

        # Create the critic network
        self.state, self.action = network.get_input_state_action(is_target=False)
        self.out = network.get_critic_out(is_target=False)
        self.params = network.get_critic_params(is_target=False)
        
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        # self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.loss = tf.nn.l2_loss(self.predicted_q_value - self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action)

    def train(self, state, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.state: state,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, state, action):
        return self.sess.run(self.out, feed_dict={
            self.state: state,
            self.action: action
        })
        
    def action_gradients(self, state, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: state,
            self.action: actions
        })


class CriticTarget:
    def __init__(self, sess, network, tau):
        self.sess = sess
        self.tau = tau

        # Create the critic network
        self.state, self.action = network.get_input_state_action(is_target=True)
        self.out = network.get_critic_out(is_target=True)
        
        # update target network
        self.params = network.get_critic_params(is_target=True)
        param_num = len(self.params)
        self.params_other = network.get_critic_params(is_target=False)
        assert(param_num == len(self.params_other))
        self.update_params = \
            [self.params[i].assign(tf.multiply(self.params_other[i], self.tau) +
                                   tf.multiply(self.params[i], 1. - self.tau))
                for i in range(param_num)]
            
    def predict(self, state, action):
        return self.sess.run(self.out, feed_dict={
            self.state: state,
            self.action: action
        })

    def train(self):
        self.sess.run(self.update_params)
        

# ===========================
#   Tensorflow Summary Ops
# ===========================


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Init network
# ===========================


def init_network(target_x, target_y, s_dim, a_dim, init_buffer):  # rewrite to init potential function
    # init_buffer = ReplayBuffer(100*100, 123)
    width = agent.STATE_WIDTH
    for i in range(0, width, 1):
        for j in range(0, width, 1):
            s = [i, j, 0, 0]  # position_x, position_y, distance_obstacle_x, distance_obstacle_y
            a = [(target_x - i)/5, (target_y - j)/5]  # velocity_x, velocity_y
            distance = (math.sqrt(pow(target_x - i, 2) + pow(target_y - j, 2)))
            if distance == 0:
                distance = 0.1
            r = 100 / distance
            if i == width and j == width:
                t = True
            else:
                t = False
            s2 = [i + a[0], j + a[1], 0, 0]
            init_buffer.add(np.reshape(s, (s_dim,)), np.reshape(a, (a_dim,)), r,
                            t, np.reshape(s2, (s_dim,)))
    s_batch, a_batch, r_batch, t_batch, s2_batch = \
        init_buffer.sample_batch(MINIBATCH_SIZE)
    return s_batch, a_batch, r_batch, t_batch, s2_batch


# ===========================
#   Agent Training
# ===========================


def train(sess, env, network_all):  # all agent train together
    arr_reward = np.zeros(MAX_EPISODES)
    arr_qmax = np.zeros(MAX_EPISODES)
    saver = tf.train.Saver()  # save

    actor = []
    actor_target = []
    critic = []
    critic_target = []
    s_dim = []
    a_dim = []
    summary_ops = []
    summary_vars = []
    replay_buffer = []
    # before train, we need to
    # init network
    s_dim_init, a_dim_init, _ = network_all[0].get_const()
    for num in range(len(network_all)):  # each agent
        network = network_all[num]
        actor.append(Actor(sess, network, ACTOR_LEARNING_RATE))
        actor_target.append(ActorTarget(sess, network, TAU))
        critic.append(Critic(sess, network, CRITIC_LEARNING_RATE))
        critic_target.append(CriticTarget(sess, network, TAU))
    
        s_dim_each, a_dim_each, _ = network.get_const()
        s_dim.append(s_dim_each)
        a_dim.append(a_dim_each)

        # Set up summary Ops
        summary_ops_each, summary_vars_each = build_summaries()
        summary_ops.append(summary_ops_each)
        summary_vars.append(summary_vars_each)
        sess.run(tf.global_variables_initializer())
        # tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        actor_target[num].train()
        critic_target[num].train()

        # init every network
        init_buffer = ReplayBuffer(100*100, RANDOM_SEED)
        init_s, init_a, init_r, init_t, init_s2 = init_network(
            agent.TARGET_X, agent.TARGET_Y, s_dim_init, a_dim_init, init_buffer)
        _ = critic_target[num].predict(init_s, actor_target[num].predict(init_s2))
        y_i = []
        for k in range(init_a.shape[0]):
            y_i.append(init_r[k])
        predicted_q_value, _ = critic[num].train(init_s, init_a, np.reshape(y_i, (init_a.shape[0], 1)))
        a_outs = actor[num].predict(init_s)
        grads = critic[num].action_gradients(init_s, a_outs)
        actor[num].train(init_s, grads[0])
        actor_target[num].train()
        critic_target[num].train()
        # Initialize replay memory
        # replay_buffer.append(init_buffer)
        replay_buffer.append(ReplayBuffer(BUFFER_SIZE, RANDOM_SEED))
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    '''restore model which has been trained'''
    has_point = os.path.isfile('./results/model.meta')
    if has_point:
        restore_saver = tf.train.import_meta_graph('./results/model.meta')
        restore_model = tf.train.latest_checkpoint('./results')
        restore_saver.restore(sess, restore_model)
    for i in range(MAX_EPISODES):  # every episode

        # ss = env.reset()
        # s = []
        # for fine in range(len(ss)):
        #     each_s = ss[fine][my_id]
        #     s.append(each_s)
        s = env.reset()  # states' collection
        ep_reward = 0
        ep_ave_max_q = 0
        if (i % 1000 == 0 and i != 0) or i == MAX_EPISODES:
            saver.save(sess, MODEL_DIR)
            start_time = time.localtime(time.time())
            print(start_time)
        for j in range(MAX_EP_STEPS):  # each step

            if RENDER_ENV:
                env.render()
                # if i > 1000:
                #     env.render()

            # Added exploration noise
            a_all = []  # actions' collection
            for num in range(len(network_all)):  # Attention: action is Box(2, )
                # print(s[:, num])
                my_test = np.reshape(s[:, num], (1, s_dim[num]))
                # a = actor[num].predict(my_test) + [(1. / (1. + i)), (1. / (1. + i))]  # rewrite:for each agent
                a = actor[num].predict(my_test)
                a_all.append(a[0])
            s1 = copy.deepcopy(s)
            s2, r, terminal, info = env.step(a_all)
            # width = len(network_all)
            # ep_ave_max_q_a = np.zeros((1, width))
            for num in range(len(network_all)):

                a_re = np.reshape(a_all[num], (a_dim[num],))
                s2_re = np.reshape(s2[:, num], (s_dim[num],))
                s_re = np.reshape(s1[:, num], (s_dim[num],))
                replay_buffer[num].add(s_re, a_re, r, terminal, s2_re)

                # Keep adding experience to the memory until
                # there are at least mini batch size samples
                if replay_buffer[num].size() > MINIBATCH_SIZE:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer[num].sample_batch(MINIBATCH_SIZE)

                    # Calculate targets
                    target_q = critic_target[num].predict(s2_batch, actor_target[num].predict(s2_batch))

                    y_i = []  # calculate reward
                    for k in range(MINIBATCH_SIZE):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + GAMMA * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = critic[num].train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                    ep_ave_max_q += np.amax(predicted_q_value)
                    # ep_ave_max_q_a[num] += np.amax(predicted_q_value)
                    # ep_ave_max_q += np.mean(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = actor[num].predict(s_batch)
                    grads = critic[num].action_gradients(s_batch, a_outs)
                    actor[num].train(s_batch, grads[0])

                    # Update target networks
                    actor_target[num].train()
                    critic_target[num].train()

            s = s2
            ep_reward += r
            # ep_ave_max_q = np.amax(ep_ave_max_q_a)
            if terminal:
                for k in range(len(summary_ops)):
                    summary_str = sess.run(summary_ops[k], feed_dict={
                        summary_vars[k][0]: ep_reward,
                        summary_vars[k][1]: ep_ave_max_q / float(j + 1)
                    })
                    writer.add_summary(summary_str, i)
                    writer.flush()

                print('Reward: ' + str(ep_reward) + ',   Episode: ' + str(i) + ',    Qmax: ' +
                      str(ep_ave_max_q / float(j + 1)))
                arr_reward[i] = ep_reward
                arr_qmax[i] = ep_ave_max_q / float(j + 1)
                
                if i % 100 == 99:
                    np.savez(RESULTS_FILE, arr_reward[0:i], arr_qmax[0:i])

                break


def main(_):
    with tf.Session() as sess:

        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        network = []
        for i in range(len(env.observation_space)):

            state_dim = env.observation_space[i].shape[0]
            action_dim = env.action_space[i].shape[0]
            action = env.action_space[i]  # action is a Box[2*2]
            # action_dim = env.action_space.shape
            action_bound = action.high[0]
            # Ensure action bound is symmetric
            # print(env.action_space[i].high)
            # print(env.action_space[i].low)
            assert (env.action_space[i].high[0] == -env.action_space[i].low[1])
            # assert (env.action_space[i].low[0] == -env.action_space[i].low[1])
            _i = NeuralNetworks(state_dim, action_dim, action_bound)
            network.append(_i)

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = wrappers.Monitor(env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, MONITOR_DIR, force=True)
        start_time = time.localtime(time.time())
        print(start_time)
        train(sess, env, network)

        if GYM_MONITOR_EN:
            env.close()


if __name__ == '__main__':
    tf.app.run()
