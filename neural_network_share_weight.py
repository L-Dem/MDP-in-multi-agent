""" 
Define neural network structures of the actor and critic method

The actor and critic networks share the layers: State ==> FC ==> ReLU ==> Feature

The algorithm is tested on the Pendulum-v0 OpenAI gym task 

Author: Shusen Wang
"""
import tensorflow as tf
import numpy as np


class NeuralNetworks:
    '''
    State_Feature: State to Feature
    Action_Feature: Action to Feature
    Actor: State_Feature to Action
    Critic: (State_Feature, Action_Feature) to Value
    '''
    
    def __init__(self, state_dim, action_dim, action_bound):
        # size of layers
        self._S_FEATURE_DIM = 64
        self._ACTOR_H1_DIM = 128
        self._CRITIC_H1_DIM = 128
        
        # constants
        self._S_DIM = state_dim
        self._A_DIM = action_dim
        self._A_BOUND = action_bound

        self.actor_params = None
        self.actor_target_params = None
        self.critic_params = None
        self.critic_target_params = None
        self.num_params = len(tf.trainable_variables())
        # Create actor network
        # features extracted from states
        self.input_state, self.state_feature = self._create_state_feature(sigma=0.3)
        param_state_feature = tf.trainable_variables()[self.num_params:]
        # actor network
        self.actor_y = self._create_actors(self.state_feature)
        # parameters of actor network

        self.actor_params = tf.trainable_variables()[self.num_params:]
        num_params1 = len(tf.trainable_variables())
        
        # Create actor target network
        # features extracted from states
        self.input_state_target, self.state_feature_target = self._create_state_feature(sigma=0.3)
        param_state_feature_target = tf.trainable_variables()[num_params1:]
        # actor target network
        self.actor_target_y = self._create_actors(self.state_feature_target)
        # parameters of actor target network
        self.actor_target_params = tf.trainable_variables()[num_params1:]
        num_params2 = len(tf.trainable_variables())
        
        # Create critic network
        # features extracted from states and actions
        self.input_state_critic = self.input_state
        self.state_feature_critic = self.state_feature
        self.input_action = tf.placeholder(tf.float32, [None, self._A_DIM])
        # critic network
        self.critic_y = self._create_critics(self.state_feature_critic, self.input_action)
        # parameters of critic network
        self.critic_params = param_state_feature + tf.trainable_variables()[num_params2:]
        num_params3 = len(tf.trainable_variables())

        # Create critic target network
        # features extracted from states and actions
        self.input_state_critic_target = self.input_state_target
        self.state_feature_critic_target = self.state_feature_target
        self.input_action_target = tf.placeholder(tf.float32, [None, self._A_DIM])
        # critic target network
        self.critic_target_y = self._create_critics(self.state_feature_critic_target, self.input_action_target)
        # parameters of critic target network
        self.critic_target_params = param_state_feature_target + tf.trainable_variables()[num_params3:]

    def get_const(self):
        return self._S_DIM, self._A_DIM, self._A_BOUND
    
    def get_input_state(self, is_target=False):
        if is_target:
            return self.input_state_target
        else:
            return self.input_state
    
    def get_actor_out(self, is_target=False):
        if is_target:
            return self.actor_target_y
        else:
            return self.actor_y
        
    def get_actor_params(self, is_target=False):
        if is_target:
            return self.actor_target_params
        else:
            return self.actor_params

    def get_input_state_action(self, is_target=False):
        if is_target:
            return (self.input_state_critic_target, self.input_action_target)
        else:
            return (self.input_state_critic, self.input_action)
    
    def get_critic_out(self, is_target=False):
        if is_target:
            return self.critic_target_y
        else:
            return self.critic_y
        
    def get_critic_params(self, is_target=False):
        if is_target:
            return self.critic_target_params
        else:
            return self.critic_params

    # ================ Shared Functions ================ #
    def weight_variable(shape, value=0.1, rand='normal'):
        initial = None
        if rand == 'normal':
            initial = tf.truncated_normal(shape, stddev=value)
        elif rand == 'uniform':
            initial = tf.random_uniform(shape, minval=-value, maxval=value)
        return tf.Variable(initial)

    def bias_variable(shape, value=0.01):
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial)
    
    # =========== Define Networks Structures =========== #
    '''State ==> FC ==> ReLU ==> Feature'''
    def _create_state_feature(self, sigma=0.1):
        x = tf.placeholder(tf.float32, [None, self._S_DIM])
        w = NeuralNetworks.weight_variable([self._S_DIM, self._S_FEATURE_DIM], value=sigma, rand='normal')
        b = NeuralNetworks.bias_variable([self._S_FEATURE_DIM])
        y = tf.nn.relu(tf.matmul(x, w) + b)
        return x, y

    ''' State_Feature ==> FC ==> ReLU ==> FC ==> Tanh ==> Scale ==> Action'''
    def _create_actors(self, feature):
        w1 = NeuralNetworks.weight_variable([self._S_FEATURE_DIM, self._ACTOR_H1_DIM])
        b1 = NeuralNetworks.bias_variable([self._ACTOR_H1_DIM])
        h1 = tf.nn.relu(tf.matmul(feature, w1) + b1)
        w2 = NeuralNetworks.weight_variable([self._ACTOR_H1_DIM, self._A_DIM])
        b2 = NeuralNetworks.bias_variable([self._A_DIM])
        h2 = tf.nn.tanh(tf.matmul(h1, w2) + b2)
        y = tf.multiply(h2, self._A_BOUND)
        return y

    '''
        Hidden Layer: 
            H1 = State_Feature * W1s + Action * W1a + Bias
        Critic:
            H1 ==> ReLU ==> FC ==> Value
    '''
    def _create_critics(self, s_feature, action):

        w1_s = NeuralNetworks.weight_variable([self._S_FEATURE_DIM, self._CRITIC_H1_DIM], value=0.3, rand='normal')
        w1_a = NeuralNetworks.weight_variable([self._A_DIM, self._CRITIC_H1_DIM])
        b1 = NeuralNetworks.bias_variable([self._CRITIC_H1_DIM], value=0.0)
        h1 = tf.add(tf.matmul(s_feature, w1_s), tf.matmul(action, w1_a))
        y1 = tf.nn.relu(tf.add(h1, b1))
        w2 = NeuralNetworks.weight_variable([self._CRITIC_H1_DIM, 1], value=0.01, rand='uniform')
        b2 = NeuralNetworks.bias_variable([1], value=0.0)
        y2 = tf.matmul(y1, w2) + b2
        return y2
