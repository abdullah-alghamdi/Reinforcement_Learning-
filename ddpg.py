# Author: Abdullah Alghamdi
''' This is a reimplementation of the Deep Deterministic Policy Griedent (DDPG) algorithm
based on the 'Continuous control with Deep Reinforcement Learning' (paper https://arxiv.org/pdf/1509.02971.pdf) 
The experiment is based on the paper where they used the OpenAI gym environment "Pendulum-v0", the deep networks structure 
and hyperparameters are slightly changed.
.
'''
from OUAction_noise import OrnsteinUhlenbeckActionNoise
import tensorflow as tf
import numpy as np
import gym
from collections import deque
import random 
import os

import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

#####################
#--> RL  hyperparameters  
GAMMA = 0.9     # discout factor
actor_alpha = 0.0001    # learning rate for actor
critic_alpha = 0.001    # learning rate for critic
soft_update_tau = 0.001 # soft update rate
#--> Experience Replay parameters
BUFFER_SIZE = 100000
BATCH_SIZE = 64

#-->soft update rate for the target network ( see the paper )

EPISODES = 200
STEPS =200
ENV_NAME = 'Pendulum-v0'

class DDPG(object):
        def __init__(self, action_dim, state_dim, action_bound):
                self.buffer = deque()
                self.counter = 0
                self.sess = tf.Session()
                self.action_dim = action_dim
                self.state_dim = state_dim
                self.action_bound = action_bound

                self.state = tf.placeholder(tf.float32, [None, state_dim], 's')
                self.nextState = tf.placeholder(tf.float32, [None, state_dim], 's_')
                self.reward = tf.placeholder(tf.float32, [None, 1], 'r')

        def actor_network(self, s, scope, trainable):
                with tf.variable_scope(scope):
                        fanin1 = 1/np.sqrt(self.state_dim)
                        fanin2 = 1/np.sqrt(400)

                        layer1 = tf.layers.dense(s, 400,activation=None, kernel_initializer=tf.random_uniform_initializer(-fanin1, fanin1), trainable=trainable)
                        b_norm = tf.layers.batch_normalization(layer1,trainable=trainable)
                        layer1 = tf.nn.relu(b_norm, 'relu')
                        layer2 = tf.layers.dense(layer1, 300,activation=None, kernel_initializer=tf.random_uniform_initializer(-fanin2, fanin2), trainable=trainable)
                        b_norm = tf.layers.batch_normalization(layer2,trainable=trainable)
                        layer2 = tf.nn.relu(b_norm, 'relu')
                        layer3 = tf.layers.dense(layer2, self.action_dim,activation=tf.tanh, kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3) , trainable=trainable)
                        scaled_action = tf.multiply(layer3, self.action_bound)
                        return scaled_action


        def critic_network(self,s,a, scope,trainable):
              
                with tf.variable_scope(scope):
                        fanin1 = 1/np.sqrt(self.state_dim)
                        fanin2 = 1/np.sqrt(400)

                        layer1 = tf.layers.dense(s, 400,activation=None, kernel_initializer=tf.random_uniform_initializer(-fanin1, fanin1), trainable=trainable)
                        b_norm = tf.layers.batch_normalization(layer1,trainable=trainable)
                        layer1 = tf.nn.relu(b_norm, 'relu')
                        w1_s = tf.Variable(tf.random_uniform([400, 300], -fanin2, fanin2), trainable=trainable)
                        w1_a = tf.Variable(tf.random_uniform([self.action_dim, 300], -fanin2, fanin2), trainable=trainable)
                        b1 = tf.Variable(tf.random_uniform([1,300], -fanin2, fanin2),trainable=trainable)

                        b_norm = tf.layers.batch_normalization(tf.matmul(layer1, w1_s) + tf.matmul(a, w1_a) + b1,trainable=trainable)
                        layer2 = tf.nn.relu(b_norm)
                        out_Q = tf.layers.dense(layer2, 1,activation=None, kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3) , trainable=trainable)
                        return out_Q
                        


        def build_graph(self):
                with tf.variable_scope('Actor'):
                    self.action = self.actor_network(self.state, scope='eval', trainable=True)
                    a_theta_prime = self.actor_network(self.nextState, scope='target', trainable=False)
                with tf.variable_scope('Critic'):
                 
                    q_theta = self.critic_network(self.state, self.action, scope='eval', trainable=True)
                    q_theta_prime = self.critic_network(self.nextState, a_theta_prime, scope='target', trainable=False)

                def soft_update(target,eval,is_init):
                        if is_init:
                            return [tf.assign(t, (1 - 0) * t + 0 * e) for t, e in zip(target, eval)]
                        else:
                            return [tf.assign(t, (1 - soft_update_tau) * t + soft_update_tau * e) for t, e in zip(target, eval)]

                # Find parameters by scope, to change it from eval to target network
                self.actor_eval = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
                self.actor_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
                # soft update the target network of the actor
                self.actor_target_update = soft_update(self.actor_target,self.actor_eval,False)
                self.init_actor_target = soft_update(self.actor_target,self.actor_eval,True)

                # do the same previous two steps for the critic
                self.critic_eval = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
                self.critic_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

                self.critic_target_update = soft_update(self.critic_target,self.critic_eval,False)
                self.init_critic_target = soft_update(self.critic_target,self.critic_eval,True)

                # optimize critic and actor

                y = self.reward + GAMMA * q_theta_prime    # equation (5) on the paper

                delta = tf.losses.mean_squared_error(labels=y, predictions=q_theta)
                self.optimize_critic = tf.train.AdamOptimizer(critic_alpha).minimize(delta, var_list=self.critic_eval)

                actor_loss = - tf.reduce_mean(q_theta) 
                self.optimize_actor = tf.train.AdamOptimizer(actor_alpha).minimize(actor_loss, var_list=self.actor_eval)


                
        # init graph
        def initialize_graph(self):
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(self.init_actor_target)
                self.sess.run(self.init_critic_target)
        #sample action give the state
        def get_action(self, s):
                return self.sess.run(self.action, {self.state: s[None, :]})[0]

        # Add experience to the experience replay buffer
        def append_experience(self, s, a, r, s2):
                experience = (s, a, r, s2)
                if self.counter < BUFFER_SIZE : 
                    self.buffer.append(experience)
                    self.counter += 1
                else:
                    self.buffer.popleft()
                    self.buffer.append(experience)

        #Sample batch from the experience replay buffer
        def sample_batch(self):
                batch = []
                if self.counter < BATCH_SIZE:
                    batch = random.sample(self.buffer, self.counter)
                else:
                    batch = random.sample(self.buffer, BATCH_SIZE)

                s_batch = np.array(list(map(lambda x: x[0], batch)))
                a_batch = np.array(list(map(lambda x: x[1], batch)))
                r_batch = np.array(list(map(lambda x: x[2], batch)))
                next_s_batch = np.array(list(map(lambda x: x[3], batch)))
                return s_batch, a_batch, np.transpose(r_batch[None,:]), next_s_batch

        #learning
        def learn(self,bs,ba,br,bs_):
                self.sess.run(self.actor_target_update)
                self.sess.run(self.critic_target_update)
                self.sess.run(self.optimize_actor , {self.state: bs})
                self.sess.run(self.optimize_critic, {self.state: bs, self.action: ba, self.reward: br, self.nextState: bs_})


# simple plot function that saves results.png to the directory
def plot_results(return_list):
        sns.set_style("whitegrid")
        grid = sns.FacetGrid(return_list, size=3, aspect=2)
        plt.plot(return_list)
        grid.set(xticks=np.arange(1,len(return_list),20), yticks=np.arange(min(return_list),max(return_list)+10,300))
        grid.set(xlabel='Episode', ylabel='Return')
        plt.tight_layout()
        plt.savefig("results.png")

if __name__ == '__main__':
        
        # init env from openAI gym
        env = gym.make(ENV_NAME)
        env = env.unwrapped
        env.seed(1)
        # get the states, actions, and action bounds dims
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]
        a_bound = env.action_space.high

        # init ddpg
        ddpg = DDPG(a_dim, s_dim, a_bound)
        # build the tensorflow graph
        ddpg.build_graph()
        # start the variables of the graph
        ddpg.initialize_graph()
        # add noise to the action ( look at the paper p.4 equation 7, the Ornstein Uhlenbeck algorithm is taken from openAI repo )
        # or just add use random to add the noise, and then clip the results between min and max actions [-2,2]
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(a_dim))
        noise = 1
        render_value = False
        return_per_episode = []
        for episode in range(EPISODES):
                # get the first state
                s = env.reset()
                # G is the sum of all rewards per episode ( return )
                G = 0
                for step in range(STEPS):
                        if render_value:
                                env.render()

                        a = ddpg.get_action(s) 
                        #a = np.clip(np.random.normal(a, noise), -2, 2) 
                        
                        a += (noise * actor_noise())

                        # get the next state and reward, then add the experience to the buffer
                        s_, r, _, _ = env.step(a)
                        ddpg.append_experience(s, a, r,s_)


                        if ddpg.counter > BATCH_SIZE:
                                #if noise>0:
                                 #   noise *= 0.9999 # reduce the noise 

                                # sample batch from the buffer, and learn
                                s_batch,a_batch,r_batch,next_s_batch = ddpg.sample_batch()
                                ddpg.learn(s_batch,a_batch,r_batch,next_s_batch)

                        #swap states and update G
                        s = s_
                        G += r

                        if step ==STEPS-1:
                            print('Episode: {} Reward: {}'.format(episode,int(G)))
                            if episode > (EPISODES/2):
                                render_value = True

                            
                return_per_episode.append(G)
        plot_results(return_per_episode)






