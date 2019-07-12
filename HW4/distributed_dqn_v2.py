# Chih-Hsiang Wamg
# 2019/6/2

import gym
import torch
import time
import os
import ray
import numpy as np
import csv

from tqdm import tqdm
from random import uniform, randint

import io
import base64
from IPython.display import HTML

from dqn_model import DQNModel
from dqn_model import _DQNModel
from memory import ReplayBuffer
from memory_remote import ReplayBuffer_remote

import matplotlib.pyplot as plt

from custom_cartpole import CartPoleEnv

FloatTensor = torch.FloatTensor

# =================== Helper Function ===================
def plot_result(total_rewards ,learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
        
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.show()

# =================== Hyperparams ===================
hyperparams_CartPole = {
    'epsilon_decay_steps' : 100000, 
    'final_epsilon' : 0.1,
    'batch_size' : 32, 
    'update_steps' : 10, 
    'memory_size' : 2000, 
    'beta' : 0.99, 
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True
}

# =================== Initialize Environment ===================
# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'
# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}
# Register the environment
env_CartPole = CartPoleEnv()

# =================== Ray Init ===================
ray.shutdown()
# ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)
ray.init()

# =================== DQN ===================
class DQN_agent(object):
    def __init__(self, env, hyper_params, action_space = len(ACTION_DICT)):
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        
        """
            beta: The discounted factor of Q-value function
            (epsilon): The explore or exploit policy epsilon. 
            initial_epsilon: When the 'steps' is 0, the epsilon is initial_epsilon, 1
            final_epsilon: After the number of 'steps' reach 'epsilon_decay_steps', 
                The epsilon set to the 'final_epsilon' determinately.
            epsilon_decay_steps: The epsilon will decrease linearly along with the steps from 0 to 'epsilon_decay_steps'.
        """
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']

        """
            episode: Record training episode
            steps: Add 1 when predicting an action
            learning: The trigger of agent learning. It is on while training agent. It is off while testing agent.
            action_space: The action space of the current environment, e.g 2.
        """
        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True
        self.action_space = action_space

        """
            input_len: The input length of the neural network. It equals to the length of the state vector.
            output_len: The output length of the neural network. It is equal to the action space.
            eval_model: The model for predicting action for the agent.
            target_model: The model for calculating Q-value of next_state to update 'eval_model'.
            use_target_model: Trigger for turn 'target_model' on/off
        """
        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)
        # memory: Store and sample experience replay.
        # self.memory = ReplayBuffer(hyper_params['memory_size'])
        
        """
            batch_size: Mini batch size for training model.
            update_steps: The frequence of traning model
            model_replace_freq: The frequence of replacing 'target_model' by 'eval_model'
        """
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
    
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate
    
    def explore_or_exploit_policy(self, state):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon, 
                               self.final_epsilon,
                               self.steps,
                               self.epsilon_decay_steps)
        
        if p < epsilon:
            #return action
            return randint(0, self.action_space - 1)
        else:
            #return action
            return self.greedy_policy(state)
        
    def greedy_policy(self, state):
        return self.eval_model.predict(state)

# =================== Ray Servers ===================
@ray.remote    
class DQNModel_server(DQN_agent):
    def __init__(self, env, hyper_params, memory):
        super().__init__(env, hyper_params)
        self.memory_server = memory


    def update_batch(self):
        batch = ray.get(self.memory_server.sample.remote(self.batch_size))
        if not batch:
            return
        (states, actions, reward, next_states,
         is_terminal) = batch
        
        states = states
        next_states = next_states
        terminal = FloatTensor([0 if t else 1 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)
        
        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]
        
        # Calculate target
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)
            
        q_target = reward + self.beta * torch.max(q_next, dim=1)[0] * terminal
        
        # update model
        self.eval_model.fit(q_values, q_target)   

    def getReturn(self):
        return self.eval_model, self.steps

    def learn(self):
        self.steps += 1

        if self.steps % self.update_steps == 0:
            self.update_batch()

        if self.steps % self.model_replace_freq == 0 and self.use_target_model:
            self.target_model.replace(self.eval_model)

# =================== Workers ===================    
@ray.remote
def collecting_worker(model, env, hyper_params, max_episode_steps, memory, tasks_num):
    for _ in range(tasks_num):
        state = env.reset()
        done = False
        steps = 0

        eval_model, total_steps = ray.get(model.getReturn.remote())
        while steps < max_episode_steps and not done:
            steps += 1
            a = ray.get(model.explore_or_exploit_policy.remote(state))
            s_, reward, done, info = env.step(a)
            memory.add.remote(state, a, reward, s_, done)
            state = s_
            model.learn.remote()

@ray.remote
def evaluation_worker(model, env, max_episode_steps, tasks_num):
    total_reward = 0
    for _ in range(tasks_num):
        state = env.reset()
        done = False
        steps = 0

        while steps < max_episode_steps and not done:
            steps += 1
            action = ray.get(model.greedy_policy.remote(state))
            state, reward, done, _ = env.step(action)
            total_reward += reward

    return total_reward / tasks_num
        
# =================== Agent ===================
class distributed_DQN_agent():
    def __init__(self, env, hyper_params, training_episodes, test_interval, cw_num = 4, ew_num = 4, trials = 30):
        self.memory_server = ReplayBuffer_remote.remote(hyper_params['memory_size'])
        self.model_server = DQNModel_server.remote(env, hyper_params, self.memory_server)
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.hyper_params = hyper_params
        self.training_episodes = training_episodes
        self.test_interval = test_interval
        self.trials = trials

    def learn_and_evaluate(self):
        results = []
        num = 0
        

        for _ in range(self.training_episodes // self.test_interval):
            num += 1
            collector_ids, evaluators_ids = [], []
            # learn
            for _ in range(self.cw_num):
                collector_ids.append(collecting_worker.remote(self.model_server, self.env, self.hyper_params,  
                                                    self.max_episode_steps, self.memory_server, test_interval//self.cw_num))
                
            # evaluate
            for _ in range(self.ew_num):
                evaluators_ids.append(evaluation_worker.remote(self.model_server, self.env, self.max_episode_steps,
                                                                                        self.trials//self.ew_num))

            total_reward = sum(ray.get(evaluators_ids))
            avg_reward = total_reward / self.ew_num
            print(num, " avg_reward: ", avg_reward)
            results.append(avg_reward)

        return results

# =================== Main ===================
env_CartPole.reset()
cw_num = 8 # collecter workers
ew_num = 4 # evaluater workers
training_episodes, test_interval, trials = 7000, 50, 30
agent = distributed_DQN_agent(env_CartPole, hyperparams_CartPole, training_episodes, test_interval, cw_num, ew_num, trials)
start_time = time.time()
result = agent.learn_and_evaluate()
print("running time: ", time.time()-start_time)
print(result)

with open('hw4_result.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(result)
file.close()

plot_result(result, test_interval, ["batch_update with target_model"])