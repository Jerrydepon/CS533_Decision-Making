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
# from memory import ReplayBuffer
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
        
    # Linear decrease function for epsilon
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
    def __init__(self, env, hyper_params, training_episodes, test_interval, memory):
        super().__init__(env, hyper_params)
        
        self.collector_done = False
        self.evaluator_done = False
        self.episode = 0
        self.training_episodes = training_episodes
        self.test_interval = test_interval
        self.memory_server = memory
        self.previous_eval = []
        self.reuslt_count = 0
        self.results = [0] * (training_episodes // test_interval + 1)
        self.update_count = 0

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

    # version 1 -- normal method
    def learn(self, done, state):
        self.steps += 1
        
        if done:
            self.episode += 1
            if self.episode % 50 == 0:
                print(self.episode)
        if self.get_done():
            return self.explore_or_exploit_policy(state), self.collector_done
        if self.episode // self.test_interval + 1 > len(self.previous_eval):
            self.previous_eval.append(self.eval_model)
        if self.steps % self.update_steps == 0:
            self.update_batch()
        if self.steps % self.model_replace_freq == 0:
            self.target_model.replace(self.eval_model)
        return self.explore_or_exploit_policy(state), self.collector_done
    # # version 2 -- simplified method  
    # def learn(self, done, update, rem, state):
    #     if update:
    #         self.steps += self.update_steps
    #         self.update_batch()
    #         # self.update_count += 1
    #         # print("update_count", self.update_count)
        
    #     self.steps += rem
    #     if done:
    #         self.episode += 1
    #         # if self.episode % 50 == 0:
    #         #     print(self.episode)
    #     if self.get_done():
    #         return self.explore_or_exploit_policy(state), self.collector_done
    #     if self.episode // self.test_interval + 1 > len(self.previous_eval):
    #         self.previous_eval.append(self.eval_model)
    #     if self.steps % self.model_replace_freq < 10:
    #         self.target_model.replace(self.eval_model)
    #     return self.explore_or_exploit_policy(state), self.collector_done
        
    def get_done(self):
        if self.episode >= self.training_episodes:
            self.collector_done = True
        return self.collector_done
    
    # evalutor
    def add_result(self, result, num):
        self.results[num] = result
    
    def get_results(self):
        return self.results
    
    def ask_evaluation(self):
        if len(self.previous_eval) > self.reuslt_count:
            num = self.reuslt_count
            evaluation = self.previous_eval[num]
            self.reuslt_count += 1
            return evaluation, False, num
        else:
            if self.episode >= self.training_episodes:
                self.evaluator_done = True
            return [], self.evaluator_done, None
        
# =================== Workers ===================    
@ray.remote
def collecting_worker(model, env, max_episode_steps, hyper_params, memory): 
    # version 1 -- normal method
    update_steps = hyper_params['update_steps']
    steps = 0
    done = False
    rem = 0
    train_done = False
    while True:
        if train_done:
            break
        state = env.reset()
        done = False
        steps = 0 + rem
        a = ray.get(model.explore_or_exploit_policy.remote(state))
        while steps < max_episode_steps and not done:
            steps += 1
            s_, reward, done, info = env.step(a)
            memory.add.remote(state, a, reward, s_, done)
            state = s_
            if steps == max_episode_steps:
                done = True
            a, train_done = ray.get(model.learn.remote(done, state))
    # # version 2 -- simplified method 
    # update_steps = hyper_params['update_steps']
    # steps = 0
    # done = False
    # rem = 0
    # train_done = False
    # while True:
    #     if train_done:
    #         break
    #     state = env.reset()
    #     done = False
    #     steps = 0 + rem
    #     a = ray.get(model.explore_or_exploit_policy.remote(state))
    #     while steps < max_episode_steps and not done:
    #         steps += 1
    #         s_, reward, done, info = env.step(a)
    #         memory.add.remote(state, a, reward, s_, done)
    #         state = s_
    #         if steps == max_episode_steps:
    #             done = True
    #         if steps % update_steps == 0:
    #             a, train_done = ray.get(model.learn.remote(done, True, 0, state))
    #     rem = steps % update_steps
    #     if rem != 0:
    #         a, train_done = ray.get(model.learn.remote(done, False, rem, state))

@ray.remote
def evaluation_worker(model, env, max_episode_steps, trials = 30):
    def greedy_policy(state, eval_model):
        return eval_model.predict(state)

    while True:
        evaluation, done, num = ray.get(model.ask_evaluation.remote())
        if done:
            break
        if not num:
            continue
        total_reward = 0
        for _ in range(trials):
            state = env.reset()
            done = False
            steps = 0

            while steps < max_episode_steps and not done:
                steps += 1
                action = greedy_policy(state, evaluation)
                state, reward, done, _ = env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials
        print(num, " avg_reward: ", avg_reward)
        model.add_result.remote(avg_reward, num)
        
# =================== Agent ===================
class distributed_DQN_agent():
    def __init__(self, env, hyper_params, training_episodes, test_interval, cw_num = 4, ew_num = 4):
        self.memory_server = ReplayBuffer_remote.remote(hyper_params['memory_size'])
        self.model_server = DQNModel_server.remote(env, hyper_params, training_episodes, test_interval, self.memory_server)
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.hyper_params = hyper_params

    def learn_and_evaluate(self):
        workers_id = []

        # learn
        for _ in range(self.cw_num):
            workers_id.append(collecting_worker.remote(self.model_server, self.env, self.max_episode_steps, self.hyper_params, self.memory_server))
        # evaluate
        for _ in range(self.ew_num):
            workers_id.append(evaluation_worker.remote(self.model_server, self.env, self.max_episode_steps))
            
        ray.wait(workers_id, len(workers_id))
        
        return ray.get(self.model_server.get_results.remote())

# =================== Main ===================
env_CartPole.reset()
cw_num = 8 # collector workers
ew_num = 4 # evaluator workers
training_episodes, test_interval = 7000, 50
agent = distributed_DQN_agent(env_CartPole, hyperparams_CartPole, training_episodes, test_interval, cw_num, ew_num)
start_time = time.time()
result = agent.learn_and_evaluate()
result = result[1:-1]
print("running time: ", time.time()-start_time)
print(result)

with open('hw4_result.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(result)
file.close()

plot_result(result, test_interval, ["batch_update with target_model"])