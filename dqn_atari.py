#!/usr/bin/env python
"""Run Atari Environment with DQN."""

import argparse
import os
import random

import numpy as np


import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.dqn import DoubleDQNAgent
from deeprl_hw2.dqn import FTDQNAgent
from deeprl_hw2.objectives import mean_huber_loss


import gym
from gym import wrappers


GAMMA = 0.99
ALPHA = 0.00025
EPSILON = 0.05
REPLAY_BUFFER_SIZE = 1000000
BATCH_SIZE = 32
IMG_ROWS , IMG_COLS = 84, 84
WINDOW = 4
TARGET_FREQ = 10000
NUM_BURN_IN = 50000
TRAIN_FREQ=4
#TRAIN_FREQ=1000
MOMENTUM = 0.8
MAX_NUM_ITERATIONS=5000000
ANNEAL_NUM_STEPS = 1000000

DEBUG = 0

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir



def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Enduro')
    parser.add_argument('--env', default='Enduro-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()



    atari_preproc = tfrl.preprocessors.AtariPreprocessor()
    replay_mem = tfrl.core.ReplayMemory(max_size=REPLAY_BUFFER_SIZE, window_length=4)



############# six questions; six experiments ##########################
    ## agent can be DoubleDQNAgent, DQNAgent,FTDQNAgent,DuelingDQNAgent
    agent = 'FTDQNAgent'
    ## network can be LINEAR, DEEP
    network = 'DEEP'
    game = 'Enduro-v0'
    env = gym.make(game)
    eval_env = gym.make(game)
    eval_env = wrappers.Monitor(eval_env,'./video/{0}_{1}'.format(agent,network),force=True)

    if DEBUG:
	    assert env is not None
	    assert eval_env is not None

    dqn_agent = getattr(tfrl.dqn,agent)(network_type = network, \
			    agent = agent, \
                           num_actions = env.action_space.n, \
                           preprocessors = atari_preproc, \
                           memory = replay_mem, \
                           burnin_policy = tfrl.policy.UniformRandomPolicy(num_actions = env.action_space.n),\
                           testing_policy   = tfrl.policy.GreedyEpsilonPolicy(0.05,env.action_space.n), \
                           training_policy    = tfrl.policy.LinearDecayGreedyEpsilonPolicy(env.action_space.n,1.0, 0.1,ANNEAL_NUM_STEPS), \
                           gamma = GAMMA, \
                           alpha = ALPHA, \
                           target_update_freq = TARGET_FREQ, \
                           num_burn_in = NUM_BURN_IN, \
                           train_freq = TRAIN_FREQ, \
                           batch_size = BATCH_SIZE )

    dqn_agent.compile(optimizer='RMSProp', loss_func='mse')
    
    dqn_agent.fit(env, eval_env, num_iterations=MAX_NUM_ITERATIONS, max_episode_length=10000)
    


if __name__ == '__main__':
    main()
