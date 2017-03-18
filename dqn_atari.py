#!/usr/bin/env python
"""Run Atari Environment with DQN."""

import argparse
import os
import random

import numpy as np

#import tensorflow as tf
#
#from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
#                          Permute)
#from keras.models import Model
#from keras.optimizers import Adam


#import skimage as skimage
#from skimage import transform, color, exposure
#from skimage.transform import rotate
#from skimage.viewer import ImageViewer

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss

import gym


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
    #TODO: input_shape as argument?
    #args.input_shape = tuple(args.input_shape)

    #args.output = get_output_folder(args.output, args.env)

    atari_preproc = tfrl.preprocessors.AtariPreprocessor(new_size=(84,84))
    history_preproc = tfrl.preprocessors.HistoryPreprocessor(history_length = 4)
    preproc = tfrl.preprocessors.PreprocessorSequence([atari_preproc, history_preproc])

    replay_mem = tfrl.core.ReplayMemory(max_size=1e6, window_length=4)

    policy_unirand = tfrl.policy.UniformRandomPolicy(num_actions = 9)
    policy_greedy = tfrl.policy.GreedyPolicy()
    policy_epsilon = tfrl.policy.GreedyEpsilonPolicy(epsilon=0.05)

    dqn_agent = DQNAgent(model_name 	    = 'LINEAR', \
			 preprocessors      = [atari_preproc, preproc], \
			 memory 	    = replay_mem, \
			 policy		    = policy_epsilon, \
			 gamma		    = 0.9, \
			 target_update_freq = 1000, \
			 num_burn_in 	    = 1000, \
			 train_freq 	    = 4, \
			 batch_size 	    = 32 )
    dqn_agent.compile(optimizer='Adam', loss_func='mse')
   
    env = gym.make('Enduro-v0')

    for episode in range(1000):
    	initial_state = env.reset()
	atari_preproc.reset()
	history_preproc.reset()
	replay_mem.clear()

	dqn_agent.fit(env, num_iterations=10000)
    


if __name__ == '__main__':
    main()
