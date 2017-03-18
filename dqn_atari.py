#!/usr/bin/env python
"""Run Atari Environment with DQN."""




import argparse
import os
import random

import numpy as np

import tensorflow as tf

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam


#import skimage as skimage
#from skimage import transform, color, exposure
#from skimage.transform import rotate
#from skimage.viewer import ImageViewer

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss

import gym


def create_model(window, input_shape, num_actions, model_name='q_network'):  # noqa: D103
    
    """Create the Q-network model.
        
        Use Keras to construct a keras.models.Model instance (you can also
        use the SequentialModel class).
        
        We highly recommend that you use tf.name_scope as discussed in
        class when creating the model and the layers. This will make it
        far easier to understnad your network architecture if you are
        logging with tensorboard.
        
        Parameters
        ----------
        window: int
        Each input to the network is a sequence of frames. This value
        defines how many frames are in the sequence.
        input_shape: tuple(int, int)
        The expected input image size.
        num_actions: int
        Number of possible actions. Defined by the gym environment.
        model_name: str
        Useful when debugging. Makes the model show up nicer in tensorboard.
        
        Returns
        -------
        keras.models.Model
        The Q-model.
        """
    
        model = Sequential()
        
        
        #TODO: do we need this???See
        #http://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras
        #model.add(BatchNormalization())
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(window,input_shape[0],input_shape[1])))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Dense(num_actions))
        
        adam = Adam(lr=LEARNING_RATE)
        
        #change the loss in order to have two networks
        model.compile(loss='mse',optimizer=adam)
        
        return model
        
        pass
    



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

    env = gym.make('Enduro-v0')
    #q_network=create_model(WINDOW,tuple(IMG_ROWS,IMG_COLS),env.action_space.n)

    #sess = tf.Session()
    #from keras import backend as K
    #K.set_session(sess)

    initial_state = env.reset()
    env.render()
    rewards = []
    num_steps = 0

    while True:
	    action = env.action_space.sample()
	    print env.action_space
	    nextstate, reward, is_terminal, debug_info = env.step(action)
	    print nextstate.shape
	    env.render()
	    rewards.append(reward)
	    state = nextstate
	    num_steps += 1

	    if is_terminal:
		break

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    





if __name__ == '__main__':
    main()
