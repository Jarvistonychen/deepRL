"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor

import copy

class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=4):
        self.hist_len = history_length
	# keras/TF requires input to be 4-dimenional: batch, channel, row ,col
	self.state_seq_mem = np.zeros((1,self.hist_len, 84,84), dtype=np.uint8)
	self.state_seq = np.zeros((1,self.hist_len, 84,84))

    # state here is already processed in AtariPreprocessor
    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
	for hist in range(self.hist_len-1):
		self.state_seq[0,hist,:,:] = self.state_seq[0,hist+1,:,:]
	self.state_seq[0,self.hist_len-1, :,:] = state
	temp_state_seq = np.copy(self.state_seq)
	return temp_state_seq
		
    # state here is already processed in AtariPreprocessor
    def process_state_for_memory(self, state):
        """You only want history when you're deciding the current action to take."""
	for hist in range(self.hist_len-1):
		self.state_seq_mem[0,hist,:,:] = self.state_seq_mem[0,hist+1,:,:]
	self.state_seq_mem[0,self.hist_len-1, :,:] = state
	temp_state_seq_mem = np.copy(self.state_seq_mem)
	return temp_state_seq_mem

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
	self.state_seq_mem = np.zeros((1,self.hist_len, 84,84), dtype=np.uint8)
	self.state_seq = np.zeros((1,self.hist_len, 84,84))
	

    def get_config(self):
        return {'history_length': self.hist_len}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size=(84,84)):
	self.new_size = new_size
	self.old_state = np.zeros((1,84,84))

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
	state_temp = np.copy(state)
	state_temp = np.amax(state_temp, axis=2)
	#print 'Atari image '
	#print state_temp
	#print state_temp.shape
	img = Image.fromarray(state_temp)
	#img.save('ori.bmp')
	img = img.resize(self.new_size, Image.ANTIALIAS)
	state_temp = np.array(img, dtype=np.uint8)
	#img = Image.fromarray(state_temp)
	#img.save('resize.bmp')
	#print 'Atari image for memory'
	#print state_temp
	#print state_temp.shape
	return state_temp

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
	state_temp = np.copy(state)
	state_temp = np.amax(state_temp, axis=2)
	img = Image.fromarray(state_temp)
	img = img.resize(self.new_size, Image.ANTIALIAS)
	state_temp = np.array(img)
	return state_temp
	

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
	samples_temp = copy.deepcopy(samples)
	#print 'before process_batch samples[0].state {0}'.format(samples[0].state)
	#print 'before process_batch samples_temp[0].state {0}'.format(samples_temp[0].state)
	for item in samples_temp:
		item.state = item.state.astype(np.float)
		item.next_state = item.next_state.astype(np.float)
	#print 'after process_batch samples[0].state {0}'.format(samples[0].state)
	#print 'after process_batch samples_temp[0].state {0}'.format(samples_temp[0].state)
	return samples_temp

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
	if reward > 0:
		return 1
	elif reward < 0:
		return -1
	else:
		return 0


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, preprocessors):
	self.atari = preprocessors[0]
	self.history = preprocessors[1]

    def get_history_for_memory(self, state):
	state = self.atari.process_state_for_memory(state)
	return self.history.process_state_for_memory(state)

    def get_history_for_network(self, state):
	state = self.atari.process_state_for_network(state)
	return self.history.process_state_for_network(state)
