"""Main DQN agent."""

from keras import optimizers
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input,
                          Multiply)
from keras.models import Model
import numpy as np
import pickle
import matplotlib.pyplot as plt

import deeprl_hw2 as tfrl

GAMMA = 0.9
ALPHA = 1e-4
EPSILON = 0.05
REPLAY_BUFFER_SIZE = 1000000
BATCH_SIZE = 32
IMG_ROWS , IMG_COLS = 84, 84
WINDOW = 4
TARGET_QNET_RESET_INTERVAL = 10000
SAMPLES_BURN_IN = 10000
TRAINING_FREQUENCY=4
LEARNING_RATE = 0.01

class DQNAgent:
    
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
		 model_name,
                 preprocessors,
                 memory,
                 policy,
                 gamma=GAMMA,
                 target_update_freq=TARGET_QNET_RESET_INTERVAL,
                 num_burn_in=SAMPLES_BURN_IN,
                 train_freq=TRAINING_FREQUENCY,
                 batch_size=BATCH_SIZE):


	self.model_name = model_name
	self.atari_proc  	= preprocessors[0]
	self.preproc     	= preprocessors[1]
        self.q_network   	= self.create_model(window = WINDOW, \
						    input_shape = (IMG_ROWS, IMG_COLS), \
						    num_actions = 9, \
						    model_name	= self.model_name)
        self.qt_network  	= self.create_model(window = WINDOW, \
						    input_shape = (IMG_ROWS, IMG_COLS), \
						    num_actions = 9, \
						    model_name	= self.model_name)
	print 'model summary'
	print self.q_network.summary()
	print self.q_network.get_config()

        self.memory	 	= memory
        self.policy	 	= policy
        self.gamma	 	= gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in 	= num_burn_in
        self.train_freq		= train_freq
        self.batch_size		= batch_size
	self.num_update 	= 0

	self.train_loss		= []
	self.mean_q		= []
	self.rand_states 	= np.random.randint(255, size=(10000, 4, 84, 84))
	self.rand_states_mask 	= np.ones((10000, 9))

    def create_model(self, window, input_shape, num_actions, model_name='DQN2layer'):  # noqa: D103
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

	if model_name == 'DQN2layer':
		#model = Sequential()
		#
		#model.add(Conv2D(16, (8, 8), strides=4, padding='same',use_bias=True,input_shape=(window,input_shape[0],input_shape[1]), data_format='channels_first'))
		#model.add(Activation('relu'))
		#        
		#model.add(Conv2D(32, (4, 4), strides=2, padding='same',use_bias=True,data_format='channels_first'))
		#model.add(Activation('relu'))
		#                
		#model.add(Flatten())
		#model.add(Dense(256))
		#model.add(Activation('relu'))
		#                        
		#model.add(Dense(num_actions))
		##model.add(Multiply())
		#model.add(Activation('softmax'))

		a1 = Input(shape=(window,input_shape[0],input_shape[1]))
		a2 = Input(shape=(num_actions,))
		b = Conv2D(16, (8, 8), strides=4, padding='same',use_bias=True,activation='relu', data_format='channels_first')(a1)
			
		c = Conv2D(32, (4, 4), strides=2, padding='same',use_bias=True,activation='relu', data_format='channels_first')(b)
		d = Flatten()(c)
		e = Dense(256, activation='relu')(d)
		f = Dense(num_actions)(e)
		h = Multiply()([f,a2])
		model = Model(inputs=[a1,a2], outputs=[h])  
		return model

	elif model_name == 'LINEAR':
		a1 = Input(shape=(window,input_shape[0],input_shape[1]))
		a2 = Input(shape=(num_actions,))
		b = Flatten()(a1)
		c = Dense(num_actions)(b)
		e = Multiply()([c,a2])
		model = Model(inputs=[a1,a2], outputs=[e])  
		return model
		
    

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """

	if optimizer == 'Adam':
        	opti = optimizers.Adam(lr=LEARNING_RATE)
	self.q_network.compile(loss=loss_func, optimizer = opti)

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
	return self.q_network.predict(state, batch_size = 1)

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
	return self.policy.select_action(self.calc_q_values(state))

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
	if self.num_update % self.target_update_freq == 0:
		print "======================= Sync target and source network ============================="
		tfrl.utils.get_hard_target_model_updates(self.qt_network, self.q_network)
		#get_soft_target_model_updates(self.qt_network, self.q_network)

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
	input_dummymask = np.ones((1,9))
 	for step in range(num_iterations):
            if step > 0:
                state_history = nextstate_history
                action = self.select_action([state_history, input_dummymask])
            else:
                action = env.action_space.sample()
            nextstate, reward, is_terminal, debug_info = env.step(action)
            nextstate_history = self.preproc.get_history(nextstate)
            if step > 0:
                self.memory.append(state_history, \
				   action, \
				   self.atari_proc.process_reward(reward), \
				   nextstate_history, \
				   is_terminal)
	    # train q_network
	    if self.num_update >= self.num_burn_in:
		if self.num_update == self.num_burn_in:
			print '================ Memory burn in ({0}) finished ==========='.format(self.num_burn_in)

		if self.num_update % self.train_freq == 0:
			#print 'Memory burn in phase completed. {0} samples'.format(self.num_burn_in)
			# generate batch samples for CNN
			mem_samples = self.memory.sample(self.batch_size)
			mem_samples = self.atari_proc.process_batch(mem_samples)
			input_state_batch=np.zeros((self.batch_size, 4, 84, 84))
			input_nextstate_batch=np.zeros((self.batch_size, 4, 84, 84))
			input_mask_batch=np.zeros((self.batch_size,9))
			input_dummymask_batch=np.ones((self.batch_size,9))
			output_target_batch=np.zeros((self.batch_size,9))
			for ind in range(self.batch_size):
				input_state_batch[ind,:,:,:] = mem_samples[ind].state
				input_nextstate_batch[ind,:,:,:] = mem_samples[ind].next_state
				input_mask_batch[ind, mem_samples[ind].action] = 1

			target_q = self.calc_q_values([input_nextstate_batch, input_dummymask_batch])
			#print target_q
			best_target_q = np.amax(target_q, axis=1)
			#print 'best Q values of batch'
			#print best_target_q

			for ind in range(self.batch_size):
				output_target_batch[ind, mem_samples[ind].action] = mem_samples[ind].reward + self.gamma*best_target_q[ind]

			self.q_network.fit(x=[input_state_batch, input_mask_batch], y=output_target_batch, batch_size=32, epochs=1)
		
	    	if  self.num_update % (self.train_freq * 25) == 0:
			self.mean_q.append(self.eval_avg_q())
			self.train_loss.append(self.q_network.evaluate(x=[input_state_batch, input_mask_batch], y=output_target_batch))

		self.save_data(freq=self.target_update_freq)

            #env.render()
            state = nextstate
            self.num_update += 1
	    self.update_policy()

            if is_terminal:
                break
    def save_data(self,freq):
	if self.num_update % freq == 0:
		plt.plot(self.mean_q)
		plt.savefig('mean_q_{0}.jpg'.format(self.model_name))
		plt.close()
		plt.plot(self.train_loss)
		plt.savefig('train_loss_{0}.jpg'.format(self.model_name))
		plt.close()
		with open('mean_q_{0}.data'.format(self.model_name),'w') as f:
			pickle.dump(self.mean_q,f)
		with open('train_loss_{0}.data'.format(self.model_name),'w') as f:
			pickle.dump(self.train_loss,f)
		self.q_network.save_weights('source_{0}.weight'.format(self.model_name))
		self.qt_network.save_weights('target_{0}.weight'.format(self.model_name))

    def eval_avg_q(self):
	return np.mean(np.amax(self.calc_q_values([self.rand_states, self.rand_states_mask]), axis=1))

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        pass
