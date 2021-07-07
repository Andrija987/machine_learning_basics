# Hidden Markov Model

#%%
# Importing moduls
import tensorflow as tf
import tensorflow_probability as tfp

#%%
# Modeling
tfd = tfp.distributions # making shortcuts
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])  # inital distribution
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])  # transition distribution
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # observation distribution

#%%
# Creating a model
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

#%%
# Runing a model
mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:  
  print(mean.numpy())

# %%
