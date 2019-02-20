# python libraries
import numpy as np
import matplotlib.pyplot as plt
import collections
import hashlib
import numbers
import matplotlib.cm as cm
from sys import getsizeof
from datetime import datetime
from pathlib import Path
import os
from IPython.display import HTML
import re

# tensorflow and its dependencies
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow.contrib import slim

## user defined modules
# kernel rnn cell
import kernl_spiking_cell_v3

# uplading mnist data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
tf.logging.set_verbosity(old_v)

# Training Parameters
weight_learning_rate = 1e-5
tensor_learning_rate=1e-3
training_steps = 50000
batch_size = 25
display_step = 25
test_len=128
grad_clip=200
# Network Parameters
# 1-input layer
num_input = 1 # MNIST data input (img shape: 28*28)
num_context_input=1
MNIST_timesteps = 28*28 # timesteps
context_timesteps=54
timesteps=MNIST_timesteps+context_timesteps
num_unit_input_layer=80 # input layer neurons
num_context_unit=1
noise_std=0.5
# 2-hidden layer
num_hidden = 200 # hidden layer num of features
# 3-output layer
num_classes = 10 # MNIST total classes (0-9 digits)

# report batch number
total_batch = int(mnist.train.num_examples / batch_size)
print("Total number of batches:", total_batch)

def kernl_SNN_all_states(x,context):
    with tf.variable_scope('context_layer') as scope:
        context_input_layer_cell=kernl_spiking_cell_v3.context_input_spike_cell(num_units=1,context_switch=MNIST_timesteps)
        context_initial_state = context_input_layer_cell.zero_state(batch_size, dtype=tf.float32)
        output_context, states_context = tf.nn.dynamic_rnn(context_input_layer_cell, dtype=tf.float32, inputs=context,initial_state=context_initial_state)

    with tf.variable_scope('input_layer') as scope:
        input_layer_cell=kernl_spiking_cell_v3.input_spike_cell(num_units=num_unit_input_layer)
        input_initial_state = input_layer_cell.zero_state(batch_size, dtype=tf.float32)
        output_l1, states_l1 = tf.nn.dynamic_rnn(input_layer_cell, dtype=tf.float32, inputs=x,initial_state=input_initial_state)

    with tf.variable_scope('hidden_layer') as scope:
        hidden_layer_cell=kernl_spiking_cell_v3.kernl_spike_Cell(num_units=num_hidden,
                                                                 num_inputs=num_unit_input_layer+num_context_unit,
                                                                 time_steps=timesteps,
                                                                 output_is_tuple=True,
                                                                 tau_refract=1.0,
                                                                 tau_m=20,
                                                                 noise_std=noise_std)
        hidden_initial_state = hidden_layer_cell.zero_state(batch_size, dtype=tf.float32)
        output_hidden, states_hidden = tf.nn.dynamic_rnn(hidden_layer_cell, dtype=tf.float32, inputs=tf.concat([output_l1,output_context],-1),initial_state=hidden_initial_state)
    with tf.variable_scope('output_layer') as scope :
        output_layer_cell=kernl_spiking_cell_v3.output_spike_cell(num_units=num_classes)
        output_voltage, voltage_states=tf.nn.dynamic_rnn(output_layer_cell,dtype=tf.float32,inputs=output_hidden.spike)

    return output_voltage,output_hidden

tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():
    # check hardware

    # define weights and inputs to the network
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    Context=tf.placeholder('float',shape=[batch_size,timesteps,num_context_input])
    # define a function for extraction of variable names
    kernl_output,kernl_hidden_states=kernl_SNN_all_states(X,Context)

    trainables=tf.trainable_variables()
    variable_names=[v.name for v in tf.trainable_variables()]
    #
    find_joing_index = lambda x, name_1,name_2 : [a and b for a,b in zip([np.unicode_.find(k.name, name_1)>-1 for k in x] ,[np.unicode_.find(k.name, name_2)>-1 for k in x])].index(True)
    # find trainable parameters for kernl
    with tf.name_scope('kernl_Trainables') as scope:
        kernl_output_weight_index= find_joing_index(trainables,'output_layer','kernel')
        kernl_temporal_filter_index= find_joing_index(trainables,'kernl','temporal_filter')
        kernl_sensitivity_tensor_index= find_joing_index(trainables,'kernl','sensitivity_tensor')
        kernl_kernel_index= find_joing_index(trainables,'hidden_layer','kernel')
    #
        kernl_tensor_training_indices=np.asarray([kernl_sensitivity_tensor_index,kernl_temporal_filter_index],dtype=np.int)
        kernl_tensor_trainables= [trainables[k] for k in kernl_tensor_training_indices]
    #
        kernl_weight_training_indices=np.asarray([kernl_kernel_index,kernl_output_weight_index],dtype=np.int)
        kernl_weight_trainables= [trainables[k] for k in kernl_weight_training_indices]


    ##################
    # kernl train ####
    ##################
    with tf.name_scope("kernl_performance") as scope:
        # outputs
        kernl_logit=tf.reduce_mean(kernl_output[:,-context_timesteps:,:],axis=1)
        kernl_loss_output_prediction=tf.losses.softmax_cross_entropy(onehot_labels=Y,logits=kernl_logit)
        kernl_prediction = tf.nn.softmax(kernl_logit)
        kernl_correct_pred = tf.equal(tf.argmax(kernl_prediction, 1), tf.argmax(Y, 1))
        kernl_accuracy = tf.reduce_mean(tf.cast(kernl_correct_pred, tf.float32))

    with tf.name_scope('kernl_train_tensors') as scope:
        kernl_loss_state_prediction=tf.losses.mean_squared_error(tf.subtract(kernl_hidden_states.v_mem_hat[:,-1,:], kernl_hidden_states.v_mem[:,-1,:]),tf.matmul(kernl_hidden_states.Theta[:,-1,:],trainables[kernl_sensitivity_tensor_index]))
        kernl_tensor_optimizer = tf.train.RMSPropOptimizer(learning_rate=tensor_learning_rate)
        kernl_tensor_grads=tf.gradients(ys=kernl_loss_state_prediction,xs=kernl_tensor_trainables)
        kernl_tensor_grad_and_vars=list(zip(kernl_tensor_grads,kernl_tensor_trainables))
        kernl_tensor_train_op=kernl_tensor_optimizer.apply_gradients(kernl_tensor_grad_and_vars)


    ##################
    # SUMMARIES ######
    ##################

    with tf.name_scope("kernl_tensor_summaries") as scope:
        # kernl sensitivity tensor
        tf.summary.histogram('kernl_sensitivity_tensor_grad',kernl_tensor_grads[0]+1e-10)
        tf.summary.histogram('kernl_sensitivity_tensor',trainables[kernl_sensitivity_tensor_index]+1e-10)
        # kernl temporal filter
        tf.summary.histogram('kernl_temporal_filter_grad',kernl_tensor_grads[1]+1e-10)
        tf.summary.histogram('kernl_temporal_filter',trainables[kernl_temporal_filter_index]+1e-10)
        # kernl loss
        tf.summary.scalar('kernl_loss_state_prediction',kernl_loss_state_prediction+1e-10)
        # kernl senstivity tensor and temporal filter
        tf.summary.image('kernl_sensitivity_tensor',tf.expand_dims(tf.expand_dims(trainables[kernl_sensitivity_tensor_index],axis=0),axis=-1))
        tf.summary.image('kernl_sensitivity_tensor_grad',tf.expand_dims(tf.expand_dims(kernl_tensor_grads[0],axis=0),axis=-1))
        tf.summary.image('kernl_temporal_filter',tf.expand_dims(tf.expand_dims(tf.expand_dims(trainables[kernl_temporal_filter_index],axis=0),axis=-1),axis=-1))
        tf.summary.image('kernl_temporal_filter_grad',tf.expand_dims(tf.expand_dims(tf.expand_dims(kernl_tensor_grads[1],axis=0),axis=-1),axis=-1))
        kernl_tensor_merged_summary_op=tf.summary.merge_all(scope="kernl_tensor_summaries")

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

log_dir = "/om/user/ehoseini/MyData/KeRNL/logs/kernl_SNN_v3/MNIST_gc_%d_eta_m_%d_eta_%d_batch_%d_run_%s" %(grad_clip,tensor_learning_rate,weight_learning_rate,batch_size, datetime.now().strftime("%Y%m%d_%H%M"))
Path(log_dir).mkdir(exist_ok=True, parents=True)
filelist = [ f for f in os.listdir(log_dir) if f.endswith(".local") ]
for f in filelist:
    os.remove(os.path.join(log_dir, f))

# write graph into tensorboard
tb_writer = tf.summary.FileWriter(log_dir,graph)
# run a training session
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for step in range(1,training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape((batch_size,MNIST_timesteps,num_input))
        batch_x_full=np.concatenate([batch_x,np.zeros((batch_size,timesteps-MNIST_timesteps,num_input))],axis=1)
        context_input=np.ones((batch_size,timesteps,num_context_input))
        kernl_tensor_train,kernl_loss_state=sess.run([kernl_tensor_train_op,kernl_loss_state_prediction], feed_dict={X: batch_x_full,Y:batch_y,Context:context_input})

        # run summaries
        kernl_tensor_merged_summary=sess.run(kernl_tensor_merged_summary_op,feed_dict={X:batch_x_full, Y:batch_y,Context:context_input})

        tb_writer.add_summary(kernl_tensor_merged_summary, global_step=step)
        #
        if step % display_step==0 or step==1 : 
            # get batch loss and accuracy
            print('Step: {}, keRNL tensor Loss {:.3f},'.format(step + 1, kernl_loss_state))


    print("Optimization Finished!")
    #test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    #test_label = mnist.test.labels[:test_len]
    #print("Testing Accuracy:",
    #    sess.run(loss_output_prediction, feed_dict={X: test_data, Y: test_label}))
    save_path = saver.save(sess, log_dir+"/model.ckpt", global_step=step,write_meta_graph=True)
    print("Model saved in path: %s" % save_path)
