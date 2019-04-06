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
## user defined modules
# kernel rnn cell
import kernl_rnn_cell

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
# Training Parameters
# Setup the Model Parameters
import sys


# Setup the Model Parameters
INPUT_SIZE=784
HIDDEN_SIZE=500
OUTPUT_SIZE = 10
START_LEARNING_RATE=1e-3
BATCH_SIZE=25
NUM_TRAINING_STEPS = 500

EPOCHS=5
TEST_LENGTH=125
DISPLAY_STEP=50
weight_learning_rate=1e-3
tensor_learning_rate = 1e-5
perturbation_std=1e-10
grad_clip=100
TIMESTEP=2
log_dir = "/om/user/ehoseini/MyData/KeRNL/logs/rnn_ffn/kernl_rnn_xaviar_mnist_eta_weight_%1.0e_batch_%1.0e_hum_hidd_%1.0e_steps_%1.0e_run_%s" %(weight_learning_rate,BATCH_SIZE,HIDDEN_SIZE,NUM_TRAINING_STEPS, datetime.now().strftime("%Y%m%d_%H%M"))
log_dir# create a training and testing dataset


## define KeRNL unit
def kernl_rnn(x,kernel_weights,kernel_bias,initial_state):
    # Define a KeRNL cell, the initialization is done inside the cell with default initializers
    with tf.variable_scope('kernl') as scope:
        kernl_rnn_unit = kernl_rnn_cell.kernl_rnn_cell(num_units=HIDDEN_SIZE,
                                                      num_inputs=INPUT_SIZE,
                                                      time_steps=TIMESTEP,
                                                      noise_param=perturbation_std,
                                                      sensitivity_initializer=tf.initializers.identity()
                                                      ,activation="relu",
                                                       kernel_initializer=tf.contrib.layers.xavier_initializer()
                                                      )
        # Get KeRNL cell output
        #initial_state = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
        kernel_outputs, kernel_states = tf.nn.dynamic_rnn(kernl_rnn_unit, inputs=x, dtype=tf.float32,time_major=False,initial_state=initial_state)
        kernl_rnn_output=tf.matmul(kernel_outputs[:,-1,:], kernel_weights) + kernel_bias
    return kernl_rnn_output, kernel_states


tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():
    initializer = tf.random_normal_initializer(stddev=0.1)
    X = tf.placeholder("float", [BATCH_SIZE, TIMESTEP, INPUT_SIZE])
    Y = tf.placeholder("float", [BATCH_SIZE, OUTPUT_SIZE])

    iniitial_state_h=tf.zeros([BATCH_SIZE,HIDDEN_SIZE])
    iniitial_state_h_hat=tf.zeros([BATCH_SIZE,HIDDEN_SIZE])
    iniitial_state_theta=tf.zeros([BATCH_SIZE,HIDDEN_SIZE])
    iniitial_state_Gamma=tf.zeros([BATCH_SIZE,HIDDEN_SIZE])
    iniitial_state_eligibility=tf.zeros([BATCH_SIZE,HIDDEN_SIZE,INPUT_SIZE+HIDDEN_SIZE])
    bias_trace=tf.zeros([BATCH_SIZE,HIDDEN_SIZE])
    initial_state = kernl_rnn_cell.kernl_state_tuple(iniitial_state_h,iniitial_state_h_hat, iniitial_state_theta,iniitial_state_Gamma,iniitial_state_eligibility,bias_trace)
    with tf.variable_scope('kernl_output') as scope:
        kernel_weights = tf.get_variable(shape=[HIDDEN_SIZE, OUTPUT_SIZE],name='output_weight',initializer=initializer)
        kernel_bias = tf.get_variable(shape=[OUTPUT_SIZE],name='output_addition',initializer=initializer)
    kernl_output,kernl_states=kernl_rnn(X,kernel_weights,kernel_bias,initial_state)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(kernl_output, 1))
    kernl_accuracy = 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    trainables=tf.trainable_variables()
    variable_names=[v.name for v in tf.trainable_variables()]
    find_joing_index = lambda x, name_1,name_2 : [a and b for a,b in zip([np.unicode_.find(k.name, name_1)>-1 for k in x] ,[np.unicode_.find(k.name, name_2)>-1 for k in x])].index(True)
    with tf.name_scope('kernl_Trainables') as scope:
        kernl_output_weight_index= find_joing_index(trainables,'kernl','output_weight')
        kernl_output_addition_index= find_joing_index(trainables,'kernl','output_addition')
        kernl_temporal_filter_index= find_joing_index(trainables,'kernl','temporal_filter')
        kernl_sensitivity_tensor_index= find_joing_index(trainables,'kernl','sensitivity_tensor')
        kernl_kernel_index= find_joing_index(trainables,'kernl','kernel')
        kernl_bias_index= find_joing_index(trainables,'kernl','bias')
    #
        kernl_tensor_training_indices=np.asarray([kernl_sensitivity_tensor_index,kernl_temporal_filter_index],dtype=np.int)
        kernl_tensor_trainables= [trainables[k] for k in kernl_tensor_training_indices]
    #
        kernl_weight_training_indices=np.asarray([kernl_kernel_index,kernl_bias_index,kernl_output_weight_index,kernl_output_addition_index],dtype=np.int)
        kernl_weight_trainables= [trainables[k] for k in kernl_weight_training_indices]

    with tf.name_scope("kernl_train") as scope:
        # outputs
        kernl_loss_output_prediction=tf.losses.mean_squared_error(Y,kernl_output)
        kernl_loss_state_prediction=tf.losses.mean_squared_error(tf.subtract(kernl_states.h_hat, kernl_states.h),tf.matmul(kernl_states.Theta,trainables[kernl_sensitivity_tensor_index]))
        # define optimizers
        kernl_weight_optimizer = tf.train.RMSPropOptimizer(learning_rate=weight_learning_rate)
        kernl_tensor_optimizer = tf.train.RMSPropOptimizer(learning_rate=tensor_learning_rate)

    with tf.name_scope('kernl_train_tensors') as scope:
            kernl_delta_sensitivity=tf.subtract(tf.matmul(kernl_states.Theta,tf.transpose(trainables[kernl_sensitivity_tensor_index])),tf.subtract(kernl_states.h_hat,kernl_states.h))
            kernl_sensitivity_tensor_update= tf.reduce_mean(tf.einsum("un,uv->unv",kernl_delta_sensitivity,kernl_states.Theta),axis=0)
            kernl_temporal_filter_update= tf.reduce_mean(tf.multiply(tf.matmul(kernl_delta_sensitivity,trainables[kernl_sensitivity_tensor_index]),kernl_states.Gamma),axis=0)
            kernl_tensor_grads_and_vars=list(zip([kernl_sensitivity_tensor_update,kernl_temporal_filter_update],kernl_tensor_trainables))
            kernl_cropped_tensor_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) if  np.unicode_.find(var.name,'output')==-1 else (grad,var) for grad,var in kernl_tensor_grads_and_vars]
            kernl_tensor_train_op = kernl_tensor_optimizer.apply_gradients(kernl_cropped_tensor_grads_and_vars)

    with tf.name_scope('kernl_train_weights') as scope:
            kernl_grad_cost_to_output=tf.gradients(kernl_loss_output_prediction,kernl_output, name= 'kernl_grad_cost_to_y')
            kernl_error_in_hidden_state=tf.matmul(kernl_grad_cost_to_output[-1],tf.transpose(trainables[kernl_output_weight_index]))
            kernl_delta_weight=tf.matmul(kernl_error_in_hidden_state,trainables[kernl_sensitivity_tensor_index])
            kernl_weight_update=tf.transpose(tf.reduce_mean(tf.einsum("un,unv->unv",kernl_delta_weight,kernl_states.eligibility_trace),axis=0))
            kernl_bias_update=tf.reduce_mean(tf.multiply(kernl_delta_weight,kernl_states.bias_trace),axis=0)
            # output layer
            kernl_grad_cost_to_output_layer=tf.gradients(kernl_loss_output_prediction,[trainables[kernl_output_weight_index],trainables[kernl_output_addition_index]], name= 'kernl_grad_cost_to_output_layer')
            # crop the gradients
            kernl_weight_grads_and_vars=list(zip([kernl_weight_update,kernl_bias_update,kernl_grad_cost_to_output_layer[0],kernl_grad_cost_to_output_layer[1]],kernl_weight_trainables))
            #kernl_cropped_weight_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) if  np.unicode_.find(var.name,'output')==-1 else (grad,var) for grad,var in kernl_weight_grads_and_vars]
            kernl_cropped_weight_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) for grad,var in kernl_weight_grads_and_vars]
            # apply gradients
            kernl_weight_train_op = kernl_weight_optimizer.apply_gradients(kernl_cropped_weight_grads_and_vars)
    #
    tf.summary.scalar('bp_rnn_loss',kernl_loss_output_prediction+1e-10)
    tf.summary.scalar('bp_rnn_accuracy',kernl_accuracy+1e-10)
    summary_op=tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
###################################################

Path(log_dir).mkdir(exist_ok=True, parents=True)
filelist = [ f for f in os.listdir(log_dir) if f.endswith(".local") ]
for f in filelist:
    os.remove(os.path.join(log_dir, f))
####################################################

# write graph into tensorboard
tb_writer = tf.summary.FileWriter(log_dir,graph)
# run a training session
tb_writer = tf.summary.FileWriter(log_dir,graph)
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(EPOCHS):
        for step in range(NUM_TRAINING_STEPS):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            batch_x_expand=np.expand_dims(batch_x,axis=1)
            batch_x_fixed=np.concatenate([batch_x_expand,0*batch_x_expand],axis=1)
            kernl_state_train, kernl_tensor_loss=sess.run([kernl_tensor_train_op,kernl_loss_state_prediction], {X: batch_x_fixed, Y: batch_y})
            kernl_weight_train, kernl_loss,accu=sess.run([kernl_weight_train_op,kernl_loss_output_prediction,kernl_accuracy], {X: batch_x_fixed, Y: batch_y})
            kernl_summary=sess.run(summary_op, {X: batch_x_fixed, Y: batch_y})
            tb_writer.add_summary(kernl_summary, global_step=epoch*NUM_TRAINING_STEPS+step+1)
            if step % DISPLAY_STEP==0 :
                print('Epoch: {}, Batch: {}, tensor Loss: {:.3f}, loss {:.3f}, accuracy : {:.3f}'.format(epoch+1,step + 1, kernl_tensor_loss,kernl_loss,accu))

    print("Optimization Finished!")
