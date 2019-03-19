from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

import collections
import hashlib
import numbers
from datetime import datetime
from pathlib import Path
import os

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

import long_short_term_spike_cell_bare as spiking_cell

def _lsnn_weight_initializer(shape,dtype=None,partition_info=None,verify_shape=None, gain=1.0):
    if dtype is None:
        dtype=tf.float32
    #extract second dimension
    W_rec=tf.divide(gain*tf.random_normal(shape=[shape[1],shape[1]],mean=0,stddev=1,dtype=dtype),tf.sqrt(tf.cast(shape[1],tf.float32)))
    new_shape=[shape[0]-shape[1],shape[1]]

    W_in = tf.divide(gain*tf.random_normal(shape=new_shape,mean=0,stddev=1,dtype=dtype),tf.sqrt(tf.cast(new_shape[0],tf.float32)))
    #W_in=tf.random_normal(new_shape,mean=0,stddev=0.001)
    return tf.concat([W_in,W_rec],axis=0)

weight_learning_rate = 1e-4
training_steps = 1000
batch_size = 200
display_step = 5
test_len=100
epochs=10
grad_clip=200
buffer_size=400
# Network Parameters
num_input = 1 # MNIST data input (img shape: 28*28)
num_context_input=1
MNIST_size=28
MNIST_timesteps = MNIST_size*MNIST_size # timesteps
context_timesteps=54
timesteps=MNIST_timesteps+context_timesteps
#
num_unit_input_layer=80 # input layer neurons
num_context_unit=1
num_hidden = 220 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)
# uplading mnist data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train.astype(np.int64)
y_test.astype(np.int64)
train_context=np.ones((x_train.shape[0],timesteps,num_context_input))
test_context=np.ones((x_test.shape[0],timesteps,num_context_input))
# fix data for snn problem
x_train_temp=x_train.reshape((-1,MNIST_timesteps,num_input))
x_train=np.concatenate([x_train_temp,np.zeros((x_train_temp.shape[0],timesteps-MNIST_timesteps,num_input))],axis=1)

x_test_temp=x_test.reshape((-1,MNIST_timesteps,num_input))
x_test=np.concatenate([x_test_temp,np.zeros((x_test_temp.shape[0],timesteps-MNIST_timesteps,num_input))],axis=1)

## define KeRNL unit
def bptt_snn_all_states(x,context):
    with tf.variable_scope('input_context') as scope:
        context_input_layer_cell=spiking_cell.context_input_spike_cell(num_units=1,context_switch=MNIST_timesteps)
        output_context, states_context = tf.nn.dynamic_rnn(context_input_layer_cell, dtype=tf.float32, inputs=context)
    with tf.variable_scope('input_layer') as scope:
        input_layer_cell=spiking_cell.input_spike_cell(num_units=num_unit_input_layer)
        output_l1, states_l1 = tf.nn.dynamic_rnn(input_layer_cell, dtype=tf.float32, inputs=x)
    with tf.variable_scope('hidden_layer') as scope:
        hidden_layer_cell=spiking_cell.long_short_term_spike_cell(num_units=num_hidden,
                                                                  num_inputs=num_unit_input_layer+num_context_unit,
                                                                  state_is_tuple=True,
                                                                  output_is_tuple=False,
                                                                  kernel_initializer=_lsnn_weight_initializer)

        output_hidden, states_hidden = tf.nn.dynamic_rnn(hidden_layer_cell, dtype=tf.float32, inputs=tf.concat([output_l1,output_context],-1))
    with tf.variable_scope('output_layer') as scope :
        output_layer_cell=spiking_cell.output_spike_cell(num_units=num_classes,tau_m=20.0,kernel_initializer=tf.initializers.random_uniform)
        output_voltage, voltage_states=tf.nn.dynamic_rnn(output_layer_cell,dtype=tf.float32,inputs=output_hidden)
    return output_voltage,output_hidden

tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():
    BATCH_SIZE=tf.placeholder(tf.int64)
    X = tf.placeholder("float", [None, timesteps, num_input],name='mnist_input')
    Y = tf.placeholder("int32", [None],name='mnist_labels')
    Context=tf.placeholder('float',[None,timesteps,num_context_input],name='context_inputs')
    # define a dataset
    dataset=tf.data.Dataset.from_tensor_slices((X,Y,Context)).batch(BATCH_SIZE).repeat()
    dataset = dataset.shuffle(buffer_size=buffer_size)
    iter = dataset.make_initializable_iterator()
    inputs,labels,contexts =iter.get_next()
    labels_temp=tf.one_hot(labels,depth=num_classes)
    # define a function for extraction of variable names
    bptt_snn_output,_=bptt_snn_all_states(inputs,contexts)
    trainables=tf.trainable_variables()
    variable_names=[v.name for v in tf.trainable_variables()]
    bptt_snn_logit=tf.reduce_mean(bptt_snn_output[:,-context_timesteps:,:],axis=1)
    bptt_snn_loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=bptt_snn_logit)
    #bptt_snn_prediction = tf.nn.softmax(bptt_snn_logit)
    #bptt_snn_correct_pred = tf.equal(tf.argmax(bptt_snn_prediction, 1), tf.argmax(labels_temp, 1))
    #bptt_snn_accuracy = tf.reduce_mean(tf.cast(bptt_snn_correct_pred, tf.float32))
    optimizer = tf.train.AdamOptimizer(learning_rate=weight_learning_rate)
    train_op = optimizer.minimize(bptt_snn_loss)
    init = tf.global_variables_initializer()
# run a training session
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(epochs):
        sess.run(iter.initializer,feed_dict={X: x_train, Y: y_train ,Context: train_context, BATCH_SIZE: batch_size})
        for step in range(training_steps):
            bptt_snn_train, bptt_loss=sess.run([train_op,bptt_snn_loss])
            if step % display_step==0 :
                 print('Epoch: {}, Batch: {},bptt train Loss: {:.3f}'.format(epoch+1,step + 1, bptt_loss))

        print('Epoch: {}, done'.format(epoch+1))
    print("Optimization Finished!")
    print("Model saved in path: %s" % save_path)
