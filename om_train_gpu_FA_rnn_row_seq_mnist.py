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
import random
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
import fa_rnn_cell
# uplading mnist data

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



# Training Parameters
# Training Parameters
# Training Parameters
learning_rate = 0.001
training_steps = 100000
batch_size = 128
display_step = 20

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 64 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)
#

#
# save dir
log_dir = "/om/user/ehoseini/MyData/KeRNL/logs/bptt_fa_rnn_seq_mnist_dataset/beta_true_normal_1e-1_fa_rnn_relu_add_eta_weight_%1.0e_batch_%1.0e_hum_hidd_%1.0e_steps_%1.0e_run_%s" %(learning_rate,batch_size,num_hidden,training_steps, datetime.now().strftime("%Y%m%d_%H%M"))
log_dir
# create a training and testing dataset
training_x, training_y = adding_problem.get_batch(batch_size=training_size,time_steps=time_steps)
testing_x, testing_y = adding_problem.get_batch(batch_size=test_size,time_steps=time_steps)

def _hinton_identity_initializer(shape,dtype=None,partition_info=None,verify_shape=None, max_val=1):
    if dtype is None:
        dtype=tf.float32
    #extract second dimension
    W_rec=tf.eye(shape[-1],dtype=dtype)
    new_shape=[shape[0]-shape[-1],shape[-1]]
    W_in = tf.get_variable("W_in", shape=new_shape,
           initializer=tf.contrib.layers.xavier_initializer())
    #W_in=tf.random_normal(new_shape,mean=0,stddev=0.001)
    return tf.concat([W_in,W_rec],axis=0)

# Define weights remember these are output weights and biases, the internal weights and biases are hidden for lstm
def RNN(x, weights, biases):

    x = tf.unstack(x, timesteps, 1)
    lstm_cell = fa_rnn_cell.fa_rnn_cell(num_hidden,kernel_initializer=tf.contrib.layers.xavier_initializer())
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases
##################################################
tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():
    with tf.variable_scope('bptt_output',initializer=tf.initializers.random_normal()) as scope:
        rnn_weights = tf.get_variable(shape=[num_hidden, num_classes],name='output_weight')
        rnn_biases = tf.get_variable(shape=[num_classes],name='output_addition')
    # define weights and inputs to the network
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    # define a function for extraction of variable names
    logits = RNN(X, rnn_weights, rnn_biases)
    prediction = tf.nn.softmax(logits)

    trainables=tf.trainable_variables()
    all_variable=tf.all_variables()
    variable_names=[v.name for v in tf.trainable_variables()]
    all_variable_names=[v.name for v in tf.all_variables()]
    #
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    grads_and_vars=optimizer.compute_gradients(loss_op)
    train_op = optimizer.apply_gradients(grads_and_vars)


    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.histogram('bptt_kernel_grad',grads_and_vars[2][0]+1e-10)
    tf.summary.histogram('bptt_kernel', trainables[2]+1e-10)
    tf.summary.histogram('bptt_output_weight_grad',grads_and_vars[0][0]+1e-10)
    tf.summary.histogram('bptt_output_weights', trainables[0]+1e-10)
    tf.summary.scalar('loss',loss_op+1e-10)
    tf.summary.scalar('accuracy',accuracy+1e-10)
    bptt_merged_summary_op=tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    ###############################################

Path(log_dir).mkdir(exist_ok=True, parents=True)
filelist = [ f for f in os.listdir(log_dir) if f.endswith(".local") ]
for f in filelist:
    os.remove(os.path.join(log_dir, f))
####################################################
# write graph into tensorboard
tb_writer = tf.summary.FileWriter(log_dir,graph)
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        train,merged_summary=sess.run([train_op,bptt_merged_summary_op], feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            tb_writer.add_summary(merged_summary, global_step=step)
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    print("Optimization Finished!")
    save_path = saver.save(sess, log_dir+"/model.ckpt", global_step=step,write_meta_graph=True)
    print("Model saved in path: %s" % save_path)
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
    save_path = saver.save(sess, log_dir+"/model.ckpt", write_meta_graph=True)
    print("Model saved in path: %s" % save_path)
