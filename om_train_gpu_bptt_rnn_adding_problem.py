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
import adding_problem

# uplading mnist data


# Training Parameters
weight_learning_rate = 1e-3
training_steps = 500
batch_size = 20
test_size=10000
display_step = 100
grad_clip=100
# Network Parameters
num_input = 2 # adding problem data input (first input are the random digits , second input is the mask)
time_steps = 200
num_hidden = 100 # hidden layer num of features
num_output = 1 # value of the addition estimation
#
# save dir
log_dir = "om2/user/ehoseini/MyData/KeRNL/logs/bptt_rnn_addition/add_eta_weight_%1.0e_batch_%1.0e_hum_hidd_%1.0e_gc_%1.0e_steps_%1.0e_run_%s" %(weight_learning_rate,batch_size,num_hidden,grad_clip,training_steps, datetime.now().strftime("%Y%m%d_%H%M"))
log_dir

# report batch number
total_batch = int(mnist.train.num_examples / batch_size)
print("Total number of batches:", total_batch)

## define KeRNL unit
def bptt_rnn(x,rnn_weights,rnn_bias):
    # Define a KeRNL cell, the initialization is done inside the cell with default initializers
    with tf.variable_scope("bptt",initializer=tf.initializers.identity()) as scope:
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_hidden,name='irnn')
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
        rnn_output=tf.matmul(rnn_outputs[:,-1,:], rnn_weights) +rnn_biases

    return rnn_output, rnn_states

tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():
    with tf.variable_scope('bptt_output',initializer=tf.initializers.random_normal()) as scope:
        rnn_weights = tf.get_variable(shape=[num_hidden, num_output],name='output_weight')
        rnn_biases = tf.get_variable(shape=[num_output],name='output_addition')
    # define weights and inputs to the network
    X = tf.placeholder("float", [None, time_steps, num_input])
    Y = tf.placeholder("float", [None, num_output])
    # define a function for extraction of variable names
    rnn_output,rnn_states=bptt_rnn(X,rnn_weights,rnn_biases)
    trainables=tf.trainable_variables()
    variable_names=[v.name for v in tf.trainable_variables()]
    #
    find_joing_index = lambda x, name_1,name_2 : [a and b for a,b in zip([np.unicode_.find(k.name, name_1)>-1 for k in x] ,[np.unicode_.find(k.name, name_2)>-1 for k in x])].index(True)
    # find trainable parameters for kernl
    with tf.name_scope("bptt_Trainables") as scope:
            # find trainables parameters for bptt
        bptt_output_weight_index= find_joing_index(trainables,'bptt','output_weight')
        bptt_output_addition_index= find_joing_index(trainables,'bptt','output_addition')
        bptt_kernel_index= find_joing_index(trainables,'bptt','kernel')
        bptt_bias_index= find_joing_index(trainables,'bptt','bias')
            #
        bptt_weight_training_indices=np.asarray([bptt_kernel_index,bptt_bias_index,bptt_output_weight_index,bptt_output_addition_index],dtype=np.int)
        bptt_weight_trainables= [trainables[k] for k in bptt_weight_training_indices]

            ##################
            ## bptt train ####
            ##################
    with tf.name_scope("bptt_train") as scope:
                # BPTT
        bptt_loss_output_prediction=tf.losses.mean_squared_error(Y,rnn_output)
                # define optimizer
        bptt_weight_optimizer = tf.train.RMSPropOptimizer(learning_rate=weight_learning_rate)
        bptt_grads=tf.gradients(bptt_loss_output_prediction,bptt_weight_trainables)
        bptt_weight_grads_and_vars=list(zip(bptt_grads,bptt_weight_trainables))
                # Apply gradient Clipping to recurrent weights
        bptt_cropped_weight_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) if  np.unicode_.find(var.name,'output')==-1 else (grad,var) for grad,var in bptt_weight_grads_and_vars]
                # apply gradients
        bptt_weight_train_op = bptt_weight_optimizer.apply_gradients(bptt_cropped_weight_grads_and_vars)
    with tf.name_scope("bptt_evaluate") as scope:
        bptt_loss_cross_validiation=tf.losses.mean_squared_error(Y,rnn_output)

    with tf.name_scope('cross_validation_summary') as scope:
        tf.summary.scalar('cross_validation_summary',bptt_loss_cross_validiation+1e-10)
        bptt_evaluate_summary_op=tf.summary.merge_all(scope="cross_validation_summary")

                ##################
                # SUMMARIES ######
                ##################

    with tf.name_scope("bptt_summaries") as scope:
                    # bptt kernel
        tf.summary.histogram('bptt_kernel_grad',bptt_grads[0]+1e-10)
        tf.summary.histogram('bptt_kernel', bptt_weight_trainables[0]+1e-10)
                    # bptt output weight
        tf.summary.histogram('bptt_output_weight_grad',bptt_grads[2]+1e-10)
        tf.summary.histogram('bptt_output_weights', bptt_weight_trainables[2]+1e-10)
                    # bptt output bias
        tf.summary.histogram('bptt_output_addition_grad',bptt_grads[3]+1e-10)
        tf.summary.histogram('bptt_output_addition', bptt_weight_trainables[3]+1e-10)
                    # bptt loss and accuracy
        tf.summary.scalar('bptt_loss_output_prediction',bptt_loss_output_prediction+1e-10)


        for grad, var in bptt_weight_grads_and_vars:
            norm = tf.norm(tf.clip_by_norm(grad, 10.), ord=2)
            tf.summary.histogram(var.name.replace(":", "_") + '/gradient_l2',
                                 tf.where(tf.is_nan(norm), tf.zeros_like(norm), norm))
        for grad, var in bptt_cropped_weight_grads_and_vars:
            norm = tf.norm(grad, ord=2)
            tf.summary.histogram(var.name.replace(":", "_") + '/gradient_clipped_l2',
                                 tf.where(tf.is_nan(norm), tf.zeros_like(norm), norm))

        bptt_merged_summary_op=tf.summary.merge_all(scope="bptt_summaries")
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    print("All parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.global_variables()]))
    print("Trainable parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))
###################################################

Path(log_dir).mkdir(exist_ok=True, parents=True)
filelist = [ f for f in os.listdir(log_dir) if f.endswith(".local") ]
for f in filelist:
    os.remove(os.path.join(log_dir, f))
####################################################

# write graph into tensorboard
tb_writer = tf.summary.FileWriter(log_dir,graph)
# run a training session
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for step in range(1,training_steps+1):
        batch_x, batch_y = adding_problem.get_batch(batch_size=batch_size,time_steps=time_steps)
        # bptt train
        bptt_train, bptt_loss,bptt_merged_summary=sess.run([bptt_weight_train_op,bptt_loss_output_prediction,bptt_merged_summary_op],feed_dict={X:batch_x, Y:batch_y})

        # run summaries
        #bptt_merged_summary=sess.run(bptt_merged_summary_op,feed_dict={X:batch_x, Y:batch_y})

        tb_writer.add_summary(bptt_merged_summary, global_step=step)

        #
        if step % display_step==0 or step==1 :
            test_batch_x, test_batch_y = adding_problem.get_batch(batch_size=test_size,time_steps=time_steps)
            bptt_test_loss, bptt_evaluate_summary=sess.run([bptt_loss_cross_validiation,bptt_evaluate_summary_op],feed_dict={X:test_batch_x, Y:test_batch_y})
            # get batch loss and accuracy
            tb_writer.add_summary(bptt_evaluate_summary, global_step=step)
            print('Step: {},bptt train Loss: {:.3f}, cross validation loss :{:.3f}'.format(step + 1, bptt_loss,bptt_test_loss))


    print("Optimization Finished!")
    #test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    #test_label = mnist.test.labels[:test_len]
    #print("Testing Accuracy:",
    #    sess.run(loss_output_prediction, feed_dict={X: test_data, Y: test_label}))
    save_path = saver.save(sess, log_dir+"/model.ckpt", global_step=step,write_meta_graph=True)
    print("Model saved in path: %s" % save_path)
