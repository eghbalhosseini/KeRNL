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
# Training Parameters
tensor_learning_rate = 1e-5
weight_learning_rate = 1e-3
training_steps = 4000
buffer_size=500
batch_size = 25
training_size=batch_size*training_steps
epochs=100
test_size=10000
display_step = 100
grad_clip=100
# Network Parameters
num_input = 2 # adding problem data input (first input are the random digits , second input is the mask)
time_steps = 100
num_hidden = 100 # hidden layer num of features
num_output = 1 # value of the addition estimation
#
# Noise Parameters
perturbation_std=1e-5

#
# save dir
log_dir = "/om2/user/ehoseini/MyData/KeRNL/logs/kernl_rnn_addition_dataset/tanh_add_eta_weight_%1.0e_batch_%1.0e_hum_hidd_%1.0e_gc_%1.0e_steps_%1.0e_run_%s" %(weight_learning_rate,batch_size,num_hidden,grad_clip,training_steps, datetime.now().strftime("%Y%m%d_%H%M"))
log_dir

# create a training and testing dataset
training_x, training_y = adding_problem.get_batch(batch_size=training_size,time_steps=time_steps)
testing_x, testing_y = adding_problem.get_batch(batch_size=test_size,time_steps=time_steps)


## define KeRNL unit
def kernl_rnn(x,kernel_weights,kernel_bias):
    # Define a KeRNL cell, the initialization is done inside the cell with default initializers
    with tf.variable_scope('kernl') as scope:
        kernl_rnn_unit = kernl_rnn_cell.kernl_rnn_cell(num_units=num_hidden,
                                                      num_inputs=num_input,
                                                      time_steps=time_steps,
                                                      noise_param=perturbation_std,
                                                      sensitivity_initializer=tf.initializers.identity
                                                      ,activation="tanh")
        # Get KeRNL cell output
        kernel_outputs, kernel_states = tf.nn.dynamic_rnn(kernl_rnn_unit, inputs=x, dtype=tf.float32,time_major=False)
        kernl_rnn_output=tf.matmul(kernel_outputs[:,-1,:], kernel_weights) + kernel_bias
    return kernl_rnn_output, kernel_states

##################################################
tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():
    with tf.variable_scope('kernl_output',initializer=tf.contrib.layers.xavier_initializer()) as scope:
        kernl_weights = tf.get_variable(shape=[num_hidden, num_output],name='output_weight')
        kernl_biases = tf.get_variable(shape=[num_output],name='output_addition')

    # define weights and inputs to the network
    BATCH_SIZE=tf.placeholder(tf.int64)
    X = tf.placeholder("float", [None, time_steps, num_input])
    Y = tf.placeholder("float", [None, num_output])
    # define a function for extraction of variable names
    dataset=tf.data.Dataset.from_tensor_slices((X,Y)).batch(BATCH_SIZE).repeat()
    dataset = dataset.shuffle(buffer_size=buffer_size)
    iter = dataset.make_initializable_iterator()
    inputs,labels =iter.get_next()
    #
    kernl_output,kernl_states=kernl_rnn(inputs,kernl_weights,kernl_biases)
    trainables=tf.trainable_variables()
    variable_names=[v.name for v in tf.trainable_variables()]
    #
    find_joing_index = lambda x, name_1,name_2 : [a and b for a,b in zip([np.unicode_.find(k.name, name_1)>-1 for k in x] ,[np.unicode_.find(k.name, name_2)>-1 for k in x])].index(True)
    # find trainable parameters for kernl
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

    # define loss functions
    ##################
    # kernl train ####
    ##################
    with tf.name_scope("kernl_train") as scope:
        # outputs
        kernl_loss_output_prediction=tf.losses.mean_squared_error(labels,kernl_output)

        # states
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

    with tf.name_scope("kernl_evaluate") as scope:
      kernl_loss_cross_validiation=tf.losses.mean_squared_error(labels,kernl_output)

    with tf.name_scope('cross_validation_summary') as scope:
        tf.summary.scalar('cross_validation_summary',kernl_loss_cross_validiation+1e-10)
        kernl_evaluate_summary_op=tf.summary.merge_all(scope="cross_validation_summary")
    ##################
    # SUMMARIES ######
    ##################

    with tf.name_scope("kernl_tensor_summaries") as scope:
        # kernl sensitivity tensor
        tf.summary.histogram('kernl_sensitivity_tensor_grad',kernl_sensitivity_tensor_update+1e-10)
        tf.summary.histogram('kernl_sensitivity_tensor',trainables[kernl_sensitivity_tensor_index]+1e-10)
        # kernl temporal filter
        tf.summary.histogram('kernl_temporal_filter_grad',kernl_temporal_filter_update+1e-10)
        tf.summary.histogram('kernl_temporal_filter',trainables[kernl_temporal_filter_index]+1e-10)
        # kernl loss
        tf.summary.scalar('kernl_loss_state_prediction',kernl_loss_state_prediction+1e-10)
        # kernl senstivity tensor and temporal filter
        #tf.summary.image('kernl_sensitivity_tensor',tf.expand_dims(tf.expand_dims(trainables[kernl_sensitivity_tensor_index],axis=0),axis=-1))
        #tf.summary.image('kernl_sensitivity_tensor_grad',tf.expand_dims(tf.expand_dims(kernl_sensitivity_tensor_update,axis=0),axis=-1))
        #tf.summary.image('kernl_temporal_filter',tf.expand_dims(tf.expand_dims(tf.expand_dims(trainables[kernl_temporal_filter_index],axis=0),axis=-1),axis=-1))
        #tf.summary.image('kernl_temporal_filter_grad',tf.expand_dims(tf.expand_dims(tf.expand_dims(kernl_temporal_filter_update,axis=0),axis=-1),axis=-1))
        kernl_tensor_merged_summary_op=tf.summary.merge_all(scope="kernl_tensor_summaries")

    with tf.name_scope("kernl_weight_summaries") as scope:
        # kernl kernel
        tf.summary.histogram('kernl_kernel_grad',kernl_weight_update+1e-10)
        tf.summary.histogram('kernl_kernel',trainables[kernl_kernel_index]+1e-10)
        # kernl bias
        tf.summary.histogram('kernl_bias_grad',kernl_bias_update+1e-10)
        tf.summary.histogram('kernl_bias',trainables[kernl_bias_index]+1e-10)
        # kernl output weight
        tf.summary.histogram('kernl_output_weight_grad',kernl_grad_cost_to_output_layer[0]+1e-10)
        tf.summary.histogram('kernl_output_weights', trainables[kernl_output_weight_index]+1e-10)
        # kernl output bias
        tf.summary.histogram('kernl_output_addition_grad',kernl_grad_cost_to_output_layer[1]+1e-10)
        tf.summary.histogram('kernl_output_addition', trainables[kernl_output_addition_index]+1e-10)
        # kernl loss
        tf.summary.scalar('kernl_loss_output_prediction',kernl_loss_output_prediction+1e-10)
        # kernl kernel and output weight
        #tf.summary.image('kernl_kernel',tf.expand_dims(tf.expand_dims(trainables[kernl_kernel_index],axis=0),axis=-1))
        #tf.summary.image('kernl_kernel_grad',tf.expand_dims(tf.expand_dims(kernl_weight_update,axis=0),axis=-1))
        #tf.summary.image('kernl_output_weight',tf.expand_dims(tf.expand_dims(trainables[kernl_output_weight_index],axis=0),axis=-1))
        #tf.summary.image('kernl_output_weight_grad',tf.expand_dims(tf.expand_dims(kernl_grad_cost_to_output_layer[0],axis=0),axis=-1))
        kernl_weight_merged_summary_op=tf.summary.merge_all(scope="kernl_weight_summaries")

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    ###############################################

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
    for epoch in range(epochs):
        sess.run(iter.initializer,feed_dict={X: training_x, Y: training_y , BATCH_SIZE: batch_size})
        for step in range(training_steps):
            kernl_state_train, kernl_tensor_loss=sess.run([kernl_tensor_train_op,kernl_loss_state_prediction])
            kernl_weight_train, kernl_loss=sess.run([kernl_weight_train_op,kernl_loss_output_prediction])
            if step % display_step==0:
                kernl_merged_tensor_summary, kernl_merged_weight_summary=sess.run([kernl_tensor_merged_summary_op,kernl_weight_merged_summary_op])
                tb_writer.add_summary(kernl_merged_tensor_summary, global_step=epoch*training_steps+step+1)
                tb_writer.add_summary(kernl_merged_weight_summary, global_step=epoch*training_steps+step+1)
                print('Epoch: {}, Batch: {} , total batch {}, total trials: {},kernl train Loss: {:.3f}, kernl_state_loss {:.3f}'.format(epoch+1,step + 1,epoch*training_steps+step+1,(epoch*training_steps+step+1)*batch_size, kernl_loss,kernl_tensor_loss))
        # run test at the end of each epoch
        sess.run(iter.initializer, feed_dict={ X: testing_x, Y: testing_y, BATCH_SIZE: testing_x.shape[0]})
        kernl_test_loss, kernl_evaluate_summary=sess.run([kernl_loss_cross_validiation,kernl_evaluate_summary_op])
        tb_writer.add_summary(kernl_evaluate_summary, global_step=epoch*training_steps+training_steps+1)
        print('cross validation loss {:.3f} at the end of epoch {}'.format(kernl_test_loss,epoch+1))
    print("Optimization Finished!")
    save_path = saver.save(sess, log_dir+"/model.ckpt", global_step=epoch*training_steps+training_steps,write_meta_graph=True)
    print("Model saved in path: %s" % save_path)
