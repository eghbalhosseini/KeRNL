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
from pandas import DataFrame
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
import keRNL_cell_v2
################################################
# logic for getting the variables from 1 system argument
# each dictionary represent the set of values for variable
# on each call an iterator is build and a
training_steps_dict={"A":5000,"B":5000,"C":5000}
batch_size_dict={"A":100,"B":100,"C":100}
num_hidden_dict={"A":200,"B":200,"C":200}
grad_clip_dict={"A":100,"B":100,"C":100}
#
num_of_variables=4
# create an iterator and use it to determine the values for parameters
variable_combinations=list(itertools.product('ABC', repeat=num_of_variables))
# use input system arg to determine what element to use
variable_condition=variable_combinations[int(sys.argv[1])-1]
# determine the value for each variable
training_steps=training_steps_dict[variable_condition[0]]
batch_size=batch_size_dict[variable_condition[1]]
num_hidden=num_hidden_dict[variable_condition[2]]
grad_clip=grad_clip_dict[variable_condition[3]]

# Training Parameters and save location
weight_learning_rate = 1e-5 # learning rate for weights in the network
tensor_learning_rate = 1e-5 # learning rate for sensitivity tensor and temporal filter tensor
display_step = 200
test_len=128
# Network Parameters
num_input = 1 # MNIST data input (img shape: 28*28)
timesteps = 28*28 # timesteps
num_classes = 10 # MNIST total classes (0-9 digits)
perturbation_std=1e-3


print('training_steps: {}, batch_size: {}, num_hidden: {}, grad_clip: {}'.format(training_steps, batch_size, num_hidden, grad_clip))

################################################
# get mnist data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
tf.logging.set_verbosity(old_v)
################################################

log_dir = "/om/user/ehoseini/MyData/KeRNL/logs/KeRNL_vs_BPTT_MNIST/tensor_learn_%.1e_weight_learn_%.1e_tr_step_%.1e_batch_%.1e_hidd_%.1e_gc_%.1e_run_%s" %(tensor_learning_rate,weight_learning_rate,training_steps,batch_size,num_hidden,grad_clip, datetime.now().strftime("%Y%m%d_%H%M"))
Path(log_dir).mkdir(exist_ok=True, parents=True)
filelist = [ f for f in os.listdir(log_dir) if f.endswith(".local") ]
for f in filelist:
    os.remove(os.path.join(log_dir, f))

#################################################
## define KeRNL unit
## define KeRNL unit
def RNN(x,kernel_weights,kernel_bias,irnn_weights,irnn_bias):
    # Define a KeRNL cell, the initialization is done inside the cell with default initializers
    with tf.variable_scope('KeRNL') as scope:
        keRNL_v2 = keRNL_cell_v2.KeRNLCell_v2(num_units=num_hidden,num_inputs=num_input,time_steps=timesteps,noise_std=perturbation_std,sensitivity_initializer=tf.initializers.identity)
        # Get KeRNL cell output
        kernel_outputs, kernel_states = tf.nn.dynamic_rnn(keRNL_v2, inputs=x, dtype=tf.float32,time_major=False)
        keRNL_output=tf.matmul(kernel_outputs[:,-1,:], kernel_weights) + kernel_bias
    #
    with tf.variable_scope("IRNN",initializer=tf.initializers.identity()) as scope:
        irnn_cell = tf.contrib.rnn.BasicRNNCell(num_hidden,name='irnn')
        irnn_outputs, irnn_states = tf.nn.dynamic_rnn(irnn_cell, x, dtype=tf.float32)
        irnn_output=tf.matmul(irnn_outputs[:,-1,:], irnn_weights) + irnn_biases

    return keRNL_output, kernel_states, irnn_output, irnn_states
################################################
# define tensorflow graph for computation
tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():
    with tf.variable_scope('KeRNL_output',initializer=tf.initializers.random_normal()) as scope:
        keRNL_weights = tf.get_variable(shape=[num_hidden, num_classes],name='output_weight')
        keRNL_biases = tf.get_variable(shape=[num_classes],name='output_addition')
    with tf.variable_scope('IRNN_output',initializer=tf.initializers.random_normal()) as scope:
        irnn_weights = tf.get_variable(shape=[num_hidden, num_classes],name='output_weight')
        irnn_biases = tf.get_variable(shape=[num_classes],name='output_addition')

    # define weights and inputs to the network
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    # define a function for extraction of variable names
    keRNL_output,keRNL_states,IRNN_output,IRNN_states=RNN(X,keRNL_weights,keRNL_biases,irnn_weights,irnn_biases)
    trainables=tf.trainable_variables()
    variable_names=[v.name for v in tf.trainable_variables()]
    #
    find_joing_index = lambda x, name_1,name_2 : [a and b for a,b in zip([np.unicode_.find(k.name, name_1)>-1 for k in x] ,[np.unicode_.find(k.name, name_2)>-1 for k in x])].index(True)
    # find trainable parameters for keRNL
    with tf.name_scope('KeRNL_Trainables') as scope:
        keRNL_output_weight_index= find_joing_index(trainables,'KeRNL','output_weight')
        keRNL_output_addition_index= find_joing_index(trainables,'KeRNL','output_addition')
        keRNL_temporal_filter_index= find_joing_index(trainables,'KeRNL','temporal_filter')
        keRNL_sensitivity_tensor_index= find_joing_index(trainables,'KeRNL','sensitivity_tensor')
        keRNL_kernel_index= find_joing_index(trainables,'KeRNL','kernel')
        keRNL_bias_index= find_joing_index(trainables,'KeRNL','bias')
    #
        keRNL_tensor_training_indices=np.asarray([keRNL_sensitivity_tensor_index,keRNL_temporal_filter_index],dtype=np.int)
        keRNL_tensor_trainables= [trainables[k] for k in keRNL_tensor_training_indices]
    #
        keRNL_weight_training_indices=np.asarray([keRNL_kernel_index,keRNL_bias_index,keRNL_output_weight_index,keRNL_output_addition_index],dtype=np.int)
        keRNL_weight_trainables= [trainables[k] for k in keRNL_weight_training_indices]

    with tf.name_scope("IRNN_Trainables") as scope:
    # find trainables parameters for IRNN
        IRNN_output_weight_index= find_joing_index(trainables,'IRNN','output_weight')
        IRNN_output_addition_index= find_joing_index(trainables,'IRNN','output_addition')
        IRNN_kernel_index= find_joing_index(trainables,'IRNN','kernel')
        IRNN_bias_index= find_joing_index(trainables,'IRNN','bias')
    #
        IRNN_weight_training_indices=np.asarray([IRNN_kernel_index,IRNN_bias_index,IRNN_output_weight_index,IRNN_output_addition_index],dtype=np.int)
        IRNN_weight_trainables= [trainables[k] for k in IRNN_weight_training_indices]
    # keRNL
    # define loss functions
    ##################
    # keRNL train ####
    ##################
    with tf.name_scope("KeRNL_train") as scope:
        # outputs
        keRNL_loss_output_prediction=tf.losses.softmax_cross_entropy(onehot_labels=Y,logits=keRNL_output)
        keRNL_prediction = tf.nn.softmax(keRNL_output)
        keRNL_correct_pred = tf.equal(tf.argmax(keRNL_prediction, 1), tf.argmax(Y, 1))
        keRNL_accuracy = tf.reduce_mean(tf.cast(keRNL_correct_pred, tf.float32))
        # states
        keRNL_loss_state_prediction=tf.losses.mean_squared_error(tf.subtract(keRNL_states.h_hat, keRNL_states.h),tf.matmul(keRNL_states.Gamma,trainables[keRNL_sensitivity_tensor_index]))
        # define optimizers
        keRNL_weight_optimizer = tf.train.RMSPropOptimizer(learning_rate=weight_learning_rate)
        keRNL_tensor_optimizer = tf.train.RMSPropOptimizer(learning_rate=tensor_learning_rate)

        with tf.name_scope('KeRNL_train_tensors') as scope:
            keRNL_delta_sensitivity=tf.subtract(tf.matmul(keRNL_states.Theta,tf.transpose(trainables[keRNL_sensitivity_tensor_index])),tf.subtract(keRNL_states.h_hat,keRNL_states.h))
            keRNL_sensitivity_tensor_update= tf.reduce_mean(tf.einsum("un,uv->unv",keRNL_delta_sensitivity,keRNL_states.Theta),axis=0)
            keRNL_temporal_filter_update= tf.reduce_mean(tf.multiply(tf.matmul(keRNL_delta_sensitivity,trainables[keRNL_sensitivity_tensor_index]),keRNL_states.Gamma),axis=0)
            keRNL_tensor_grads_and_vars=list(zip([keRNL_sensitivity_tensor_update,keRNL_temporal_filter_update],keRNL_tensor_trainables))
            keRNL_cropped_tensor_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) if  np.unicode_.find(var.name,'output')==-1 else (grad,var) for grad,var in keRNL_tensor_grads_and_vars]
            keRNL_tensor_train_op = keRNL_tensor_optimizer.apply_gradients(keRNL_cropped_tensor_grads_and_vars)

        with tf.name_scope('KeRNL_train_weights') as scope:
            keRNL_grad_cost_to_output=tf.gradients(keRNL_loss_output_prediction,keRNL_output, name= 'keRNL_grad_cost_to_y')
            keRNL_error_in_hidden_state=tf.matmul(keRNL_grad_cost_to_output[-1],tf.transpose(trainables[keRNL_output_weight_index]))
            keRNL_delta_weight=tf.matmul(keRNL_error_in_hidden_state,trainables[keRNL_sensitivity_tensor_index])
            keRNL_weight_update_test=tf.einsum("un,unv->unv",keRNL_delta_weight,keRNL_states.eligibility_trace)
            keRNL_weight_update=tf.transpose(tf.reduce_mean(keRNL_weight_update_test,axis=0))

            keRNL_bias_update_test=tf.multiply(keRNL_delta_weight,keRNL_states.bias_trace)
            keRNL_bias_update=tf.reduce_mean(keRNL_bias_update_test,axis=0)
            # output layer
            keRNL_grad_cost_to_output_layer=tf.gradients(keRNL_loss_output_prediction,[trainables[keRNL_output_weight_index],trainables[keRNL_output_addition_index]], name= 'keRNL_grad_cost_to_output_layer')
            # crop the gradients
            keRNL_weight_grads_and_vars=list(zip([keRNL_weight_update,keRNL_bias_update,keRNL_grad_cost_to_output_layer[0],keRNL_grad_cost_to_output_layer[1]],keRNL_weight_trainables))
            keRNL_cropped_weight_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) if  np.unicode_.find(var.name,'output')==-1 else (grad,var) for grad,var in keRNL_weight_grads_and_vars]
            # apply gradients
            keRNL_weight_train_op = keRNL_weight_optimizer.apply_gradients(keRNL_cropped_weight_grads_and_vars)
    ##################
    # BPTT train #####
    ##################
    with tf.name_scope("IRNN_train") as scope:
        # BPTT
        IRNN_loss_output_prediction=tf.losses.softmax_cross_entropy(onehot_labels=Y,logits=IRNN_output)
        IRNN_prediction = tf.nn.softmax(IRNN_output)
        IRNN_correct_pred = tf.equal(tf.argmax(IRNN_prediction, 1), tf.argmax(Y, 1))
        IRNN_accuracy = tf.reduce_mean(tf.cast(IRNN_correct_pred, tf.float32))
        # define optimizer
        IRNN_weight_optimizer = tf.train.RMSPropOptimizer(learning_rate=weight_learning_rate)
        IRNN_grads=tf.gradients(IRNN_loss_output_prediction,IRNN_weight_trainables)
        IRNN_weight_grads_and_vars=list(zip(IRNN_grads,IRNN_weight_trainables))
        # Apply gradient Clipping to recurrent weights
        IRNN_cropped_weight_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) if  np.unicode_.find(var.name,'output')==-1 else (grad,var) for grad,var in IRNN_weight_grads_and_vars]
        # apply gradients
        IRNN_weight_train_op = IRNN_weight_optimizer.apply_gradients(IRNN_cropped_weight_grads_and_vars)
    ##################
    # SUMMARIES ######
    ##################
    with tf.name_scope("IRNN_summaries") as scope:
        # IRNN kernel
        tf.summary.histogram('IRNN_kernel_grad',IRNN_grads[0]+1e-10)
        tf.summary.histogram('IRNN_kernel', IRNN_weight_trainables[0]+1e-10)
        # IRNN output weight
        tf.summary.histogram('IRNN_output_weight_grad',IRNN_grads[2]+1e-10)
        tf.summary.histogram('IRNN_output_weights', IRNN_weight_trainables[2]+1e-10)
        # IRNN output bias
        tf.summary.histogram('irnn_output_addition_grad',IRNN_grads[3])
        tf.summary.histogram('irnn_output_addition', IRNN_weight_trainables[3]+1e-10)
        # IRNN loss and accuracy
        tf.summary.scalar('IRNN_loss_output_prediction',IRNN_loss_output_prediction)
        tf.summary.scalar('IRNN_accuracy',IRNN_accuracy)
        # IRNN kernel and ouput matrix
        tf.summary.image('IRNN_kernel_matrix',tf.expand_dims(tf.expand_dims(IRNN_weight_trainables[0],axis=0),axis=-1))
        tf.summary.image('IRNN_kernel_grad',tf.expand_dims(tf.expand_dims(IRNN_grads[0],axis=0),axis=-1))
        tf.summary.image('IRNN_output_matrix',tf.expand_dims(tf.expand_dims(IRNN_weight_trainables[2],axis=0),axis=-1))
        tf.summary.image('IRNN_output_matrix_grad',tf.expand_dims(tf.expand_dims(IRNN_grads[2],axis=0),axis=-1))
        IRNN_merged_summary_op=tf.summary.merge_all(scope="IRNN_summaries")

    with tf.name_scope("keRNL_tensor_summaries") as scope:
        # keRNL sensitivity tensor
        tf.summary.histogram('keRNL_sensitivity_tensor_grad',keRNL_sensitivity_tensor_update+1e-10)
        tf.summary.histogram('keRNL_sensitivity_tensor',trainables[keRNL_sensitivity_tensor_index]+1e-10)
        # keRNL temporal filter
        tf.summary.histogram('keRNL_temporal_filter_grad',keRNL_temporal_filter_update+1e-10)
        tf.summary.histogram('keRNL_temporal_filter',trainables[keRNL_temporal_filter_index]+1e-10)
        # keRNL loss
        tf.summary.scalar('keRNL_loss_state_prediction',keRNL_loss_state_prediction+1e-10)
        # keRNL senstivity tensor and temporal filter
        tf.summary.image('keRNL_sensitivity_tensor',tf.expand_dims(tf.expand_dims(trainables[keRNL_sensitivity_tensor_index],axis=0),axis=-1))
        tf.summary.image('keRNL_sensitivity_tensor_grad',tf.expand_dims(tf.expand_dims(keRNL_sensitivity_tensor_update,axis=0),axis=-1))
        tf.summary.image('keRNL_temporal_filter',tf.expand_dims(tf.expand_dims(tf.expand_dims(trainables[keRNL_temporal_filter_index],axis=0),axis=-1),axis=-1))
        tf.summary.image('keRNL_temporal_filter_grad',tf.expand_dims(tf.expand_dims(tf.expand_dims(keRNL_temporal_filter_update,axis=0),axis=-1),axis=-1))
        keRNL_tensor_merged_summary_op=tf.summary.merge_all(scope="keRNL_tensor_summaries")

    with tf.name_scope("keRNL_weight_summaries") as scope:
        # keRNL kernel
        tf.summary.histogram('keRNL_kernel_grad',keRNL_weight_update+1e-10)
        tf.summary.histogram('keRNL_kernel',trainables[keRNL_kernel_index]+1e-10)
        # keRNL bias
        tf.summary.histogram('keRNL_bias_grad',keRNL_bias_update+1e-10)
        tf.summary.histogram('keRNL_bias',trainables[keRNL_bias_index]+1e-10)
        # keRNL output weight
        tf.summary.histogram('keRNL_output_weight_grad',keRNL_grad_cost_to_output_layer[0]+1e-10)
        tf.summary.histogram('keRNL_output_weights', trainables[keRNL_output_weight_index]+1e-10)
        # keRNL output bias
        tf.summary.histogram('keRNL_output_addition_grad',keRNL_grad_cost_to_output_layer[1]+1e-10)
        tf.summary.histogram('keRNL_output_addition', trainables[keRNL_output_addition_index]+1e-10)
        # keRNL loss
        tf.summary.scalar('keRNL_loss_output_prediction',keRNL_loss_output_prediction+1e-10)
        tf.summary.scalar('keRNL_accuracy',keRNL_accuracy)
        # keRNL kernel and output weight
        tf.summary.image('keRNL_kernel',tf.expand_dims(tf.expand_dims(trainables[keRNL_kernel_index],axis=0),axis=-1))
        tf.summary.image('keRNL_kernel_grad',tf.expand_dims(tf.expand_dims(keRNL_weight_update,axis=0),axis=-1))
        tf.summary.image('keRNL_output_weight',tf.expand_dims(tf.expand_dims(trainables[keRNL_output_weight_index],axis=0),axis=-1))
        tf.summary.image('keRNL_output_weight_grad',tf.expand_dims(tf.expand_dims(keRNL_grad_cost_to_output_layer[0],axis=0),axis=-1))
        keRNL_weight_merged_summary_op=tf.summary.merge_all(scope="keRNL_weight_summaries")

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
############################################################################
# write graph into tensorboard
tb_writer = tf.summary.FileWriter(log_dir,graph)
print('graph saved to '+log_dir)
############################################################################
# run a training session
# write graph into tensorboard
tb_writer = tf.summary.FileWriter(log_dir,graph)
# run a training session
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for step in range(1,training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape((batch_size,timesteps,num_input))
        # IRNN train
        IRNN_train, IRNN_loss,IRNN_accu=sess.run([IRNN_weight_train_op,IRNN_loss_output_prediction,IRNN_accuracy],feed_dict={X:batch_x, Y:batch_y})

        # keRNL state  train
        keRNL_state_train, keRNL_tensor_loss=sess.run([keRNL_tensor_train_op,keRNL_loss_state_prediction],feed_dict={X:batch_x, Y:batch_y})

        # keRNL weight  train
        keRNL_weight_train, keRNL_loss,keRNL_accu=sess.run([keRNL_weight_train_op,keRNL_loss_output_prediction,keRNL_accuracy],feed_dict={X:batch_x, Y:batch_y})

        # run summaries
        IRNN_merged_summary=sess.run(IRNN_merged_summary_op,feed_dict={X:batch_x, Y:batch_y})
        keRNL_tensor_merged_summary=sess.run(keRNL_tensor_merged_summary_op,feed_dict={X:batch_x, Y:batch_y})
        keRNL_weight_merged_summary=sess.run(keRNL_weight_merged_summary_op,feed_dict={X:batch_x, Y:batch_y})

        tb_writer.add_summary(IRNN_merged_summary, global_step=step)
        tb_writer.add_summary(keRNL_tensor_merged_summary, global_step=step)
        tb_writer.add_summary(keRNL_weight_merged_summary, global_step=step)
        #
        if step % display_step==0 or step==1 :
            # get batch loss and accuracy
            print('Step: {},IRNN train Loss: {:.3f},IRNN accu: {:.3f}, keRNL tensor Loss {:.3f}, keRNL train Loss: {:.3f},keRNL accu: {:.3f}'.format(step + 1, IRNN_loss,IRNN_accu,keRNL_tensor_loss,keRNL_loss,keRNL_accu))


    print("Optimization Finished!")
    #test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    #test_label = mnist.test.labels[:test_len]
    #print("Testing Accuracy:",
    #    sess.run(loss_output_prediction, feed_dict={X: test_data, Y: test_label}))
    save_path = saver.save(sess, log_dir+"/model.ckpt", global_step=step,write_meta_graph=True)
    print("Model saved in path: %s" % save_path)
