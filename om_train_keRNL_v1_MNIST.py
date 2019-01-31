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
# kernel rnn cell
import keRNL_cell_v1
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
# Training Parameters and save location
weight_learning_rate = 1e-8 # learning rate for weights in the network
tensor_learning_rate = 1e-5 # learning rate for sensitivity tensor and temporal filter tensor
training_steps = 20
batch_size = 50
display_step = 10
test_len=128
grad_clip=2
# Network Parameters
num_input = 1 # MNIST data input (img shape: 28*28)
timesteps = 28*28 # timesteps
num_hidden = 100 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)
perturbation_std=1e-3

log_dir = "/om/user/ehoseini/MyData/KeRNL/logs/kernel_rnn_v1/two_optimizaer/MNIST_gc_%d_eta_m_%d_eta_%d_batch_%d_run_%s"
 %(grad_clip,tensor_learning_rate,weight_learning_rate,batch_size, datetime.now().strftime("%Y%m%d_%H%M"))
Path(log_dir).mkdir(exist_ok=True, parents=True)
filelist = [ f for f in os.listdir(log_dir) if f.endswith(".local") ]
for f in filelist:
    os.remove(os.path.join(log_dir, f))

#################################################
## define KeRNL unit
def kernel_RNN_v1(x, weights, biases):
    # Define a KeRNL cell, the initialization is done inside the cell with default initializers
    keRNL_v1 = keRNL_cell_v1.KeRNLCell_v1(num_units=num_hidden,
                                       num_inputs=num_input,
                                       time_steps=timesteps,
                                       noise_std=perturbation_std,
                                       sensitivity_initializer=tf.initializers.identity)
        # Get KeRNL cell output
    kernel_outputs, kernel_states = tf.nn.dynamic_rnn(keRNL_v1, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(kernel_outputs[:,-1,:], weights['out']) + biases['out'], kernel_states
################################################
# define tensorflow graph for computation
tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():

    # define weights and inputs to the network
    with tf.variable_scope('output_layer', initializer=tf.contrib.layers.xavier_initializer()) as scope:
        weights = {'out': tf.get_variable(shape=[num_hidden, num_classes],name='output_weight')}
        biases = {'out': tf.get_variable(shape=[num_classes],name='output_addition')}
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # define network output and trainiables
    logits,states = kernel_RNN_v1(X, weights, biases)
    variable_names=[v.name for v in tf.trainable_variables()]
    trainables=tf.trainable_variables()

    # get the index of trainable variables
    temporal_filter_index=[np.unicode_.find(k.name,'temporal_filter')>-1 for k in trainables].index(True)
    sensitivity_tensor_index=[np.unicode_.find(k.name,'sensitivity_tensor')>-1 for k in trainables].index(True)
    kernel_index=[np.unicode_.find(k.name,'kernel')>-1 for k in trainables].index(True)
    #bias_index=[np.unicode_.find(k.name,'bias')>-1 for k in trainables].index(True)
    output_weight_index=[np.unicode_.find(k.name,'output_weight')>-1 for k in trainables].index(True)
    output_addition_index=[np.unicode_.find(k.name,'output_addition')>-1 for k in trainables].index(True)

    # trainables for tensors
    tensor_training_indices=np.asarray([sensitivity_tensor_index,
                                        temporal_filter_index],dtype=np.int)
    tensor_trainables= [trainables[k] for k in tensor_training_indices]

    # trainables for weights
    weight_training_indices=np.asarray([kernel_index,
                                        output_weight_index,
                                        output_addition_index],dtype=np.int)
    weight_trainables= [trainables[k] for k in weight_training_indices]

## compute lossses
    # compute loss for predictions.
    loss_output_prediction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
    prediction = tf.nn.softmax(logits)

    # compute loss for estimating sensitivity tensor and temporal_filter_coeff,
    loss_state_prediction=tf.losses.mean_squared_error(tf.subtract(states.h_hat, states.h),
                                                       tf.matmul(states.Gamma,trainables[sensitivity_tensor_index]))

## define optimizers
    # define optimizers learning the weights
    weight_optimizer = tf.train.RMSPropOptimizer(learning_rate=weight_learning_rate)

    # define optimizer for learning the sensitivity tensor and temporal filter
    tensor_optimizer = tf.train.RMSPropOptimizer(learning_rate=tensor_learning_rate)

## get gradients and apply them
## optimize for temporal_filter and sensitivity_tensor
    # calculate gradients
    #sensitivity_tensor_update=tf.gradients(xs=trainables[sensitivity_tensor_index],ys=loss_state_prediction)
    #temporal_filter_update=tf.gradients(xs=trainables[temporal_filter_index],ys=loss_state_prediction)
    #tensor_grads_and_vars=list(zip([sensitivity_tensor_update[-1],temporal_filter_update[-1]],tensor_trainables))
    #cropped_tensor_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) if  np.unicode_.find(var.name,'output')==-1 else
    #                        (grad,var) for grad,var in tensor_grads_and_vars]
    # manually calculate gradients
    delta_sensitivity=tf.subtract(tf.matmul(states.Theta,
                                            tf.transpose(trainables[sensitivity_tensor_index])),
                                  tf.subtract(states.h_hat,states.h))
    sensitivity_tensor_update= tf.reduce_mean(tf.einsum("un,uv->unv",delta_sensitivity,states.Theta),axis=0)
    #
    temporal_filter_update= tf.reduce_mean(tf.multiply(tf.matmul(delta_sensitivity,
                                                                 trainables[sensitivity_tensor_index]),
                                                      states.Gamma),axis=0)
    tensor_grads_and_vars=list(zip([sensitivity_tensor_update,temporal_filter_update],tensor_trainables))
    cropped_tensor_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) if  np.unicode_.find(var.name,'output')==-1 else
                            (grad,var) for grad,var in tensor_grads_and_vars]
    # apply gradients
    tensor_train_op = tensor_optimizer.apply_gradients(tensor_grads_and_vars)

## optimize for recurrent weights and output weights
    # 1- gradient for the recurrent weights
    grad_cost_to_output=tf.gradients(loss_output_prediction,logits, name= 'grad_cost_to_y')
    error_in_hidden_state=tf.matmul(grad_cost_to_output[-1],tf.transpose(trainables[output_weight_index]))
    weight_update_aux=tf.matmul(error_in_hidden_state,trainables[sensitivity_tensor_index])
    weight_update=tf.transpose(tf.reduce_mean(tf.multiply(states.eligibility_trace,tf.expand_dims(weight_update_aux,axis=-1)),axis=0))

    #2- gradient for output weight
    grad_cost_to_output_layer=tf.gradients(loss_output_prediction,[trainables[output_weight_index],trainables[output_addition_index]], name= 'grad_cost_to_output_layer')
    #grad_cost_to_output_bias=tf.gradients(loss_output_prediction,trainables[output_addition_index], name= 'grad_cost_to_output_bias')
    # zip gradients and vars
    weight_grads_and_vars=list(zip([weight_update,grad_cost_to_output_layer[0],grad_cost_to_output_layer[1]],weight_trainables))
    # Apply gradient Clipping to recurrent weights
    cropped_weight_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) if  np.unicode_.find(var.name,'output')==-1 else
                            (grad,var) for grad,var in weight_grads_and_vars]

    # apply gradients
    weight_train_op = weight_optimizer.apply_gradients(cropped_weight_grads_and_vars)
    # create a train operation
    train_op=tf.group(tensor_train_op,weight_train_op)

    ## Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    ## get variables to save to tensorboard
    # network output
    tf.summary.histogram('prediction',prediction+1e-8)
    tf.summary.histogram('logits',logits+1e-8)

    # tensor training parameters
    tf.summary.histogram('sensitivity_updates',sensitivity_tensor_update[-1]+1e-10)
    tf.summary.histogram('temporal_filter_updates',temporal_filter_update[-1]+1e-10)
    tf.summary.histogram('sensitivity_tensor',trainables[sensitivity_tensor_index]+1e-10)
    tf.summary.histogram('temporal_filter',trainables[temporal_filter_index]+1e-10)
    tf.summary.scalar('loss_state_prediction',loss_state_prediction)

    # weight training parameters
    tf.summary.histogram('weight_updates',weight_update+1e-10)
    tf.summary.histogram('output_weight_updates',grad_cost_to_output_layer[0]+1e-10)
    tf.summary.histogram('output_bias_updates',grad_cost_to_output_layer[1]+1e-10)
    tf.summary.histogram('weights', trainables[kernel_index]+1e-10)
    tf.summary.histogram('output_weights', trainables[output_weight_index]+1e-10)
    tf.summary.histogram('output_addition', trainables[output_addition_index]+1e-10)
    tf.summary.histogram('error_in_hidden_state', error_in_hidden_state+1e-10)
    tf.summary.histogram('weight_update_aux', weight_update_aux+1e-10)
    tf.summary.scalar('loss_output_prediction',loss_output_prediction)

    #
    tf.summary.image('kernel_matrix',tf.expand_dims(tf.expand_dims(trainables[kernel_index],axis=0),axis=-1))
    tf.summary.image('sensitivity_matrix',tf.expand_dims(tf.expand_dims(trainables[sensitivity_tensor_index],axis=0),axis=-1))

    # merge and save all
    merged_summary_op=tf.summary.merge_all()

    # save training
    saver = tf.train.Saver()

############################################################################
# write graph into tensorboard
tb_writer = tf.summary.FileWriter(log_dir,graph)
print('graph saved to '+log_dir)
############################################################################
# run a training session
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for step in range(1,training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape((batch_size,timesteps,num_input))

        # run optimizer
        train_opt, tensor_loss,weight_loss,acc_train=sess.run([train_op,loss_state_prediction,loss_output_prediction,accuracy],
                                                              feed_dict={X:batch_x, Y:batch_y})
        # run summaries
        merged_summary=sess.run(merged_summary_op,feed_dict={X:batch_x, Y:batch_y})
        tb_writer.add_summary(merged_summary, global_step=step)

        if step % display_step==0 or step==1 :
            # get batch loss and accuracy
            print('Step: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, state loss: {:.3f}'.format(
            step + 1, weight_loss, acc_train, tensor_loss))


    print("Optimization Finished!")
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:",
        sess.run(loss_output_prediction, feed_dict={X: test_data, Y: test_label}))
    save_path = saver.save(sess, log_dir+"/model.ckpt", global_step=step,write_meta_graph=True)
    print("Model saved in path: %s" % save_path)
