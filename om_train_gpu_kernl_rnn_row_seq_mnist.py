# python libraries
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
import tensorflow as tf

import sys
sys.path.append('../')

import adding_problem
import kernl_rnn_cell


from pathlib import Path
import random
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MNIST data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
tf.logging.set_verbosity(old_v)


# Training Parameters
# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 64 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# Training Parameters
tensor_learning_rate = 1e-5
weight_learning_rate = 1e-3
training_steps = 4000
batch_size = 20
display_step = 25
test_len=128
grad_clip=100
# Network Parameters
# Noise Parameters
perturbation_std=1e-10
#
# save dir
log_dir = "/om/user/ehoseini/MyData/KeRNL/logs/kernl_rnn_row_seq_mnist/kernl_rnn_tanh_row_mnist_T_%1.0e_tanh_add_eta_W_%1.0e_eta_T_%1.0e_Noise_%1.0e_batch_%1.0e_hum_hidd_%1.0e_gc_%1.0e_steps_%1.0e_run_%s" %(timesteps,weight_learning_rate,tensor_learning_rate,perturbation_std,batch_size,num_hidden,grad_clip,training_steps, datetime.now().strftime("%Y%m%d_%H%M"))
log_dir
# create a training and testing dataset
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
## define KeRNL unit
def kernl_rnn(x,kernel_weights,kernel_bias):
    # Define a KeRNL cell, the initialization is done inside the cell with default initializers
    with tf.variable_scope('kernl') as scope:
        kernl_rnn_unit = kernl_rnn_cell.kernl_rnn_cell(num_units=num_hidden,
                                                      num_inputs=num_input,
                                                      time_steps=timesteps,
                                                      noise_param=perturbation_std,
                                                      sensitivity_initializer=tf.contrib.layers.xavier_initializer()
                                                      ,activation="tanh",
                                                      kernel_initializer=tf.contrib.layers.xavier_initializer()
                                                      )
        # Get KeRNL cell output
        kernel_outputs, kernel_states = tf.nn.dynamic_rnn(kernl_rnn_unit, inputs=x, dtype=tf.float32,time_major=False)
        kernl_rnn_output=tf.matmul(kernel_outputs[:,-1,:], kernel_weights) + kernel_bias
    return kernl_rnn_output, kernel_states
##################################################
tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():
    with tf.variable_scope('kernl_output',initializer=tf.contrib.layers.xavier_initializer()) as scope:
        kernl_weights = tf.get_variable(shape=[num_hidden, num_classes],name='output_weight')
        kernl_biases = tf.get_variable(shape=[num_classes],name='output_addition')

    # define weights and inputs to the network
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    # define a function for extraction of variable names
    kernl_output,kernl_states=kernl_rnn(X,kernl_weights,kernl_biases)
    prediction = tf.nn.softmax(kernl_output)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
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
        kernl_loss_output_prediction=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=kernl_output, labels=Y))

        # states
        kernl_loss_state_prediction=tf.losses.mean_squared_error(tf.subtract(kernl_states.h_hat, kernl_states.h),tf.matmul(kernl_states.Theta,trainables[kernl_sensitivity_tensor_index])) # define optimizers
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
            kernl_weight_train_op = kernl_weight_optimizer.apply_gradients(kernl_cropped_weight_grads_and_vars)    ##################
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
        tf.summary.image('kernl_sensitivity_tensor',tf.expand_dims(tf.expand_dims(trainables[kernl_sensitivity_tensor_index],axis=0),axis=-1))
        tf.summary.image('kernl_sensitivity_tensor_grad',tf.expand_dims(tf.expand_dims(kernl_sensitivity_tensor_update,axis=0),axis=-1))
        tf.summary.image('kernl_temporal_filter',tf.expand_dims(tf.expand_dims(tf.expand_dims(trainables[kernl_temporal_filter_index],axis=0),axis=-1),axis=-1))
        tf.summary.image('kernl_temporal_filter_grad',tf.expand_dims(tf.expand_dims(tf.expand_dims(kernl_temporal_filter_update,axis=0),axis=-1),axis=-1))
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
        tf.summary.scalar('kernl_accuracy',kernl_accuracy)
        # kernl kernel and output weight
        tf.summary.image('kernl_kernel',tf.expand_dims(tf.expand_dims(trainables[kernl_kernel_index],axis=0),axis=-1))
        tf.summary.image('kernl_kernel_grad',tf.expand_dims(tf.expand_dims(kernl_weight_update,axis=0),axis=-1))
        tf.summary.image('kernl_output_weight',tf.expand_dims(tf.expand_dims(trainables[kernl_output_weight_index],axis=0),axis=-1))
        tf.summary.image('kernl_output_weight_grad',tf.expand_dims(tf.expand_dims(kernl_grad_cost_to_output_layer[0],axis=0),axis=-1))
        tf.summary.scalar('loss',kernl_loss_output_prediction+1e-10)
        tf.summary.scalar('accuracy',accuracy+1e-10)
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
# write graph into tensorboard
tb_writer = tf.summary.FileWriter(log_dir,graph)
# run a training session
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        kernl_state_train, kernl_tensor_loss=sess.run([kernl_tensor_train_op,kernl_loss_state_prediction],feed_dict={X:batch_x, Y:batch_y})
        kernl_weight_train, kernl_loss,kernl_accu=sess.run([kernl_weight_train_op,kernl_loss_output_prediction,accuracy],feed_dict={X:batch_x, Y:batch_y})
        kernl_tensor_merged_summary=sess.run(kernl_tensor_merged_summary_op,feed_dict={X:batch_x, Y:batch_y})
        kernl_weight_merged_summary=sess.run(kernl_weight_merged_summary_op,feed_dict={X:batch_x, Y:batch_y})
        #
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            tb_writer.add_summary(kernl_tensor_merged_summary, global_step=step)
            tb_writer.add_summary(kernl_weight_merged_summary, global_step=step)
            loss, acc = sess.run([kernl_loss_output_prediction, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    save_path = saver.save(sess, log_dir+"/model.ckpt", write_meta_graph=True)
    print("Model saved in path: %s" % save_path)
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
