# python libraries
import numpy as np

import collections
import hashlib
import numbers
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
import spiking_cell_bare as spiking_cell
import adding_problem

# uplading mnist data


# Training Parameters
# Training Parameters
weight_learning_rate = 1e-3
training_steps = 4000
batch_size = 25
buffer_size=500
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

# save dir
log_dir = "/om2/user/ehoseini/MyData/KeRNL/logs/bptt_snn_addition_dataset/bptt_snn_add_eta_weight_%1.0e_batch_%1.0e_hum_hidd_%1.0e_gc_%1.0e_steps_%1.0e_run_%s" %(weight_learning_rate,batch_size,num_hidden,grad_clip,training_steps, datetime.now().strftime("%Y%m%d_%H%M"))
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
    W_in=tf.random_normal(new_shape,mean=0,stddev=0.001)
    return tf.concat([W_in,W_rec],axis=0) 

## define KeRNL unit
def bptt_snn_all_states(x):
    with tf.variable_scope('hidden_layer') as scope:
        hidden_layer_cell=spiking_cell.conductance_spike_cell(num_units=num_hidden,output_is_tuple=True,tau_refract=1.0,tau_m=20.0,kernel_initializer=_hinton_identity_initializer)
        output_hidden, states_hidden = tf.nn.dynamic_rnn(hidden_layer_cell, dtype=tf.float32, inputs=x)
    with tf.variable_scope('output_layer') as scope :
        output_layer_cell=spiking_cell.output_spike_cell(num_units=num_output)
        output_voltage, voltage_states=tf.nn.dynamic_rnn(output_layer_cell,dtype=tf.float32,inputs=output_hidden.spike)
    return output_voltage,output_hidden

tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():

    BATCH_SIZE=tf.placeholder(tf.int64)
    X = tf.placeholder("float", [None, time_steps, num_input])
    Y = tf.placeholder("float", [None, num_output])
    # define a dataset
    dataset=tf.data.Dataset.from_tensor_slices((X,Y)).batch(BATCH_SIZE).repeat()
    dataset = dataset.shuffle(buffer_size=buffer_size)
    iter = dataset.make_initializable_iterator()
    inputs,labels =iter.get_next()
    # define a function for extraction of variable names
    bptt_output,bptt_hidden_states=bptt_snn_all_states(inputs)
    trainables=tf.trainable_variables()
    variable_names=[v.name for v in tf.trainable_variables()]
    #
    find_joing_index = lambda x, name_1,name_2 : [a and b for a,b in zip([np.unicode_.find(k.name, name_1)>-1 for k in x] ,[np.unicode_.find(k.name, name_2)>-1 for k in x])].index(True)
    # find trainable parameters for bptt
    with tf.name_scope('bptt_Trainables') as scope:
        bptt_output_weight_index= find_joing_index(trainables,'output_layer','kernel')
        bptt_kernel_index= find_joing_index(trainables,'hidden_layer','kernel')
        bptt_weight_training_indices=np.asarray([bptt_kernel_index,bptt_output_weight_index],dtype=np.int)
        bptt_weight_trainables= [trainables[k] for k in bptt_weight_training_indices]

    with tf.name_scope('bptt_train_weights') as scope:
        bptt_weight_optimizer = tf.train.RMSPropOptimizer(learning_rate=weight_learning_rate)
        bptt_loss_output_prediction=tf.losses.mean_squared_error(labels,bptt_output[:,-1,:])
        bptt_grad_cost_trainables=tf.gradients(bptt_loss_output_prediction,bptt_weight_trainables)
        bptt_weight_grads_and_vars=list(zip(bptt_grad_cost_trainables,bptt_weight_trainables))
        bptt_cropped_weight_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) if  np.unicode_.find(var.name,'output')==-1 else (grad,var) for grad,var in bptt_weight_grads_and_vars]
        bptt_weight_train_op = bptt_weight_optimizer.apply_gradients(bptt_cropped_weight_grads_and_vars)


            ##################
            ## bptt train ####
            ##################

    with tf.name_scope("bptt_evaluate") as scope:
        bptt_loss_cross_validiation=tf.losses.mean_squared_error(labels,bptt_output[:,-1,:])

    with tf.name_scope('cross_validation_summary') as scope:
        tf.summary.scalar('cross_validation_summary',bptt_loss_cross_validiation+1e-10)
        bptt_evaluate_summary_op=tf.summary.merge_all(scope="cross_validation_summary")

                ##################
                # SUMMARIES ######
                ##################

    with tf.name_scope("bptt_weight_summaries") as scope:
        # bptt sensitivity tensor
        tf.summary.histogram('bptt_kernel_grad',bptt_grad_cost_trainables[0]+1e-10)
        tf.summary.histogram('bptt_kernel', bptt_grad_cost_trainables[0]+1e-10)
                    # bptt output weight
        tf.summary.histogram('bptt_output_weight_grad',bptt_grad_cost_trainables[1]+1e-10)
        tf.summary.histogram('bptt_output_weights', bptt_grad_cost_trainables[1]+1e-10)
                    # bptt loss and accuracy
        tf.summary.scalar('bptt_loss_output_prediction',bptt_loss_output_prediction+1e-10)

        # bptt senstivity tensor and temporal filter
        bptt_tensor_merged_summary_op=tf.summary.merge_all(scope="bptt_weight_summaries")

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
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(epochs):
        sess.run(iter.initializer,feed_dict={X: training_x, Y: training_y , BATCH_SIZE: batch_size})
        for step in range(training_steps):
            bptt_weight_train, bptt_loss=sess.run([bptt_weight_train_op,bptt_loss_output_prediction])
            if step % display_step==0:
                bptt_tensor_merged_summary=sess.run(bptt_tensor_merged_summary_op)
                tb_writer.add_summary(bptt_tensor_merged_summary, global_step=epoch*training_steps+step+1)
                print('Epoch: {}, Batch: {} , total batch {}, total trials: {},bptt train Loss: {:.3f}'.format(epoch+1,step + 1,epoch*training_steps+step+1,(epoch*training_steps+step+1)*batch_size, bptt_loss))
        # run test at the end of each epoch
        sess.run(iter.initializer, feed_dict={ X: testing_x, Y: testing_y, BATCH_SIZE: testing_x.shape[0]})
        bptt_test_loss,bptt_evaluate_summary=sess.run([bptt_loss_cross_validiation,bptt_evaluate_summary_op])
        tb_writer.add_summary(bptt_evaluate_summary, global_step=epoch*training_steps+training_steps+1)
        print('cross validation loss {:.3f} at the end of epoch {}'.format(bptt_test_loss,epoch+1))
    print("Optimization Finished!")
    save_path = saver.save(sess, log_dir+"/model.ckpt", global_step=epoch*training_steps+training_steps,write_meta_graph=True)
    print("Model saved in path: %s" % save_path)
