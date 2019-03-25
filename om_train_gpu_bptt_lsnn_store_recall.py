from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import collections
import hashlib
import numbers
from datetime import datetime
from pathlib import Path
import os
import scipy
from scipy import stats
import sys
import poisson_spike_cell
import long_short_term_spike_cell_bare as spiking_cell

##
learn_rate=1e-1
training_steps=200
batch_size=128
display_step=2
num_inputs=1
num_input_units=50
num_hidden_units=80.0
num_hidden_adaptive=20.0
rate=50*1e-3
num_of_trials=batch_size*training_steps
num_time_steps=12
num_spike_inputs=200
num_outputs=1
total_time=2400
tau_beta=400
beta_coeff=1.7
beta_baseline=1
grad_clip=200
epochs=1
# save dir
#log_dir = "/om/user/ehoseini/MyData/KeRNL/logs/bptt_lsnn_v4_seq_mnist/bptt_snn_v4_mnist_eta_W_%1.0e_batch_%1.0e_hum_hidd_%1.0e_gc_%1.0e_steps_%1.0e_epoch_%1.0e_run_%s" %(weight_learning_rate,batch_size,num_hidden,grad_clip,training_steps,epochs, datetime.now().strftime("%Y%m%d_%H%M"))
#log_dir = os.environ['HOME']+"/MyData/KeRNL/logs/bptt_lsnn_store_recall/lsnn_store_recall_eta_W_%1.0e_batch_%1.0e_hum_hidd_%1.0e_gc_%1.0e_steps_%1.0e_epoch_%1.0e_run_%s"%(learn_rate,batch_size,num_hidden_units,grad_clip,training_steps,epochs, datetime.now().strftime("%Y%m%d_%H%M"))
log_dir = "/om/user/ehoseini/MyData/KeRNL/logs/bptt_lsnn_store_recall/lsnn_store_recall_eta_W_%1.0e_batch_%1.0e_hum_hidd_%1.0e_gc_%1.0e_steps_%1.0e_epoch_%1.0e_run_%s"%(learn_rate,batch_size,num_hidden_units,grad_clip,training_steps,epochs, datetime.now().strftime("%Y%m%d_%H%M"))

log_dir
###

# Generate data
template=np.ones([num_of_trials,num_time_steps,200])
store_vec=np.zeros([num_of_trials,num_time_steps,1])
recall_vec=np.zeros([num_of_trials,num_time_steps,1])
value_1=np.random.randint(0,2,[num_of_trials,num_time_steps,1])
value_1_tensor=np.reshape(np.multiply(value_1,template),[num_of_trials,-1,1])

values_0=1-value_1
value_0_tensor=np.reshape(np.multiply(values_0,template),[num_of_trials,-1,1])

store=np.random.randint(0,6,num_of_trials)
n_values = np.max(num_time_steps)
store_vec=np.eye(n_values)[store].reshape(num_of_trials,num_time_steps,1)
store_tensor=np.reshape(np.multiply(store_vec,template),[num_of_trials,-1,1])

recall=np.random.randint(0,6,[1,num_of_trials])
recall=recall+store+1
recall_vec=np.eye(n_values)[recall].reshape(num_of_trials,num_time_steps,1)
recall_tensor=np.reshape(np.multiply(recall_vec,template),[num_of_trials,-1,1])
goal= np.multiply(recall_tensor,np.expand_dims(np.sum(np.multiply(value_1_tensor,store_tensor),axis=1)-
                                               np.sum(np.multiply(value_0_tensor,store_tensor),axis=1),axis=1)/200)
value_0_tensor=np.multiply(value_0_tensor,1-recall_tensor)
value_1_tensor=np.multiply(value_1_tensor,1-recall_tensor)
start_index=np.argmax(np.abs(goal))
tf.reset_default_graph()

tf_value_1_signal=tf.placeholder("float32",[None, num_time_steps*200, num_inputs])
tf_value_0_signal=tf.placeholder("float32",[None, num_time_steps*200, num_inputs])
tf_store_signal=tf.placeholder("float32",[None, num_time_steps*200, num_inputs])
tf_recall_signal=tf.placeholder("float32",[None, num_time_steps*200, num_inputs])

value_1_cell = poisson_spike_cell.poisson_spike_cell(num_units=num_input_units)
value_0_cell = poisson_spike_cell.poisson_spike_cell(num_units=num_input_units)
store_cell = poisson_spike_cell.poisson_spike_cell(num_units=num_input_units)
recall_cell = poisson_spike_cell.poisson_spike_cell(num_units=num_input_units)

outputs_v_1, _ = tf.nn.dynamic_rnn(cell=value_1_cell, dtype=tf.float32, inputs=tf_value_1_signal)
outputs_v_0, _ = tf.nn.dynamic_rnn(cell=value_0_cell, dtype=tf.float32, inputs=tf_value_0_signal)
outputs_store, _ = tf.nn.dynamic_rnn(cell=store_cell, dtype=tf.float32, inputs=tf_store_signal)
outputs_recall, _ = tf.nn.dynamic_rnn(cell=recall_cell, dtype=tf.float32, inputs=tf_recall_signal)

cell_outputs=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_v_1 , output_v_0, output_store, output_recall = sess.run([outputs_v_1,outputs_v_0, outputs_store,outputs_recall],
                                                    feed_dict={tf_value_1_signal:rate*value_1_tensor,
                                                              tf_value_0_signal:rate*value_0_tensor,
                                                              tf_store_signal:rate*store_tensor,
                                                              tf_recall_signal:rate*recall_tensor})
    variables_names =[v.name for v in tf.global_variables()]
    values = sess.run(variables_names)
    for k,v in zip(variables_names, values):
        print(k, v)


SNN_input=np.concatenate([output_v_0,output_v_1,output_store,output_recall],axis=-1)
output=np.concatenate([output_v_0[0,:,:],output_v_1[0,:,:],output_store[0,:,:],output_recall[0,:,:]],axis=1)

## define SNN unit
def _lsnn_weight_initializer(shape,dtype=None,partition_info=None,verify_shape=None, gain=0.5):
    if dtype is None:
        dtype=tf.float32
    #extract second dimension
    W_rec=tf.divide(gain*tf.random_normal(shape=[shape[1],shape[1]],mean=0,stddev=1,dtype=dtype),tf.sqrt(tf.cast(shape[1],tf.float32)))
    new_shape=[shape[0]-shape[1],shape[1]]

    W_in = tf.divide(gain*tf.random_normal(shape=new_shape,mean=0,stddev=1,dtype=dtype),tf.sqrt(tf.cast(new_shape[0],tf.float32)))
    #W_in=tf.random_normal(new_shape,mean=0,stddev=0.001)
    return tf.concat([W_in,W_rec],axis=0)

def _tau_beta_double_initializer(input_shape,dtype=None,partition_info=None,verify_shape=None, gain=1.0):
    if dtype is None:
        dtype=tf.float32
    #extract second dimensio
    adaptive_shape=[1,input_shape[0]]
    non_adaptive_shape=[1,input_shape[1]]
    tau_beta_adaptive=tf.constant(value=1200.0,shape=adaptive_shape,dtype=dtype)
    tau_beta_non_adaptive = tf.constant(value=0,shape=non_adaptive_shape,dtype=dtype)
    #W_in=tf.random_normal(new_shape,mean=0,stddev=0.001)
    return tf.concat([tau_beta_adaptive,tau_beta_non_adaptive],axis=-1)

tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():
    # input to graph
    X_input=tf.placeholder('float',shape=[batch_size,total_time,num_spike_inputs])
    Y_input=tf.placeholder('float',shape=[batch_size,total_time,num_outputs])
    start_ind=tf.placeholder(tf.int32,shape=[batch_size,1])
    #
    start_mat=tf.concat([start_ind,tf.constant(0,shape=[batch_size,1])],axis=1)
    start_mat_dummy=tf.concat([start_ind*0+200,tf.constant(1,shape=[batch_size,1])],axis=1)
    start_mat_unstack=tf.unstack(start_mat,axis=0)
    start_mat_dummy_unstack=tf.unstack(start_mat_dummy,axis=0)
    #
    hidden_layer_cell=spiking_cell.long_short_term_spike_cell(num_units=num_hidden_units,
                                                              num_inputs=num_spike_inputs,
                                                              output_is_tuple=True,
                                                              tau_m=20.0,
                                                              beta_baseline=beta_baseline,
                                                              beta_coeff=beta_coeff,
                                                              num_adaptive=num_hidden_adaptive,
                                                              tau_beta_initializer=_tau_beta_double_initializer
                                                             )
    output_hidden, _ = tf.nn.dynamic_rnn(hidden_layer_cell, dtype=tf.float32, inputs=X_input)

    output_layer_cell=spiking_cell.output_spike_cell(num_units=num_outputs,kernel_initializer=tf.initializers.random_normal)
    output_voltage,_ =tf.nn.dynamic_rnn(output_layer_cell,dtype=tf.float32,inputs=output_hidden.spike)
    #

    output_voltage_unstack=tf.unstack(output_voltage,axis=0)
    temp=list(zip(output_voltage_unstack,start_mat_unstack,start_mat_dummy_unstack))
    t_sliced=[ tf.slice(a,b,c) for a , b,c  in temp ]
    output_sliced_tensor=tf.convert_to_tensor(t_sliced)
    output_sliced_sign=tf.sign(tf.reduce_mean(output_sliced_tensor,axis=1))
    #
    Y_input_unstack=tf.unstack(Y_input,axis=0)
    temp1=list(zip(Y_input_unstack,start_mat_unstack,start_mat_dummy_unstack))
    Y_sliced_tensor=tf.convert_to_tensor([ tf.slice(a,b,c) for a , b,c  in temp1 ])
    output_loss=tf.losses.absolute_difference(labels=tf.reduce_mean(Y_sliced_tensor,axis=1),predictions=output_sliced_sign)
    optimizer=tf.train.AdamOptimizer()
    gradients=optimizer.compute_gradients(output_loss)
    train=optimizer.apply_gradients(gradients)
    #
    trainables=tf.trainable_variables()
    variable_names=[v.name for v in tf.trainable_variables()]

    with tf.name_scope("snn_summaries") as scope:
        tf.summary.scalar('output_loss',output_loss+1e-10)
        merged_summary_op=tf.summary.merge_all(scope="snn_summaries")
    #with tf.name_scope('bptt_snn_evaluate') as scope:
    #    tf.summary.scalar('bptt_snn_accuracy',bptt_snn_accuracy+1e-10)
    #    bptt_snn_evaluate_merged_summary_op=tf.summary.merge_all(scope="bptt_snn_evaluate")
    init=tf.global_variables_initializer()
    saver = tf.train.Saver()


###################################################

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
    for step in range(1,training_steps+1):
        batch_id=np.random.randint(0,num_of_trials,[batch_size])
        batch_x =SNN_input[batch_id,:,:]
        batch_y =goal[batch_id,:,:]
        batch_goal=np.argmax(np.abs(batch_y),axis=1)
        training , loss,merged_summary = sess.run([train,output_loss,merged_summary_op], feed_dict={X_input: batch_x, Y_input:batch_y, start_ind:batch_goal})
        #
        tb_writer.add_summary(merged_summary, global_step=step)
        if step % display_step==0 :
            # get batch loss and accuracy
            print('Step: {},IRNN train Loss: {:.3f}'.format(step + 1, loss))
    print("Optimization Finished!")
    ave_path = saver.save(sess, log_dir+"/model.ckpt",write_meta_graph=True)
