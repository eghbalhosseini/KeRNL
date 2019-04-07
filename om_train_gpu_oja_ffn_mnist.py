from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
from datetime import datetime
from pathlib import Path

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


import sys


# Setup the Model Parameters
# Setup the Model Parameters
INPUT_SIZE=784
HIDDEN_SIZE=1000
TOTAL_SIZE=INPUT_SIZE+HIDDEN_SIZE
OUTPUT_SIZE = 10
START_LEARNING_RATE=1e-3
BATCH_SIZE=25
NUM_TRAINING_STEPS = 500
EPOCHS=5
TEST_LENGTH=125
DISPLAY_STEP=25
weight_learning_rate=1e-3

log_dir = "/om/user/ehoseini/MyData/KeRNL/logs/ffn/bp_tanh_mnist_eta_weight_%1.0e_batch_%1.0e_hum_hidd_%1.0e_steps_%1.0e_run_%s" %(weight_learning_rate,BATCH_SIZE,HIDDEN_SIZE,NUM_TRAINING_STEPS, datetime.now().strftime("%Y%m%d_%H%M"))
log_dir

def drelu(x):
    return 1 - tf.maximum(0.0, tf.sign(-x))


def dtanh(x):
    return(1-tf.multiply(tf.nn.tanh(x),tf.nn.tanh(x)))


## define KeRNL unit
tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():

    # define weights and inputs to the network
    X = tf.placeholder('float', shape=[None, INPUT_SIZE])
    Y = tf.placeholder('float', shape=[None, OUTPUT_SIZE])
    initializer = tf.random_normal_initializer(stddev=0.1)
    # define a function for extraction of variable names
     # Hidden Layer Variables
    W_1 = tf.get_variable("Hidden_W", shape=[INPUT_SIZE, HIDDEN_SIZE], initializer=tf.contrib.layers.xavier_initializer())
    b_1 = tf.get_variable("Hidden_b", shape=[HIDDEN_SIZE], initializer=initializer)
  # output layer variables
    W_2 = tf.get_variable("Output_W", shape=[HIDDEN_SIZE, OUTPUT_SIZE], initializer=initializer)
    b_2 = tf.get_variable("Output_b", shape=[OUTPUT_SIZE], initializer=initializer)
    # return weight
    B=tf.get_variable('B',shape=[OUTPUT_SIZE,HIDDEN_SIZE],initializer=tf.initializers.random_uniform(minval=-0.5,maxval=0.5))
    trainables=[W_1,b_1,W_2,b_2,B]

    variable_names=[v.name for v in tf.trainable_variables()]
    #
    #define transformation from input to output
  # Hidden Layer Transformation
    g_hidden=tf.matmul(X, W_1) + b_1
    hidden = tf.nn.tanh(g_hidden)
  # Output Layer Transformation
    output = tf.matmul(hidden, W_2) + b_2


            ##################
            ## kernl train ####
            ##################
    with tf.name_scope("kernl_train") as scope:
        loss = tf.losses.mean_squared_error(Y, output)
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(output, 1))
        accuracy = 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        optimizer  = tf.train.AdamOptimizer(learning_rate=weight_learning_rate)
  # compute and apply gradiants
        dW_2=tf.reduce_mean(tf.transpose(tf.einsum('uv,un->uvn',tf.subtract(output,Y),(hidden))),axis=-1)
        db_2=tf.reduce_mean(tf.subtract(output,Y),axis=0)
        dg_hidden=dtanh(g_hidden)
        dg_hidden_diag=tf.linalg.diag(dg_hidden)
        dW_1=tf.transpose(tf.reduce_mean(tf.einsum('uv,ug->uvg',tf.einsum('uv,uvg->ug',tf.einsum('un,nv->uv',tf.subtract(output,Y),tf.transpose(W_2)),dg_hidden_diag),X),axis=0))
        db_1=tf.transpose(tf.reduce_mean(tf.einsum('uv,ug->ug',tf.einsum('un,nv->uv',tf.subtract(output,Y),tf.transpose(W_2)),dg_hidden),axis=0))
        #dW_1=tf.transpose(tf.reduce_mean(tf.einsum('uv,ug->uvg',tf.einsum('uv,uvg->ug',tf.einsum('un,nv->uv',tf.subtract(output,Y),B),dg_hidden_diag),X),axis=0))
        #db_1=tf.transpose(tf.reduce_mean(tf.einsum('uv,ug->ug',tf.einsum('un,nv->uv',tf.subtract(output,Y),B),dg_hidden),axis=0))
        # gradient for B
        dB=tf.negative(tf.reduce_mean(tf.einsum('uv,uz->uvz',output,tf.subtract(hidden,tf.einsum('uv,vz->uz',output,0*B))),axis=0))
        new_ffn_grads=list(zip([dW_1,db_1,dW_2,db_2,dB],trainables))
        ffn_train_op=optimizer.apply_gradients(new_ffn_grads)


    with tf.name_scope("evaluate") as scope:
        kernl_loss_cross_validiation=tf.losses.mean_squared_error(Y,output)
        kernl_correct_pred_cross_val=tf.equal(tf.argmax(Y, 1), tf.argmax(output, 1))
        kernl_accu_cross_validation=100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('cross_validation_summary') as scope:
        tf.summary.scalar('cross_validation_loss',kernl_loss_cross_validiation+1e-10)
        tf.summary.scalar('cross_validation_accu',kernl_accu_cross_validation+1e-10)

        kernl_evaluate_summary_op=tf.summary.merge_all(scope="cross_validation_summary")

                ##################
                # SUMMARIES ######
                ##################

                    # kernl kernel
    tf.summary.histogram('kernl_hidd_W',W_1+1e-10)
    tf.summary.histogram('return_B',B+1e-10)
                    # kernl output weight
    tf.summary.histogram('kernl_output_W',W_2+1e-10)
                    # kernl output bias
                    # kernl loss and accuracy
    tf.summary.scalar('loss_output_prediction',loss+1e-10)
    tf.summary.scalar('accuracy',accuracy+1e-10)
    merged_summary_op=tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
###################################################

Path(log_dir).mkdir(exist_ok=True, parents=True)
filelist = [ f for f in os.listdir(log_dir) if f.endswith(".local") ]
for f in filelist:
    os.remove(os.path.join(log_dir, f))
####################################################

tb_writer = tf.summary.FileWriter(log_dir,graph)
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(EPOCHS):
        for step in range(NUM_TRAINING_STEPS):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            train_minimize,bptt_loss,accu, summary  = sess.run([ffn_train_op,loss,accuracy,merged_summary_op], {X: batch_x, Y: batch_y})
            tb_writer.add_summary(summary, global_step=epoch*NUM_TRAINING_STEPS+step+1)
            if step % DISPLAY_STEP==0 :
                print('Epoch: {}, Batch: {}, train Loss: {:.3f}, accuracy : {:.3f}'.format(epoch+1,step + 1, bptt_loss,accu))
    print("Optimization Finished!")
