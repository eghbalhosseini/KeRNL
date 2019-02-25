tf.reset_default_graph()
graph=tf.Graph()
with graph.as_default():
     with tf.variable_scope('bptt_output',initializer=tf.initializers.random_normal()) as scope:
        rnn_weights = tf.get_variable(shape=[num_hidden, num_output],name='output_weight')
        rnn_biases = tf.get_variable(shape=[num_output],name='output_addition')
    # define weights and inputs to the network
    X = tf.placeholder("float", [None, timesteps, num_input])
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
        bptt_prediction = tf.nn.softmax(bptt_output)
        bptt_correct_pred = tf.equal(tf.argmax(bptt_prediction, 1), tf.argmax(Y, 1))
        bptt_accuracy = tf.reduce_mean(tf.cast(bptt_correct_pred, tf.float32))
        # define optimizer
        bptt_weight_optimizer = tf.train.RMSPropOptimizer(learning_rate=weight_learning_rate)
        bptt_grads=tf.gradients(bptt_loss_output_prediction,bptt_weight_trainables)
        bptt_weight_grads_and_vars=list(zip(bptt_grads,bptt_weight_trainables))
        # Apply gradient Clipping to recurrent weights
        bptt_cropped_weight_grads_and_vars=[(tf.clip_by_norm(grad, grad_clip),var) if  np.unicode_.find(var.name,'output')==-1 else (grad,var) for grad,var in bptt_weight_grads_and_vars]
        # apply gradients
        bptt_weight_train_op = bptt_weight_optimizer.apply_gradients(bptt_cropped_weight_grads_and_vars)


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
          tf.summary.histogram('bptt_output_addition_grad',bptt_grads[3])
          tf.summary.histogram('bptt_output_addition', bptt_weight_trainables[3]+1e-10)
          # bptt loss and accuracy
          tf.summary.scalar('bptt_loss_output_prediction',bptt_loss_output_prediction)
          tf.summary.scalar('bptt_accuracy',bptt_accuracy)
          # bptt kernel and ouput matrix
          tf.summary.image('bptt_kernel_matrix',tf.expand_dims(tf.expand_dims(bptt_weight_trainables[0],axis=0),axis=-1))
          tf.summary.image('bptt_kernel_grad',tf.expand_dims(tf.expand_dims(bptt_grads[0],axis=0),axis=-1))
          tf.summary.image('bptt_output_matrix',tf.expand_dims(tf.expand_dims(bptt_weight_trainables[2],axis=0),axis=-1))
          tf.summary.image('bptt_output_matrix_grad',tf.expand_dims(tf.expand_dims(bptt_grads[2],axis=0),axis=-1))
          bptt_merged_summary_op=tf.summary.merge_all(scope="bptt_summaries")


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
