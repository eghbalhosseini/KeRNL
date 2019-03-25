
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import collections
import hashlib
import numbers

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

_FEEDBACK_VARIABLE_NAME="beta"
_TENSOR_VARIABLE_NAME="kernel"
_BIAS_VARIABLE_NAME="bias"

class fa_rnn_cell(tf.contrib.rnn.RNNCell):

    def __init__(self,
                 num_units,
                 num_inputs,
                 activation=None,
                 reuse=None,
                 state_is_tuple=True,
                 output_is_tuple=False,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name="fa_rnn_cell"):

        super(fa_rnn_cell,self).__init__(_reuse=reuse)
        self._num_units = num_units


        self._activation = activation or tf.nn.relu
        self._linear = None
        self._state_is_tuple=state_is_tuple
        self._output_is_tuple= output_is_tuple
        self._name=name


        if kernel_initializer is None:
            self._kernel_initializer=tf.initializers.identity()
        else:
            self._kernel_initializer=kernel_initializer

        # bias initializer
        if bias_initializer is None:
            self._bias_initializer=tf.initializers.zeros()
        else:
            self._bias_initializer=bias_initializer

    @property
    # h,h_hat,Theta, Gamma,eligibility_trace
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    # call function routine
    def call(self, inputs, state):

        # initialize temporal_filter
        #scope=vs.get_variable_scope()
        #with vs.variable_scope(scope,initializer=self._temporal_filter_initializer) as temporal_filter_scope:
        #    temporal_filter=tf.get_variable(_TEMPORAL_FILTER_NAME,shape=[self._num_units],dtype=tf.float32,trainable=True)
        if self._linear is None:
            self._linear = _Linear([inputs, state],self._num_units,True,kernel_initializer=self._kernel_initializer,bias_initializer=self._bias_initializer)

        # propagate data forward
        state_new=self._activation(self._linear([inputs, h]),name='update_h')



        return state_new, state_new
