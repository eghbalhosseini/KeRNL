############################################
############################################
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




##############################################

def _calculate_poisson_spikes(x,rate):
    """input - x : a 2D tensor with batch x n ex 10*1
    outputs a tensor with size batch x output_size, where outputsize is twice the size of thresholds_size
    """
    shape_x=tf.shape(x)
    #
    x_aux=tf.random_uniform(shape=shape_x,dtype=tf.float32)
    res_out=1 - tf.maximum(0.0, tf.sign(x_aux-rate))

    return res_out


###########################################
#### define input spiking cell ############
###########################################

## define LSNNOutcell
class poisson_spike_cell(tf.contrib.rnn.RNNCell):
    def __init__(self,
                num_units=50,
                reuse=None):
        super(poisson_spike_cell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._calculate_poisson_spikes = _calculate_poisson_spikes
    @property
    def state_size(self):
        return  self._num_units
    @property
    def output_size(self):
        return  self._num_units

    def call(self, inputs, state):
        new_state=self._calculate_poisson_spikes(state,inputs)
        return new_state, new_state
