"""Module implementing FA_rnn_cell."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.utils import tf_utils
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
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

_FEEDBACK_VARIABLE_NAME="beta"
_TENSOR_VARIABLE_NAME="kernel"
_BIAS_VARIABLE_NAME="bias"

@tf.custom_gradient
def g_hidden(x,output_size,kernel_initializer):

    shape_x=tf.get_shape(x)
    scope = vs.get_variable_scope()

    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            _TENSOR_VARIABLE_NAME, [output_size,shape_x[1]],
            dtype=dtype)
            #initializer=kernel_initializer)
        beta = vs.get_variable(
                    _FEEDBACK_VARIABLE_NAME, [output_size,shape_x[1]],
                    dtype=dtype,
                    initializer=tf.initializer.random_uniform)
        biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=tf.initializer.zeros_initializer)

    res = math_ops.matmul(x,weights)
    res = nn_ops.bias_add(res, biases)
    def grad(dy):
        dres=weights
        return dyres
    return res, [grad,0,0]

class fa_rnn_cell(tf.contrib.rnn.LayerRNNCell):

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    super(fa_rnn_cell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnRNNTanh for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)

    self._num_units = num_units
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh
    self._g_hidden=_g_hidden
  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units


  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    gate_inputs = self._g_hidden(tf.concatenate([inputs, state], 1),self._num_units)

    output = self._activation(gate_inputs)
    return output, output

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(BasicRNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
