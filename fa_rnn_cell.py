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
from tensorflow.python.ops import gen_math_ops
_FEEDBACK_VARIABLE_NAME="beta"
_WEIGHTS_VARIABLE_NAME="kernel"
_BIAS_VARIABLE_NAME="bias"

@tf.custom_gradient
def _beta_mat_mul(a,b):
    total_arg_x_size=0
    total_arg_y_size=0
    shapes = [s.get_shape() for s in [b]]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("tensor linear is expecting 2D arguments: %s" % shapes)
        if shape.dims[-1].value is None:
            raise ValueError("linear expects shape[2] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[-1]))
        else:
            total_arg_y_size += shape.dims[1].value
            total_arg_x_size += shape.dims[0].value

    dtype = [s.dtype for s in [b]][0]
    scope = vs.get_variable_scope()
    #with vs.variable_scope(scope) as outer_scope:
    #    beta = vs.get_variable(
    #        _FEEDBACK_VARIABLE_NAME, [total_arg_x_size, total_arg_y_size], dtype=dtype,initializer=tf.initializers.random_uniform,trainable=False
    #        ,use_resource=True)
    beta=tf.random.normal([total_arg_x_size,total_arg_y_size],mean=0,stddev=0.1,dtype=dtype,seed=None,name='beta')
    #beta=tf.random.uniform([total_arg_x_size,total_arg_y_size],minval=-.2,maxval=.2,dtype=dtype,seed=None,name='beta')
    #beta=tf.scalar_mul(0.0000000,b)

    res=gen_math_ops.mat_mul(a,b)
    def grad(dy):
        grad_a = gen_math_ops.mat_mul(dy, beta, transpose_b=True)
        grad_b = gen_math_ops.mat_mul(a, dy, transpose_a=True)
        return  [grad_a , grad_b]
    return res, grad

def _fa_linear(args, output_size, bias, bias_initializer=None,kernel_initializer=None):
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]
  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]
  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype,initializer=kernel_initializer)
    if len(args) == 1:
      #res = math_ops.matmul(args[0], weights)
      res= _beta_mat_mul(args[0], weights)
    else:
      #res = math_ops.matmul(array_ops.concat(args, 1), weights)
      res = _beta_mat_mul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable( _BIAS_VARIABLE_NAME, [output_size],dtype=dtype,initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)

class fa_rnn_cell(tf.contrib.rnn.RNNCell):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 state_is_tuple=False,
                 output_is_tuple=False,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name="fa_rnn_cell"):
        super(fa_rnn_cell,self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.nn.relu
        self._linear = _fa_linear
        self._state_is_tuple=state_is_tuple
        self._output_is_tuple= output_is_tuple
        self._name=name
        # kernel initializer
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
    def state_size(self):
        return self._num_units
    @property
    def output_size(self):
        return self._num_units
    # call function routine
    def call(self, inputs, state):
        hidden_new=self._linear([inputs, state],self._num_units,True,kernel_initializer=self._kernel_initializer,bias_initializer=self._bias_initializer)
        state_new=self._activation(hidden_new)
        return state_new, state_new
