"""Module implementing kernl_rnn_cell.

This module provides a copy of kernel RNN cells.
Eghbal Hosseini - 2019-02-21


TODO :
    - batch kernel flag for batch version, and online version
"""
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

_TEMPORAL_FILTER_NAME= "temporal_filter"
_SENSITIVITY_TENSOR_NAME= "sensitivity_tensor"


def heaviside(x):
    return 1 - tf.maximum(0.0, tf.sign(-x))


def tanh_diff(x):
    return 1-tf.math.square(tf.tanh(x))

def _gaussian_noise_perturbation(input_layer, magnitude=1e-3):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=magnitude, dtype=tf.float32)
    return tf.multiply(input_layer,0) + noise

def _uniform_noise_perturbation(input_layer,magnitude=1e-3):
    noise = tf.random_uniform(shape=tf.shape(input_layer), minval=0.000, maxval=magnitude, dtype=tf.float32)
    return tf.multiply(input_layer,0) + noise

def _temporal_filter_initializer(shape,dtype=None,partition_info=None,verify_shape=None, max_val=1):
    if dtype is None:
        dtype=tf.float32
    return tf.random_uniform(shape,0,max_val,dtype=dtype)


_kernl_state_tuple = collections.namedtuple("kernl_state_tuple", ("h","h_hat","Theta", "Gamma","eligibility_trace","bias_trace"))
_kernl_output_tuple = collections.namedtuple("kernl_output_tuple", ("h","h_hat","Theta","Gamma", "eligibility_trace","bias_trace","psi"))

class kernl_state_tuple(_kernl_state_tuple):
  """Tuple used by kernel RNN Cells for `state_variables `.
  Stores 5 elements: `(h, h_hat, Theta, Gamma, eligibility_trace`, in that order.
  always is used for this type of cell
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, h_hat,Theta , Gamma, eligibility_trace,bias_trace ) = self
    if h.dtype != h_hat.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(h.dtype), str(h_hat.dtype)))
    return h_hat.dtype


class kernl_output_tuple(_kernl_output_tuple):
  """Tuple used by kernel Cells for output state.
  Stores 5 elements: `(h,h_hat, Theta, Gamma, eligibility_trace)`,
  Only used when `output_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, h_hat,Theta , Gamma, input_trace,eligibility_trace,bias_trace,psi) = self
    if h.dtype != h_hat.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(h.dtype), str(h_hat.dtype)))
    return h_hat.dtype

class kernl_rnn_cell(tf.contrib.rnn.RNNCell):
    """Kernel recurrent neural network Cell
      Args:
        num_units: int, The number of units in the cell.
        activation: Nonlinearity to use.  Default: `Relu`.
        eligibility_filter: kernel funtion to use for elibility
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.  If not `True`, and the existing scope already has
         the given variables, an error is raised.
        kernel_initializer: (optional) The initializer to use for the weight and
        projection matrices.
        bias_initializer: (optional) The initializer to use for the bias.
    """
    def __init__(self,
                 num_units,
                 num_inputs,
                 time_steps=1,
                 noise_param=0.5,
                 activation="relu",
                 reuse=None,
                 eligibility_filter=None,
                 state_is_tuple=True,
                 output_is_tuple=False,
                 batch_kernl=True,
                 sensitivity_initializer=None,
                 temporal_filter_initializer=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name="kernl_rnn_cell"):

        super(kernl_rnn_cell,self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._time_steps = time_steps

        self._activation = activation or tf.nn.relu
        self._temporal_filter = eligibility_filter or math_ops.exp
        self._noise_param=noise_param
        self._linear = None
        self._state_is_tuple=state_is_tuple
        self._output_is_tuple= output_is_tuple
        self._batch_kernl=batch_kernl
        self._perturbation=_gaussian_noise_perturbation
        self._name=name
        self._activation_diff=None

        # sensitivty_tensor
        if sensitivity_initializer is None:
            self._sensitivity_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32)
        else:
            self._sensitivity_initializer=sensitivity_initializer

        # temporal_filter
        if temporal_filter_initializer is None:
            self._temporal_filter_initializer=tf.initializers.random_uniform(maxval=2/self._time_steps)
        else:
            self._temporal_filter_initializer=temporal_filter_initializer

        # _kernel_initializer
        if kernel_initializer is None:
            self._kernel_initializer=tf.random_normal_initializer(stddev=0.1)
        else:
            self._kernel_initializer=kernel_initializer

        # bias initializer
        if bias_initializer is None:
            self._bias_initializer=tf.initializers.zeros()
        else:
            self._bias_initializer=bias_initializer

        # gradient of output function
        if activation == "relu":
            self._activation=tf.nn.relu
            self._activation_diff=heaviside

        elif activation == "tanh":
            self._activation=tf.nn.tanh
            self._activation_diff=tanh_diff
        else:
            logging.error("only relu and tanh are implemented at this time")

    @property
    # h,h_hat,Theta, Gamma,eligibility_trace
    def state_size(self):
        return (kernl_state_tuple(self._num_units,
                                self._num_units,
                                self._num_units,
                                self._num_units,
                                np.array([self._num_units,self._num_units+self._num_inputs]),
                                self._num_units)
                if self._state_is_tuple else self._num_units)
    @property
    def output_size(self):
        return (kernl_output_tuple(self._num_units,
                                 self._num_units,
                                 self._num_units,
                                 self._num_units,
                                 np.array([self._num_units,self._num_units+self._num_inputs]),
                                 self._num_units,
                                 self._num_units)
                if self._output_is_tuple else self._num_units)

    # call function routine
    def call(self, inputs, state):
        """Kernel RNN cell (kernl).
        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `kernl_state_tuple` of state tensors, shaped as following
            h:                   [batch_size x self.state_size]`
            h_hat:               [batch_size x self.state_size]`
            Theta:               [batch_size x self.state_size]`
            Gamma:               [batch_size x self.state_size]`
            eligibility_trace    [batch_size x self.state_size x (self._state_size, self.input_size)]`
            bias_trace           [batch_size x self.state_size]`
        Returns:
          A pair containing the new output, and the new state as SNNStateTuple
          output has the following shape
            h:                   [batch_size x self.state_size]`
            h_hat:               [batch_size x self.state_size]`
            Theta:               [batch_size x self.state_size]`
            Gamma:               [batch_size x self.state_size]`
            eligibility_trace    [batch_size x self.state_size x (self._state_size, self.input_size)]`
            bias_trace           [batch_size x self.state_size]`
            psi                  [batch_size x self.state_size]`
        """
        # initialize temporal_filter
        scope=vs.get_variable_scope()
        with vs.variable_scope(scope,initializer=self._temporal_filter_initializer) as temporal_filter_scope:
            temporal_filter=tf.get_variable(_TEMPORAL_FILTER_NAME,shape=[self._num_units],dtype=tf.float32,trainable=True)

        # initialize Sensitivity_tensor
        scope=vs.get_variable_scope()
        with vs.variable_scope(scope,initializer= self._sensitivity_initializer) as sensitivity_scope:
            sensitivity_tensor=tf.get_variable(_SENSITIVITY_TENSOR_NAME,shape=[self._num_units,self._num_units],dtype=tf.float32,trainable=True)

        if self._state_is_tuple:
            h, h_hat, Theta, Gamma, eligibility_trace, bias_trace= state
        else:
            logging.error("State has to be tuple for this type of cell")
        # define the linear operation (g) for inputs and weights, note that the
        # input size is [batch, input_size+state_size] and output is
        # [batch, state_size] , the weight in the function is [input_size+state_size, state_size]
        # the matrix operation is [input,recurrent]* [w_input,w_recurrent]^T
        if self._linear is None:
            self._linear = _Linear([inputs, h],self._num_units,True,kernel_initializer=self._kernel_initializer,bias_initializer=self._bias_initializer)
        # create noise for current timestep.
        psi_new=self._perturbation(h,magnitude=self._noise_param)
        # propagate data forward
        h_new=self._activation(self._linear([inputs, h]),name='update_h')
        # propagate noisy data forward
        h_hat_update=tf.add(h_hat,psi_new,name='add_noise')
        h_hat_new= self._activation(self._linear([inputs, h_hat_update]),name='update_h_hat')
        # integrate over perturbations
        Theta_new=tf.add(tf.multiply(self._temporal_filter(-temporal_filter),Theta),psi_new,'update_theta')
        # derivative of perturbation w.r.t to temporal_filter
        Gamma_new=tf.subtract(tf.multiply(self._temporal_filter(-temporal_filter),Gamma),
                              tf.multiply(self._temporal_filter(-temporal_filter),Theta),name='update_gamma')

        # update elgibility traces
        g_new=self._linear([inputs,h]) # size : batch*num_units
        pre_activation=self._activation(g_new) # size : batch*num_units
        activation_gradients=self._activation_diff(g_new)
        kernel_decay=tf.expand_dims(self._temporal_filter(-temporal_filter),axis=-1)
        eligibility_trace_new=tf.add(tf.multiply(kernel_decay,eligibility_trace),
                                     tf.einsum("un,uv->unv",activation_gradients,array_ops.concat([inputs,h],1)))
        #
        bias_trace_new=tf.add(tf.multiply(self._temporal_filter(-temporal_filter),bias_trace),activation_gradients)

        if self._state_is_tuple:
            new_state=kernl_state_tuple(h_new,h_hat_new,Theta_new,Gamma_new,eligibility_trace_new,bias_trace_new)
        if self._output_is_tuple:
            new_output=kernl_output_tuple(h_new,h_hat_new,Theta_new,Gamma_new,eligibility_trace_new,bias_trace_new,psi_new)
        else:
            new_output=h_new

        return new_output, new_state
