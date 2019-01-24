"""Module implementing Kernal_RNN Cells.

This module provides a copy of kernel RNN cells.
Eghbal Hosseini - 2019-01-23


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

_KERNEL_COEF_NAME= "temporal_kernel_coeff"
_SENSITIVITY_TENSOR_NAME= "sensitivity_tensor"


## expand dimensions for incoming recurrent and input activations for multiplication with current acivation
def _tensor_expand_dim(x,y,output_size):
    """input - x : a 2D tensor with batch x n
    y is a 2D with size batch x m
    outputs is 3D tensor with size batch x n x n and batch x n x m
    """
    shape_x=x.get_shape()
    shape_y=y.get_shape()
    #y=tf.cast(y,tf.float32)
    # define a matrix for removing the diagonal in recurrent spikes
    diag_zero= lambda:tf.subtract(tf.constant(1.0,shape=[shape_x[1],shape_x[1]]),
                                                    tf.eye(output_size))
    x_diag_fixer = tf.Variable(initial_value=diag_zero, dtype=tf.float32,name='diag_fixer', trainable=False)
    # expand x
    x_temp=tf.reshape(tf.tile(x,[1,output_size]),[-1,output_size,shape_x[1]])
    # remove diagonal
    x_expand=tf.multiply(x_temp,x_diag_fixer)
    # expand y
    y_expand=tf.reshape(tf.tile(y,[1,output_size]),[-1,output_size,shape_y[1]])
    return x_expand, y_expand


def _gaussian_noise_perturbation(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return tf.multiply(input_layer,0) + noise


def _kernel_coeff_initializer(shape,dtype=None,partition_info=None,verify_shape=None, max_val=1):
    if dtype is None:
        dtype=tf.float32

    return tf.random_uniform(shape,0,max_val,dtype=dtype)


_KernelRNNStateTuple = collections.namedtuple("KernelRNNStateTuple", ("h","h_hat","Theta", "Gamma","input_trace","recurrent_trace","delta_sensitivity"))
_KernelRNNOutputTuple = collections.namedtuple("KernelRNNOutputTuple", ("h","h_hat","Theta","Gamma", "input_trace","recurrent_trace","delta_sensitivity"))

class KernelRNNStateTuple(_KernelRNNStateTuple):
  """Tuple used by kernel RNN Cells for `state_variables `.
  Stores 9 elements: `(h, h_hat, Theta, Gamma, input_trace,recurrent_trace, sensitivty_tensor, kernel_coeff`, in that order.
  always is used for this type of cell
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, h_hat,Theta , Gamma, input_trace,recurrent_trace, delta_sensitivity ) = self
    if h.dtype != h_hat.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(h.dtype), str(h_hat.dtype)))
    return h_hat.dtype


class KernelRNNOutputTuple(_KernelRNNOutputTuple):
  """Tuple used by kernel Cells for output state.
  Stores 7 elements: `(h,h_hat, Theta, Gamma, input_trace, recurrent_trace)`,
  Only used when `output_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, h_hat,Theta , Gamma, input_trace,recurrent_trace, delta_sensitivity) = self
    if h.dtype != h_hat.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(h.dtype), str(h_hat.dtype)))
    return h_hat.dtype


class KernelRNNCell(tf.contrib.rnn.RNNCell):
    """Kernel recurrent neural network Cell
      Args:
        num_units: int, The number of units in the cell.
        activation: Nonlinearity to use.  Default: `Relu`.
        eligibility_kernel: kernel funtion to use for elibility
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
                 noise_std=1.0,
                 activation=None,
                 reuse=None,
                 eligibility_kernel=None,
                 state_is_tuple=True,
                 output_is_tuple=False,
                 batch_KeRNL=True,
                 sensitivity_initializer=None,
                 kernel_coeff_initializer=None,
                 kernel_initializer=None,
                 bias_initializer=None):

        super(KernelRNNCell,self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._num_inputs= num_inputs
        self._time_steps= time_steps
        self._activation = activation or math_ops.tanh
        self._eligibility_kernel = eligibility_kernel or math_ops.exp
        self._noise_std=noise_std
        self._linear = None
        self._state_is_tuple=state_is_tuple
        self._output_is_tuple= output_is_tuple
        self._batch_KeRNL=batch_KeRNL
        self._tensor_expand_dim=_tensor_expand_dim
        self._gaussian_noise_perturbation=_gaussian_noise_perturbation
        self._sensitivity_initializer=sensitivity_initializer
        self._kernel_coeff_initializer=kernel_coeff_initializer
        self._kernel_initializer=kernel_initializer
        self._bias_initializer=bias_initializer

    @property
    # h,h_hat,Theta, Gamma,input_trace,recurrent_trace,sensitivty_tensor,kernel_coeff
    def state_size(self):
        return (KernelRNNStateTuple(self._num_units,
                                    self._num_units,
                                    self._num_units,
                                    self._num_units,
                                    np.array([self._num_units,self._num_inputs]),
                                    np.array([self._num_units,self._num_units]),
                                    self._num_units)
                if self._state_is_tuple else self._num_units)
    @property
    def output_size(self):
        return (KernelRNNOutputTuple(self._num_units,
                                    self._num_units,
                                    self._num_units,
                                    self._num_units,
                                    np.array([self._num_units,self._num_inputs]),
                                    np.array([self._num_units,self._num_units]),
                                    self._num_units)
                if self._output_is_tuple else self._num_units)

    # call function routine
    def call(self, inputs, state):
        """Kernel RNN cell (KernelRNN).
        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `KernelRNNStateTuple` of state tensors, shaped as following
            h:                   [batch_size x self.state_size]`
            h_hat:               [batch_size x self.state_size]`
            Theta:               [batch_size x self.state_size]`
            Gamma:               [batch_size x self.state_size]`
            input_trace          [batch_size x self.state_size x self.input_size]`
            recurrent_trace      [batch_size x self.state_size x self.state_size]`
            delta_sensitivity    [batch_size x self.state_size]`
        Returns:
          A pair containing the new output, and the new state as SNNStateTuple
          output has the following shape
            h:                   [batch_size x self.state_size]`
            h_hat:               [batch_size x self.state_size]`
            Theta:               [batch_size x self.state_size]`
            Gamma:               [batch_size x self.state_size]`
            input_trace          [batch_size x self.state_size x self.input_size]`
            recurrent_trace      [batch_size x self.state_size x self.state_size]`
            delta_sensitivity    [batch_size x self.state_size]`
        """
        # initialize kernel_coeff
        scope=vs.get_variable_scope()
        if self._kernel_coeff_initializer is None:
            kernel_initializer=init_ops.constant_initializer(1/self._time_steps,dtype=tf.float32)
        else:
            kernel_initializer=self._kernel_coeff_initializer
        with vs.variable_scope(scope,initializer=kernel_initializer) as kernel_scope:
            kernel_coeff=tf.get_variable(_KERNEL_COEF_NAME,shape=[self._num_units],dtype=tf.float32,trainable=True)

        # initialize Sensitivity_tensor
        scope=vs.get_variable_scope()
        if self._sensitivity_initializer is None:
            sensitivity_initializer=init_ops.truncated_normal_initializer
        else:
            sensitivity_initializer=self._sensitivity_initializer
        with vs.variable_scope(scope,initializer=sensitivity_initializer) as sensitivity_scope:
            sensitivity_tensor=tf.get_variable(_SENSITIVITY_TENSOR_NAME,shape=[self._num_units,self._num_units],dtype=tf.float32,trainable=True)



        if self._state_is_tuple:
            h, h_hat, Theta, Gamma, input_trace, recurrent_trace, delta_sensitivity= state
        else:
            logging.error("State has to be tuple for this type of cell")

        if self._linear is None:
            self._linear = _Linear([inputs, h], self._num_units, True)
        psi_new=self._gaussian_noise_perturbation(h,self._noise_std)

        # propagate data forward
        h_new=self._activation(self._linear([inputs,h]))

        # propagate noisy data forward
        h_hat_update=tf.add(h_hat,psi_new)
        h_hat_new= self._activation(self._linear([inputs, h_hat_update]))
        # TODO : check if weights get reused

        # integrate over perturbations
        Theta_new=tf.add(tf.multiply(self._eligibility_kernel(-kernel_coeff),
                              Theta),psi_new)

        # derivative of perturbation w.r.t to kernel_coeff
        Gamma_new=tf.subtract(tf.multiply(self._eligibility_kernel(-kernel_coeff),
                              Gamma),
                             tf.multiply(self._eligibility_kernel(-kernel_coeff),
                              Theta))

        # update elgibility traces for input and recurrent units
        recurrent_expand,inputs_expand=self._tensor_expand_dim(h,inputs,self._num_units)
        g_new=self._linear([inputs,h])
        pre_activation=self._activation(g_new)
        activation_gradients=tf.gradients(pre_activation,g_new)[0] # convert list to a tensor
        gradient_expansion=tf.expand_dims(activation_gradients,axis=-1)
        input_trace_update=tf.multiply(gradient_expansion,inputs_expand)
        recurrent_trace_update=tf.multiply(gradient_expansion,recurrent_expand)

        # calculate error in predicting effect of perturbation
        delta_sensitivity_new=tf.subtract(tf.matmul(Theta_new,tf.transpose(sensitivity_tensor)),
                                         tf.subtract(h_hat_new,h_new)
                                         )
        #logging.warn("%s: input_trace ", input_trace.get_shape())
        kernel_decay=tf.expand_dims(self._eligibility_kernel(-kernel_coeff),axis=-1)
        # update input trace
        input_trace_decay=tf.multiply(kernel_decay,input_trace)
        input_trace_new=tf.add(input_trace_decay,input_trace_update)

        # update recurrent trace
        recurrent_trace_decay=tf.multiply(kernel_decay,recurrent_trace)
        recurrent_trace_new=tf.add(recurrent_trace_decay,recurrent_trace_update)

        # calculate updates for sensitivty_tensor and kernel_coeff
        sensitivity_tensor_update=tf.matmul(delta_sensitivity_new,tf.transpose(delta_sensitivity_new))

        if self._state_is_tuple:
            new_state=KernelRNNStateTuple(h_new,h_hat_new,Theta_new,Gamma_new,input_trace_new,
                                          recurrent_trace_new,delta_sensitivity_new)
        if self._output_is_tuple:
            new_output=KernelRNNOutputTuple(h_new,h_hat_new,Theta_new,Gamma_new,input_trace_new,
                                          recurrent_trace_new,delta_sensitivity_new)
        else:
            new_output=h_new

        return new_output, new_state
