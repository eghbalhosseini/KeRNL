
"""Module implementing kernl spiking cell.

This module provides a copy of different types of kernl spiking cells.
Eghbal Hosseini - 2019-02-13
version 4.0 : (2019-02-20)
    - the euler method is introduced to caluclate the membrane and input dynamics

version 3.0 : (2019-02-19)
    - the noise perturbation and state update are changed.
    - rearranged how the sequence of updating happen in the while loop
        - first V_mem is updated, based on the t_reset,
        - then spike is calculated based on the beta_current
        - next beta is updated for spikes
version 2.0 : (2019-02-18)
    - the noise perturbation and state update are updated.

version 1.0 : (2019-02-13)
    - 4 types of cell are introduced,


version 0.0 : (2019-02-07)
    - 3 types of cell are introduced,
        - spiking input cell : the probability of firing depends on the input
        - conductance spiking cell: a LIF neuron with spike adaptation.
        - output spiking cell : input spikes are integrated into output weights by addition of their
          weights to cell membrane voltage

        - context input cell: the output of the cell is usually 0 except when a temporal threshold is crossed,
                        after which the output becomes 1.

TODO :
    - implement input synaptic dynamics.
    - fixed error in xavier initialization of weights

TODO :
    - implement input synaptic dynamics.
    - make the threshold value for _calculate_spikes normalized

"""
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
###########################################
###### variables used in the model ########
###########################################

_TEMPORAL_FILTER_NAME= "temporal_filter"
_SENSITIVITY_TENSOR_NAME= "sensitivity_tensor"

###########################################
###### functions used in the model ########
###########################################
# perturbation function
def _gaussian_noise_perturbation(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.000, stddev=std, dtype=tf.float32)
    return tf.multiply(input_layer,0) + noise

def _temporal_filter_initializer(shape,dtype=None,partition_info=None,verify_shape=None, max_val=1):
    if dtype is None:
        dtype=tf.float32

    return tf.random_uniform(shape,0,max_val,dtype=dtype)
# crossing function
@tf.custom_gradient
def _calcualte_crossings(x):
    """input :x : a 2D tensor with batch x n
    outputs a tensor with the same size as x
    and values of 0 or 1 depending on comparison between
    x and threshold"""
    dtype=x.dtype
    res=tf.greater_equal(x,0.0)
    def grad(dy):
        # calculate 1-|x|
        temp=1-tf.abs(x)
        dyres=tf.scalar_mul(0.3,tf.maximum(temp,0.0))
        return dyres
    return tf.cast(res,dtype=dtype), grad

def _calculate_prob_spikes(x,threshold):
    """input - x : a 2D tensor with batch x n ex 10*1
    outputs a tensor with size batch x output_size, where outputsize is twice the size of thresholds_size
    """
    shape_x=tf.shape(x)
    #
    logging.warn("%s: Please use float ", shape_x[0])
    x_aux=tf.random_uniform(shape=shape_x,dtype=tf.float32)
    logging.warn("%s: Please use float ", x_aux.get_shape())
    res_out=tf.cast(tf.divide(tf.negative(tf.sign(x_aux-threshold)-1),2),tf.float32)

    return res_out
###########################################
###### definition of tuples for cells #####
###########################################

_LSNNStateTuple = collections.namedtuple("LSNNStateTuple", ("v_mem","v_mem_hat","spike","S","Theta","b_threshold","b_threshold_hat","t_reset","t_reset_hat","eligibility_trace"))
_LSNNOutputTuple = collections.namedtuple("LSNNOutputTuple", ("v_mem","v_mem_hat","spike","S","Theta","b_threshold","t_reset","eligibility_trace","psi"))


class LSNNStateTuple(_LSNNStateTuple):
  """Tuple used by LSNN Cells for `state_variables `, and output state.
  Stores five elements: `(v_mem,spike, t_reset, I_syn)`, in that order. Where `v_mem` is the hidden state
  , spike is output, `S_rec` and 'S_in' are spike history, and t_reset refractory history.
  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (v_mem,v_mem_hat,spike,S,Theta,b_threshold,b_threshold_hat,t_reset,t_reset_hat,eligibility_trace) = self
    if v_mem.dtype != spike.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(v_mem.dtype), str(spike.dtype)))
    return spike.dtype
class LSNNOutputTuple(_LSNNOutputTuple):
  """Tuple used by SNN Cells for output state.
  Stores six elements: `(v_mem,spike,t_reset,I_syn)`,
  Only used when `output_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (v_mem,v_mem_hat,spike,S,Theta,b_threshold,t_reset,eligibility_trace,psi) = self
    if v_mem.dtype != spike.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(v_mem.dtype), str(spike.dtype)))
    return spike.dtype

###########################################
#### define conductance spiking cell ######
###########################################


class kernl_spike_Cell(tf.contrib.rnn.RNNCell):

  """ conductance_spike_Cell
  Args:
  """
  def __init__(self,
               num_units,
               num_inputs,
               time_steps,
               tau_m=20.0,
               v_theta=1.0,
               v_reset=0.0,
               R_mem=2.0,
               tau_s=5.0,
               tau_refract=1.0,
               tau_beta=1.0,
               dt=1.0,
               beta_baseline=1.0,
               beta_coeff=.0,
               gradient_gamma=0.3,
               noise_param=0.1,
               activation=None,
               eligibility_filter=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               sensitivity_initializer=None,
               temporal_filter_initializer=None,
               state_is_tuple=True,
               output_is_tuple=False):

    super(kernl_spike_Cell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._num_inputs= num_inputs
    self.tau_m=tau_m
    self.v_theta=v_theta
    self.v_reset=v_reset
    self.R_mem=R_mem
    self.tau_s=tau_s
    self.tau_refract=tau_refract
    self.tau_beta=tau_beta
    self._time_steps=time_steps
    self.dt=dt
    self.beta_baseline=beta_baseline
    self.beta_coeff=beta_coeff
    self.gradient_gamma=gradient_gamma
    self._noise_param=noise_param
    self.kernel_factor=np.divide(self.dt,np.multiply(self.R_mem,np.sqrt(self._num_units)))
    self._linear = None
    self._eligibility_filter = eligibility_filter or math_ops.exp
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._sensitivity_initializer=sensitivity_initializer
    self._temporal_filter_initializer=temporal_filter_initializer
    self._state_is_tuple= state_is_tuple
    self._output_is_tuple= output_is_tuple
    self._calculate_crossing= _calcualte_crossings
    self._noise_perturbation=_gaussian_noise_perturbation

    # create intializers
    if kernel_initializer is None:
        self._kernel_initializer=tf.initializers.random_normal(mean=0,stddev=1/self.kernel_factor)
    else:
        self._kernel_initializer=kernel_initializer

    # sensitivty_tensor
    if sensitivity_initializer is None:
        self._sensitivity_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32)
    else:
        self._sensitivity_initializer=sensitivity_initializer

    if temporal_filter_initializer is None:
        self._temporal_filter_initializer=tf.initializers.random_uniform(maxval=1/self._time_steps)
    else:
        self._temporal_filter_initializer=temporal_filter_initializer

  @property
  def state_size(self):
    return (LSNNStateTuple(self._num_units,
                          self._num_units,
                          self._num_units,
                          self._num_units,
                          self._num_units,
                          self._num_units,
                          self._num_units,
                          self._num_units,
                          self._num_units,
                          np.array([self._num_units,self._num_units+self._num_inputs])) if self._state_is_tuple else self._num_units)


  @property
  def output_size(self):
    return (LSNNOutputTuple(self._num_units,
                          self._num_units,
                          self._num_units,
                          self._num_units,
                          self._num_units,
                          self._num_units,
                          self._num_units,
                          np.array([self._num_units,self._num_units+self._num_inputs]),
                          self._num_units) if self._output_is_tuple else self._num_units)


  def call(self, inputs, state):
    """ (conductance_spike_Cell call function).
    Args:
        inputs: `2-D` tensor with shape `[batch_size x input_size]`.
        state: An `LSSNStateTuple` of state tensors, shaped as following
          v_mem:            [batch_size x self.state_size]`
          v_mem_hat:        [batch_size x self.state_size]`
          spike:            [batch_size x self.state_size]`
          S:                [batch_size x self.state_size]`
          Theta             [batch_size x self.state_size]`
          b_threshold       [batch_size x self.state_size]`
          b_threshold_hat   [batch_size x self.state_size]`
          t_reset           [batch_size x self.state_size]`
          t_reset_hat       [batch_size x self.state_size]`
          eligibility_trace [batch_size x self.state_size x (self.state_size+self.input_size)]
    Returns:
      A pair containing the new output, and the new state as SNNStateTuple
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
        v_mem,v_mem_hat,spike,S,Theta,b_threshold,b_threshold_hat,t_reset,t_reset_hat, eligibility_trace = state
    else:
        logging.error("this cell only accept state as tuple ", self)

    if self._linear is None:
        self._linear = _Linear([inputs,S],self._num_units,False,kernel_initializer=self._kernel_initializer)


    # calculate noise for current time_step

    with tf.name_scope('update_v_mem') as scope:
        # inputs
        I_syn=self._linear([inputs,S])
        # constant
        kernel_decay=tf.expand_dims(self._eligibility_filter(-temporal_filter),axis=-1)
        Beta= self.beta_baseline + tf.multiply(self.beta_coeff,b_threshold)
        eligilible_update=tf.cast(tf.less(t_reset,self.dt),tf.float32)
        v_mem_delta=tf.multiply(tf.multiply(tf.divide(tf.subtract(tf.scalar_mul(self.R_mem, I_syn),v_mem),self.tau_m),self.dt),eligilible_update)
        v_mem_update=tf.add(v_mem,v_mem_delta)
        v_mem_norm=tf.divide(tf.subtract(v_mem_update,Beta),Beta)
        spike_new=self._calculate_crossing(v_mem_norm)
        v_reseting=tf.multiply(v_mem_update,spike_new)
        v_mem_new=tf.subtract(v_mem_update,v_reseting)
        b_threshold_new=tf.add(tf.multiply(self.dt,tf.divide(b_threshold,self.tau_beta)),spike_new)
        t_update=tf.clip_by_value(tf.subtract(t_reset,self.dt),0.0,100)
        t_reset_new=tf.add(tf.multiply(spike_new,self.tau_refract),t_update)
        S_update=tf.subtract(S,tf.divide(tf.scalar_mul(self.dt,S),self.tau_s))
        S_new=tf.add(S_update,spike_new)
        activation_gradients=S_new
        eligibility_trace_update=tf.einsum("un,uv->unv",activation_gradients,array_ops.concat([inputs,S_new],1))
        eligibility_trace_new=tf.add(tf.multiply(kernel_decay,eligibility_trace),eligibility_trace_update)

    with tf.name_scope('update_v_mem_hat') as scope:
        psi_new=self._noise_perturbation(v_mem,self._noise_param)
        Theta_new=tf.add(tf.multiply(self._eligibility_filter(-temporal_filter),Theta),psi_new)
        I_syn_hat=self._linear([inputs,S+psi_new])
        eligilible_update_hat=tf.cast(tf.less(t_reset_hat,self.dt),tf.float32)
        Beta_hat= self.beta_baseline + tf.multiply(self.beta_coeff,b_threshold_hat)
        v_mem_delta_hat=tf.multiply(tf.multiply(tf.divide(tf.subtract(tf.scalar_mul(self.R_mem, I_syn_hat),v_mem_hat),self.tau_m),self.dt),eligilible_update_hat)
        v_mem_update_hat=tf.add(v_mem_hat,v_mem_delta_hat)
        v_mem_norm_hat=tf.divide(tf.subtract(v_mem_update_hat,Beta_hat),Beta_hat)
        spike_new_hat=self._calculate_crossing(v_mem_norm_hat)
        v_reseting_hat=tf.multiply(v_mem_update_hat,spike_new_hat)
        v_mem_new_hat=tf.subtract(v_mem_update_hat,v_reseting_hat)
        b_threshold_new_hat=tf.add(tf.multiply(self.dt,tf.divide(b_threshold_hat,self.tau_beta)),spike_new_hat)
        t_update_hat=tf.clip_by_value(tf.subtract(t_reset_hat,self.dt),0.0,100)
        t_reset_new_hat=tf.add(tf.multiply(spike_new_hat,self.tau_refract),t_update_hat)

    if self._state_is_tuple:
        new_state = LSNNStateTuple( v_mem_new,v_mem_new_hat,spike_new, S_new,Theta_new,b_threshold_new,b_threshold_new_hat,t_reset_new,t_reset_new_hat,eligibility_trace_new )
    if self._output_is_tuple:
        new_output = LSNNOutputTuple( v_mem_new,v_mem_new_hat,spike_new, S_new,Theta_new,b_threshold_new,t_reset_new,eligibility_trace_new,psi_new )
    else:
        new_output = spike_new
    return new_output, new_state

###########################################
#### define output spiking cell ###########
###########################################

class output_spike_cell(tf.contrib.rnn.RNNCell):
  """LSNN Cell
  Args:
    num_units:
  """
  def __init__(self,
               num_units,
               tau_m=10.0,
               dt=1.0,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):

    super(output_spike_cell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self.tau_m=tau_m
    self.dt=dt
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._linear = None

  @property
  def state_size(self):
    return  self._num_units

  @property
  def output_size(self):
    return  self._num_units

  def call(self, inputs, state):
    """ (LSNN).
    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: An `SNNStateTuple` of state tensors, shaped as following
              `[batch_size x self.state_size]`
    Returns:
      A pair containing the new output, and the new state as SNNStateTuple
    """
    v_mem = state

    if self._linear is None:
        self._linear = _Linear([inputs],self._num_units,False,
                                    kernel_initializer=self._kernel_initializer,
                                    bias_initializer=self._bias_initializer)

    # calculate new Isyn = W*S
    I_syn_new=self._linear([inputs])

    ## update membrane potential

    # calculate factor for updating
    alpha=tf.exp(tf.negative(tf.divide(self.dt,self.tau_m)))
        # update voltage
    v_mem_new=tf.add(tf.scalar_mul(alpha,v_mem),I_syn_new)

    ## return variables
    new_state =  v_mem_new
    new_output = v_mem_new

    return new_output, new_state


###########################################
#### define input spiking cell ############
###########################################

## define LSNNOutcell
class input_spike_cell(tf.contrib.rnn.RNNCell):
    def __init__(self,
               num_units=80,
               reuse=None):
        super(input_spike_cell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._calculate_prob_spikes = _calculate_prob_spikes
    @property
    def state_size(self):
        return  self._num_units
    @property
    def output_size(self):
        return  self._num_units

    def call(self, inputs, state):
        spike_state = state
        # calculate new Isyn = W*S
        new_spikes=self._calculate_prob_spikes(spike_state,inputs)
        new_state =  new_spikes
        new_output = new_spikes
        return new_output, new_state



###########################################
#### define context input cell ############
###########################################

class context_input_spike_cell(tf.contrib.rnn.RNNCell):
    def __init__(self,
               num_units=1,
               reuse=None,
               context_switch=784.0,
               kernel_initializer=None,
               bias_initializer=None):
        super(context_input_spike_cell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._calculate_prob_spikes = _calculate_prob_spikes
        self._context_switch=context_switch

    @property
    def state_size(self):
        return  self._num_units
    @property
    def output_size(self):
        return  self._num_units

    def call(self, inputs, state):
        # calculate new Isyn = W*S
        new_state=tf.add(tf.cast(state,tf.float32),tf.cast(inputs,tf.float32))
        new_output = tf.cast(tf.greater(new_state,self._context_switch),tf.float32)
        return new_output, new_state
