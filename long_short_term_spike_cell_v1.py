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

_SYNAPTIC_DELAY_NAME="synaptic_delay"
_INPUT_DELAY_NAME="input_delay"


###########################################
###### functions used in the model ########
###########################################

def _delay_initializer(shape,dtype=None,partition_info=None,verify_shape=None, max_val=5):
    if dtype is None:
        dtype=tf.float32
    return tf.cast(tf.random_uniform(shape,1,max_val,dtype=tf.int32),dtype)
    #return tf.cast(2*tf.ones(shape,dtype=tf.int32),dtype)

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

def _get_input_spike(S,spike,delay_tensor,depth):
    """input - x : a 3D tensor with batch x delay x n , and a 2D tensor batch x n
    outputs a tensor with size batch x output_size, where outputsize is twice the size of thresholds_size
    """
    # roll S and get new input,
    S_update=tf.roll(S,shift=-1,axis=1)
    new_input=S_update[:,0,:]
    synaptic_shape=tf.shape(S)
    # add new spikes to S
    correction_mat=tf.expand_dims(tf.concat([tf.ones([synaptic_shape[1]-1,synaptic_shape[2]],tf.float32),tf.zeros([1,synaptic_shape[2]],dtype=tf.float32)],axis=0),axis=0)
    S_cut=tf.multiply(S_update,correction_mat)
    # add spikes to S_cut
    spike_tensor=tf.multiply(tf.expand_dims(tf.cast(spike,tf.float32),axis=-2),tf.one_hot(tf.cast(tf.multiply(delay_tensor,spike),tf.int32),depth=depth+1,axis=1))
    # update S and return output
    S_out=tf.add(S_cut,spike_tensor)
    return S_out, new_input

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
_LSNNStateTuple = collections.namedtuple("LSNNStateTuple", ("v_mem","spike","S_in","S_rec","b_threshold","t_reset"))
_LSNNOutputTuple = collections.namedtuple("LSNNOutputTuple", ("v_mem","spike","S_in","S_rec","b_threshold","t_reset"))

class LSNNStateTuple(_LSNNStateTuple):
  """Tuple used by LSNN Cells for `state_variables `, and output state.
  Stores five elements: `(v_mem,spike, t_reset, I_syn)`, in that order. Where `v_mem` is the hidden state
  , spike is output, `S_rec` and 'S_in' are spike history, and t_reset refractory history.
  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (v_mem,spike,S_in,S_rec,b_threshold,t_reset) = self
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
    (v_mem,spike,S_in,S_rec,b_threshold,t_reset) = self
    if v_mem.dtype != spike.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(v_mem.dtype), str(spike.dtype)))
    return spike.dtype

###########################################
### define long_short_term_spike_cell #####
###########################################


class long_short_term_spike_cell(tf.contrib.rnn.RNNCell):
  """ long_short_term_spike_Cell
  Args:
  """
  def __init__(self,
               num_units,
               num_inputs,
               tau_m=20.0,
               v_theta=1.0,
               v_reset=0.0,
               R_mem=2.0,
               tau_s=5.0,
               tau_refract=1.0,
               tau_beta=1.0,
               dt=1.0,
               beta_baseline=1.0,
               beta_coeff=0.1,
               activation=None,
               reuse=None,
               max_input_delay=3,
               max_synapse_delay=3,
               input_delay_initializer=None,
               synaptic_delay_initializer=None,
               kernel_initializer=None,
               bias_initializer=None,
               state_is_tuple=True,
               output_is_tuple=False):
    super(long_short_term_spike_cell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._num_inputs = num_inputs
    self.max_input_delay=max_input_delay
    self.max_synapse_delay=max_synapse_delay
    self.tau_m=tau_m
    self.v_theta=v_theta
    self.v_reset=v_reset
    self.R_mem=R_mem
    self.tau_s=tau_s
    self.tau_refract=tau_refract
    self.tau_beta=tau_beta
    self.dt=dt
    self.beta_baseline=beta_baseline
    self.beta_coeff=beta_coeff
    self.kernel_factor=np.divide(self.dt,np.multiply(self.R_mem,np.sqrt(self._num_units)))
    self._linear = None
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._synaptic_delay_initializer=synaptic_delay_initializer
    self._input_delay_initializer=input_delay_initializer
    self._state_is_tuple= state_is_tuple
    self._output_is_tuple= output_is_tuple
    self._calculate_crossing= _calcualte_crossings
    self._get_input_spike=_get_input_spike
    self._delay_initializer=_delay_initializer

# create intializers
    if kernel_initializer is None:
        self._kernel_initializer=tf.initializers.random_normal(mean=0,stddev=1/self.kernel_factor)
    else:
        self._kernel_initializer=kernel_initializer

    if synaptic_delay_initializer is None:
        self._synaptic_delay_initializer= self._delay_initializer
    else:
        self._synaptic_delay_initializer=synaptic_delay_initializer

    if input_delay_initializer is None:
        self._input_delay_initializer= self._delay_initializer
    else:
        self._input_delay_initializer=input_delay_initializer



  @property
  def state_size(self):
    return (LSNNStateTuple(self._num_units,
                          self._num_units,
                          np.array([self.max_input_delay+1,self._num_inputs]),
                          np.array([self.max_synapse_delay+1,self._num_units]),
                          self._num_units,
                          self._num_units) if self._state_is_tuple else self._num_units)
  @property
  def output_size(self):
    return (LSNNOutputTuple(self._num_units,
                          self._num_units,
                          np.array([self.max_input_delay+1,self._num_inputs]),
                          np.array([self.max_synapse_delay+1,self._num_units]),
                          self._num_units,
                          self._num_units) if self._output_is_tuple else self._num_units)

  def call(self, inputs, state):
    """ (conductance_spike_Cell call function).
    Args:
        inputs: `2-D` tensor with shape `[batch_size x input_size]`.
        state: An `LSSNStateTuple` of state tensors, shaped as following
          v_mem:            [batch_size x self.state_size]`
          spike:            [batch_size x self.state_size]`
          S:                [self._num_units+self._num_inputs,self.synaptic_delay[-1]]`
          b_threshold       [batch_size x self.state_size]`
          t_reset           [batch_size x self.state_size]`
    Returns:
      A pair containing the new output, and the new state as SNNStateTuple
    """
    scope=vs.get_variable_scope()
    with vs.variable_scope(scope,initializer= self._synaptic_delay_initializer) as synaptic_delay_scope:
        synaptic_delay_tensor=tf.get_variable(_SYNAPTIC_DELAY_NAME,shape=[self._num_units],dtype=tf.float32,trainable=False)

    scope=vs.get_variable_scope()
    with vs.variable_scope(scope,initializer= self._input_delay_initializer) as _input_delay_initializer:
        input_delay_tensor=tf.get_variable(_INPUT_DELAY_NAME,shape=[self._num_inputs],dtype=tf.float32,trainable=False)


    if self._state_is_tuple:
        v_mem,spike,S_in,S_rec,b_threshold,t_reset = state
    else:
        logging.error("this cell only accept state as tuple ", self)
    # determine the spike which gets integrated

    S_in_new, I_in = self._get_input_spike(S_in,inputs,input_delay_tensor,self.max_input_delay)
    S_rec_new, I_rec = self._get_input_spike(S_rec,spike,synaptic_delay_tensor,self.max_synapse_delay)

    if self._linear is None:
        self._linear = _Linear([I_in,I_rec],self._num_units,False,
                                    kernel_initializer=self._kernel_initializer,
                                    bias_initializer=self._bias_initializer)

    # compute outputs and states
    I_syn=self._linear([I_in,I_rec])
    alpha=tf.exp(tf.negative(tf.divide(self.dt,self.tau_m)))
    rho=tf.exp(tf.negative(tf.divide(self.dt,self.tau_beta)))
    Beta= self.beta_baseline + tf.multiply(self.beta_coeff,b_threshold)
    eligilible_update=tf.cast(tf.less(t_reset,self.dt),tf.float32)
    not_eligilible_update=1-eligilible_update
    # modify alpha so that only affect neurons that are beyond their refractory period
    alpha_vec=tf.scalar_mul(alpha,eligilible_update)+not_eligilible_update
    v_mem_update=tf.add(tf.multiply(alpha_vec,v_mem),tf.multiply(tf.multiply(1-alpha_vec,self.R_mem),I_syn))
    v_mem_norm=tf.divide(tf.subtract(v_mem_update,Beta),Beta)
    spike_new=self._calculate_crossing(v_mem_norm)
    v_reseting=tf.multiply(v_mem_update,spike_new)
    v_mem_new=tf.subtract(v_mem_update,v_reseting)
    b_threshold_new = tf.add(tf.scalar_mul(rho,b_threshold), tf.scalar_mul(1-rho,spike))
    t_update=tf.clip_by_value(tf.subtract(t_reset,self.dt),0.0,100)
    t_reset_new=tf.add(tf.multiply(spike_new,self.tau_refract),t_update)
    # return variables
    if self._state_is_tuple:
        new_state = LSNNStateTuple( v_mem_new,spike_new, S_in_new,S_rec_new, b_threshold_new, t_reset_new )
    if self._output_is_tuple:
        new_output = LSNNOutputTuple( v_mem_new,spike_new, S_in_new,S_rec_new, b_threshold_new, t_reset_new )
    else:
        new_output = spike_new
    return new_output, new_state


class output_spike_cell(tf.contrib.rnn.RNNCell):
  """LSNN Cell
  Args:
    num_units:
  """
  def __init__(self,
               num_units,
               tau_m=20.0,
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
        self._linear = _Linear([inputs],self._num_units,False,kernel_initializer=self._kernel_initializer)

    # calculate new Isyn = W*S

    I_syn_new=self._linear([inputs])
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
