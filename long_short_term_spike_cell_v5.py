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


def _append_input_spike(S,input_spikes):
    """input -
    S : a 4D tensor with batch n x d ,
    input: a 2D tensor batch x n,
    """
    S_update=tf.roll(S,shift=1,axis=-1)
    _,S_cut=tf.split(S_update,[1,-1],-1)
    S_out=tf.concat([tf.expand_dims(input_spikes,axis=-1),S_cut],axis=-1)
    # add spikes to S_cut
    return S_out


########################################
########################################
def _tensor_linear(args,output_size,delay_tensor,bias=False,bias_initializer=None,kernel_initializer=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

  # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 3:
            raise ValueError("tensor linear is expecting 3D arguments: %s" % shapes)
        if shape.dims[-1].value is None:
            raise ValueError("linear expects shape[2] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[-1]))
        else:
            total_arg_size += shape.dims[1].value
    logging.warn("%s: total_arg_size ", total_arg_size)
    dtype = [a.dtype for a in args][0]
  # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            _TENSOR_VARIABLE_NAME, [output_size,total_arg_size],
            dtype=dtype,
            initializer=kernel_initializer)

    logging.warn("%s: weights ", weights)
    logging.warn("%s: delay_tensor ", delay_tensor)
    weights_delayed=tf.einsum('un,unv->unv',weights,delay_tensor)
    logging.warn("%s: weights_delayed ", weights_delayed)
    if len(args) == 1:
        res= tf.einsum('unv,pnv->up',args[0],weights_delayed)
    else:
        logging.warn("%s: input ", array_ops.concat(args, 1))
        res = tf.einsum('unv,pnv->up',array_ops.concat(args, 1),weights_delayed)
    logging.warn("%s: res ", res)
    #if not bias:
    return res
    #with vs.variable_scope(outer_scope) as inner_scope:
    #    inner_scope.set_partitioner(None)
    #if bias_initializer is None:
    #    bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
    #biases = vs.get_variable( _BIAS_VARIABLE_NAME, [output_size],dtype=dtype,initializer=bias_initializer)

    #return nn_ops.bias_add(res, biases)
##############################################

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
###### functions used in the model ########
###########################################


#class _delay_one_hot_initializer(tf.keras.initializers.Initializer):
#  """Initializer that generates tensors initialized to 1."""
#  def __init__(self,max_val=2, dtype=dtypes.float32):
#    self.dtype = dtypes.as_dtype(dtype)
#    self.max_val=max_val
 # def __call__(self, shape, dtype=None, partition_info=None):
#    if dtype is None:
#      dtype = self.dtype

#    return tf.one_hot(tf.random_uniform(shape,0,self.max_val+1,dtype=tf.int32),depth=self.max_val+1)

#  def get_config(self):
#    return {"dtype": self.dtype.name}

def _delay_one_hot_initializer(shape,dtype=None,partition_info=None,verify_shape=None, max_val=2):
    if dtype is None:
        dtype=tf.float32
    return tf.one_hot(tf.random_uniform(shape,1,max_val+1,dtype=tf.int32),depth=max_val+1)

########################################
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
########################################

def _spike_activation_fcn(v,I_in,Beta,t_reset,dt,tau_m,R_mem,tau_refract,_calculate_crossing):
    alpha=tf.exp(tf.negative(tf.divide(dt,tau_m)))
    eligilible_update=tf.cast(tf.less(t_reset,dt),tf.float32)
    v_update=tf.add(tf.multiply(alpha,v),tf.multiply(tf.multiply(1-alpha,R_mem),I_in))
    v_norm=tf.divide(tf.subtract(v_update,Beta),Beta)
    spike=tf.multiply(eligilible_update,_calculate_crossing(v_norm))
    v_new=tf.subtract(v_update,tf.multiply(v_update,spike))
    t_reset_new=tf.add(tf.multiply(spike,tau_refract),tf.clip_by_value(tf.subtract(t_reset,dt),0.0,100))

    return v_new, spike, t_reset_new
###########################################
###### variables used in the model ########
###########################################

_DELAY_TENSOR_NAME="delay_tensor"
_TENSOR_VARIABLE_NAME='kernel'


_LSNNStateTuple = collections.namedtuple("LSNNStateTuple", ("v_mem","spike","S_in","S_rec","b_threshold","t_reset"))
_LSNNOutputTuple = collections.namedtuple("LSNNOutputTuple", ("v_mem","spike"))

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
    (v_mem,spike) = self
    if v_mem.dtype != spike.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s"
                      (str(v_mem.dtype), str(spike.dtype)))
    return spike.dtype

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

    if self._linear is None:
        self._linear = _Linear([inputs],self._num_units,False,kernel_initializer=self._kernel_initializer)
    alpha=tf.exp(tf.negative(tf.divide(self.dt,self.tau_m)))
    state_new=tf.add(tf.multiply(alpha,state),tf.multiply(1-alpha,self._linear([inputs])))
    ## return variables
    return state_new, state_new


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


##########################################
_LSNNStateTuple = collections.namedtuple("LSNNStateTuple", ("v_mem","spike","S_in","S_rec","b_threshold","t_reset"))
_LSNNOutputTuple = collections.namedtuple("LSNNOutputTuple", ("v_mem","spike"))

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

class LSNNOutputTuple(_LSNNOutputTuple):
  """Tuple used by SNN Cells for output state.
  Stores six elements: `(v_mem,spike,t_reset,I_syn)`,
  Only used when `output_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (v_mem,spike) = self


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
               tau_refract=1.0,
               tau_beta=1.0,
               dt=1.0,
               beta_baseline=1.0,
               beta_coeff=0.1,
               activation=None,
               reuse=None,
               max_delay=3,
               delay_initializer=None,
               kernel_initializer=None,
               bias_initializer=None,
               state_is_tuple=True,
               output_is_tuple=False):
    super(long_short_term_spike_cell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._num_inputs = num_inputs
    self._max_delay=max_delay
    self.tau_m=tau_m
    self.v_theta=v_theta
    self.v_reset=v_reset
    self.R_mem=R_mem
    self.tau_refract=tau_refract
    self.tau_beta=tau_beta
    self.dt=dt
    self.beta_baseline=beta_baseline
    self.beta_coeff=beta_coeff
    self.kernel_factor=np.divide(self.dt,np.multiply(self.R_mem,np.sqrt(self._num_units)))
    self._linear = _tensor_linear
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._delay_initializer=delay_initializer
    self._state_is_tuple= state_is_tuple
    self._output_is_tuple= output_is_tuple
    self._calculate_crossing= _calcualte_crossings
    self._append_input_spike=_append_input_spike
    self._delay_one_hot_initializer=_delay_one_hot_initializer

# create intializers
    if kernel_initializer is None:
        self._kernel_initializer=tf.initializers.random_normal(mean=0,stddev=1/self.kernel_factor)
    else:
        self._kernel_initializer=kernel_initializer

    if delay_initializer is None:
        self._delay_initializer= _delay_one_hot_initializer
    else:
        self._delay_initializer=delay_initializer


  @property
  def state_size(self):
    return (LSNNStateTuple(self._num_units,
                          self._num_units,
                          np.array([self._num_inputs,self._max_delay]),
                          np.array([self._num_units,self._max_delay]),
                          self._num_units,
                          self._num_units) if self._state_is_tuple else self._num_units)
  @property
  def output_size(self):
    return (LSNNOutputTuple(self._num_units,self._num_units) if self._output_is_tuple else self._num_units)

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
    if self._state_is_tuple:
        v_mem,spike,S_in,S_rec,b_threshold,t_reset = state
    else:
        logging.error("this cell only accept state as tuple ", self)
    scope=vs.get_variable_scope()
    with vs.variable_scope(scope,initializer= self._delay_initializer) as _delay_scope:
        delay_tensor=tf.get_variable(_DELAY_TENSOR_NAME,shape=[self._num_units,self._num_units+self._num_inputs],dtype=tf.float32,trainable=False)
    logging.warn("%s: S_in ", S_in)
    logging.warn("%s: S_rec ", S_rec)
    I_syn=self._linear([S_in,S_rec],self._num_units,delay_tensor,kernel_initializer=self._kernel_initializer)
    Beta= self.beta_baseline + tf.multiply(self.beta_coeff,b_threshold)
    v_mem_new, spike_new, t_reset_new= _spike_activation_fcn(v_mem,I_syn,Beta,t_reset,self.dt,self.tau_m,self.R_mem,self.tau_refract,self._calculate_crossing)

    rho=tf.exp(tf.negative(tf.divide(self.dt,self.tau_beta)))
    b_threshold_new = tf.add(tf.scalar_mul(rho,b_threshold), tf.scalar_mul(1-rho,spike_new))

    S_in_new = self._append_input_spike(S_in,inputs)
    S_rec_new = self._append_input_spike(S_rec,spike_new)
    # return variables
    if self._state_is_tuple:
        new_state = LSNNStateTuple( v_mem_new,spike_new, S_in_new,S_rec_new, b_threshold_new, t_reset_new )
    if self._output_is_tuple:
        new_output = LSNNOutputTuple( v_mem_new,spike_new )
    else:
        new_output = spike_new
    return new_output, new_state
