import numpy as np
import matplotlib.pyplot as plt
import collections
import hashlib
import numbers
import matplotlib.cm as cm
from sys import getsizeof
import sys
from datetime import datetime
from pathlib import Path
import os
from pandas import DataFrame
from IPython.display import HTML
import itertools


# tensorflow and its dependencies
import tensorflow as tf
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
## user defined modules
# kernel rnn cell
import keRNL_cell_v1
################################################
# logic for getting the variables from 1 system argument
# each dictionary represent the set of values for variable
# on each call an iterator is build and a
training_steps_dict={"A":500,"B":2000,"C":5000}
batch_size_dict={"A":50,"B":100,"C":200}
num_hidden_dict={"A":100,"B":200,"C":400}
grad_clip_dict={"A":10,"B":100,"C":200}
#
num_of_variables=4
# create an iterator and use it to determine the values for parameters
variable_combinations=list(itertools.product('ABC', repeat=num_of_variables))
# use input system arg to determine what element to use
variable_condition=variable_combinations[int(sys.argv[1])-1]
# determine the value for each variable
training_steps=training_steps_dict[variable_condition[0]]
batch_size=batch_size_dict[variable_condition[1]]
num_hidden=num_hidden_dict[variable_condition[2]]
grad_clip=grad_clip_dict[variable_condition[3]]

print('training_steps: {}, batch_size: { }, num_hidden: { }, grad_clip: { }'.format(training_steps, batch_size, num_hidden, grad_clip))
