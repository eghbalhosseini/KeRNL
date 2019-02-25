""" code for generating sample for addting problem in RNN ,
source : https://github.com/batzner/indrnn/blob/master/examples/addition_rnn.py
"""

def get_batch():
  """Generate the adding problem dataset"""
  # Build the first sequence
  add_values = np.random.rand(BATCH_SIZE, TIME_STEPS)

  # Build the second sequence with one 1 in each half and 0s otherwise
  add_indices = np.zeros_like(add_values)
  half = int(TIME_STEPS / 2)
  for i in range(BATCH_SIZE):
    first_half = np.random.randint(half)
    second_half = np.random.randint(half, TIME_STEPS)
    add_indices[i, [first_half, second_half]] = 1

  # Zip the values and indices in a third dimension:
  # inputs has the shape (batch_size, time_steps, 2)
  inputs = np.dstack((add_values, add_indices))
  targets = np.sum(np.multiply(add_values, add_indices), axis=1)
  return inputs, targets


# from the following : https://minpy.readthedocs.io/en/latest/tutorial/rnn_tutorial/rnn_tutorial.html
def adding_problem_generator(N, seq_len=6, high=1):
    """ A data generator for adding problem.

    The data definition strictly follows Quoc V. Le, Navdeep Jaitly, Geoffrey E.
    Hintan's paper, A Simple Way to Initialize Recurrent Networks of Rectified
    Linear Units.

    The single datum entry is a 2D vector with two rows with same length.
    The first row is a list of random data; the second row is a list of binary
    mask with all ones, except two positions sampled by uniform distribution.
    The corresponding label entry is the sum of the masked data. For
    example:

     input          label
     -----          -----
    1 4 5 3  ----->   9 (4 + 5)
    0 1 1 0

    :param N: the number of the entries.
    :param seq_len: the length of a single sequence.
    :param p: the probability of 1 in generated mask
    :param high: the random data is sampled from a [0, high] uniform distribution.
    :return: (X, Y), X the data, Y the label.
    """
    X_num = np.random.uniform(low=0, high=high, size=(N, seq_len, 1))
    X_mask = np.zeros((N, seq_len, 1))
    Y = np.ones((N, 1))
    for i in xrange(N):
        # Default uniform distribution on position sampling
        positions = np.random.choice(seq_len, size=2, replace=False)
        X_mask[i, positions] = 1
        Y[i, 0] = np.sum(X_num[i, positions])
    X = np.append(X_num, X_mask, axis=2)
    return X, Y
