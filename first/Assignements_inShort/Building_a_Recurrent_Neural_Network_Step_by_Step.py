#!/usr/bin/env python
# coding: utf-8

import numpy as np
from rnn_utils import *
from public_tests import *
import inspect

# --------------------------------------------------------------------------------------
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: rnn_cell_forward

def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    ### START CODE HERE ### (≈2 lines)
    
    # compute next activation state using the formula given above
    
    #print ( 'aprev:' , a_prev.shape )
    #print ( 'waa:' , Waa.shape )
    #print ( 'xt:' , xt.shape )
    #print ( 'wax:' , Wax.shape )
    #print ( 'ba:' , ba.shape )
    
    #print ( 'xt:' , xt.shape )
    #print ( 'wax:' , Wax.shape )
    #print ( 'by:' , by.shape )
    
    #print ( inspect.getsource ( softmax ) )
    
    #def softmax(x):
    #    e_x = np.exp(x - np.max(x))
    #    return e_x / e_x.sum(axis=0)
    
    a_next = np.tanh (
        np.dot ( Waa , a_prev )
        + np.dot ( Wax , xt )
        + ba
    )

    # compute output of the current cell using the formula given above
    yt_pred = softmax (
        np.dot ( Wya , a_next )
        + by
    )
    
    # print ( np.dot ( Wax , xt ).shape == xt.shape ) # false
    # HOW CAN I DECIDE m FOR Wax ? # not len ( x ) = MxM as mx
    
    #print ( a_prev.shape )
    #print ( a_next.shape )
    #print ( yt_pred.shape )
    
    ### END CODE HERE ###
    
    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache

np.random.seed(1)
xt_tmp = np.random.randn(3, 10)
a_prev_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5, 5)
parameters_tmp['Wax'] = np.random.randn(5, 3)
parameters_tmp['Wya'] = np.random.randn(2, 5)
parameters_tmp['ba'] = np.random.randn(5, 1)
parameters_tmp['by'] = np.random.randn(2, 1)

a_next_tmp, yt_pred_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)
print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = \n", a_next_tmp.shape)
print("yt_pred[1] =\n", yt_pred_tmp[1])
print("yt_pred.shape = \n", yt_pred_tmp.shape)

# UNIT TESTS
rnn_cell_forward_tests(rnn_cell_forward)

# **Expected Output**: 
# ```Python
# a_next[4] = 
#  [ 0.59584544  0.18141802  0.61311866  0.99808218  0.85016201  0.99980978
#  -0.18887155  0.99815551  0.6531151   0.82872037]
# a_next.shape = 
#  (5, 10)
# yt_pred[1] =
#  [ 0.9888161   0.01682021  0.21140899  0.36817467  0.98988387  0.88945212
#   0.36920224  0.9966312   0.9982559   0.17746526]
# yt_pred.shape = 
#  (2, 10)
# 
# ```

# --------------------------------------------------------------------------------------
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: rnn_forward

def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    
    # Initialize "caches" which will contain the list of all caches
    caches = []
    
    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    ### START CODE HERE ###
    
    #print ( 'n_x:' , n_x )
    #print ( 'm:' , m )
    #print ( 'T_x:' , T_x )
    
    #print ( 'n_y:' , n_y )
    #print ( 'n_a:' , n_a )
    
    # initialize "a" and "y_pred" with zeros (≈2 lines)
    a = np.zeros ( (n_a, m, T_x) ) # ; a[ :,:,0 ] = a0
    y_pred = np.zeros ( (n_y, m, T_x) )
    
    # Initialize a_next (≈1 line)
    a_next = a0
    
    # loop over all time-steps
    for t in range(T_x):
        
        # Update next hidden state, compute the prediction, get the cache (≈1 line)
        a_next, yt_pred, cache = rnn_cell_forward( x[:,:,t] , a_next, parameters )
        
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        
        # Save the value of the prediction in y (≈1 line)
        y_pred[:,:,t] = yt_pred
        
        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)
    
    ### END CODE HERE ###
    
    # store values needed for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches

np.random.seed(1)
x_tmp = np.random.randn(3, 10, 4)
a0_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5, 5)
parameters_tmp['Wax'] = np.random.randn(5, 3)
parameters_tmp['Wya'] = np.random.randn(2, 5)
parameters_tmp['ba'] = np.random.randn(5, 1)
parameters_tmp['by'] = np.random.randn(2, 1)

a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][1] = \n", a_tmp[4][1])
print("a.shape = \n", a_tmp.shape)
print("y_pred[1][3] =\n", y_pred_tmp[1][3])
print("y_pred.shape = \n", y_pred_tmp.shape)
print("caches[1][1][3] =\n", caches_tmp[1][1][3])
print("len(caches) = \n", len(caches_tmp))

#UNIT TEST    
rnn_forward_test(rnn_forward)

# **Expected Output**:
# 
# ```Python
# a[4][1] = 
#  [-0.99999375  0.77911235 -0.99861469 -0.99833267]
# a.shape = 
#  (5, 10, 4)
# y_pred[1][3] =
#  [ 0.79560373  0.86224861  0.11118257  0.81515947]
# y_pred.shape = 
#  (2, 10, 4)
# caches[1][1][3] =
#  [-1.1425182  -0.34934272 -0.20889423  0.58662319]
# len(caches) = 
#  2
# ```

# --------------------------------------------------------------------------------------
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: lstm_cell_forward

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight (notice the variable name)
    bi = parameters["bi"] # (notice the variable name)
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    ### START CODE HERE ###

    # Concatenate a_prev and xt (≈1 line)
    concat = np.concatenate( (a_prev, xt), axis=0 )

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid ( np.dot ( Wf , concat ) + bf )
    it = sigmoid ( np.dot ( Wi , concat ) + bi )
    ot = sigmoid ( np.dot ( Wo , concat ) + bo )
    
    #print ( inspect.getsource ( sigmoid ) )
    #def sigmoid(x): return 1 / (1 + np.exp(-x))
    
    cct = np.tanh ( np.dot ( Wc , concat ) + bc )
    
    c_next = ( ft * c_prev ) + ( it * cct )
    a_next = ot * np.tanh ( c_next )
    
    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = softmax ( np.dot ( Wy , a_next ) + by )
    
    ### END CODE HERE ###

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache

np.random.seed(1)
xt_tmp = np.random.randn(3, 10)
a_prev_tmp = np.random.randn(5, 10)
c_prev_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5 + 3)
parameters_tmp['bf'] = np.random.randn(5, 1)
parameters_tmp['Wi'] = np.random.randn(5, 5 + 3)
parameters_tmp['bi'] = np.random.randn(5, 1)
parameters_tmp['Wo'] = np.random.randn(5, 5 + 3)
parameters_tmp['bo'] = np.random.randn(5, 1)
parameters_tmp['Wc'] = np.random.randn(5, 5 + 3)
parameters_tmp['bc'] = np.random.randn(5, 1)
parameters_tmp['Wy'] = np.random.randn(2, 5)
parameters_tmp['by'] = np.random.randn(2, 1)

a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)

print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = ", a_next_tmp.shape)
print("c_next[2] = \n", c_next_tmp[2])
print("c_next.shape = ", c_next_tmp.shape)
print("yt[1] =", yt_tmp[1])
print("yt.shape = ", yt_tmp.shape)
print("cache[1][3] =\n", cache_tmp[1][3])
print("len(cache) = ", len(cache_tmp))

# UNIT TEST
lstm_cell_forward_test(lstm_cell_forward)


# **Expected Output**:
# 
# ```Python
# a_next[4] = 
#  [-0.66408471  0.0036921   0.02088357  0.22834167 -0.85575339  0.00138482
#   0.76566531  0.34631421 -0.00215674  0.43827275]
# a_next.shape =  (5, 10)
# c_next[2] = 
#  [ 0.63267805  1.00570849  0.35504474  0.20690913 -1.64566718  0.11832942
#   0.76449811 -0.0981561  -0.74348425 -0.26810932]
# c_next.shape =  (5, 10)
# yt[1] = [ 0.79913913  0.15986619  0.22412122  0.15606108  0.97057211  0.31146381
#   0.00943007  0.12666353  0.39380172  0.07828381]
# yt.shape =  (2, 10)
# cache[1][3] =
#  [-0.16263996  1.03729328  0.72938082 -0.54101719  0.02752074 -0.30821874
#   0.07651101 -1.03752894  1.41219977 -0.37647422]
# len(cache) =  10
# ```

# --------------------------------------------------------------------------------------
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: lstm_forward

def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []
    
    ### START CODE HERE ###
    
    Wy = parameters['Wy'] # saving parameters['Wy'] in a local variable in case students use Wy instead of parameters['Wy']
    
    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros ( (n_a, m, T_x) )
    c = np.zeros ( (n_a, m, T_x) )
    y = np.zeros ( (n_y, m, T_x) )
    
    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = 0
    
    # loop over all time-steps
    for t in range(T_x):
        
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:,:,t]
        
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache = lstm_cell_forward( x[:,:,t], a_next, c_next, parameters )
        
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        
        # Save the value of the next cell state (≈1 line)
        c[:,:,t]  = c_next
        
        # Save the value of the prediction in y (≈1 line)
        y[:,:,t] = yt
        
        # Append the cache into caches (≈1 line)
        caches.append( cache )
        
    ### END CODE HERE ###
    
    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches

np.random.seed(1)
x_tmp = np.random.randn(3, 10, 7)
a0_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5 + 3)
parameters_tmp['bf'] = np.random.randn(5, 1)
parameters_tmp['Wi'] = np.random.randn(5, 5 + 3)
parameters_tmp['bi']= np.random.randn(5, 1)
parameters_tmp['Wo'] = np.random.randn(5, 5 + 3)
parameters_tmp['bo'] = np.random.randn(5, 1)
parameters_tmp['Wc'] = np.random.randn(5, 5 + 3)
parameters_tmp['bc'] = np.random.randn(5, 1)
parameters_tmp['Wy'] = np.random.randn(2, 5)
parameters_tmp['by'] = np.random.randn(2, 1)

a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][3][6] = ", a_tmp[4][3][6])
print("a.shape = ", a_tmp.shape)
print("y[1][4][3] =", y_tmp[1][4][3])
print("y.shape = ", y_tmp.shape)
print("caches[1][1][1] =\n", caches_tmp[1][1][1])
print("c[1][2][1]", c_tmp[1][2][1])
print("len(caches) = ", len(caches_tmp))

# UNIT TEST    
lstm_forward_test(lstm_forward)

# **Expected Output**:
# 
# ```Python
# a[4][3][6] =  0.172117767533
# a.shape =  (5, 10, 7)
# y[1][4][3] = 0.95087346185
# y.shape =  (2, 10, 7)
# caches[1][1][1] =
#  [ 0.82797464  0.23009474  0.76201118 -0.22232814 -0.20075807  0.18656139
#   0.41005165]
# c[1][2][1] -0.855544916718
# len(caches) =  2
# ```

