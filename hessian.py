#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:54:17 2022

@author: Bas
"""

import numpy as np 


def hessian(fun, x, precision_loss = 500):
    """
    Compute the 2-sided Hessian matrix for any function!

    Parameters
    ----------
    fun : function
        The function for which to compute the Hessian matrix. Must have a single
        output. Note the output of any function may be manipulated using lambda
        functions.
    x : list
        Optimal parameters, or a point of interest.
    precision_loss : float, optional
        A correction parameter to ensure the machine can comprehend the small 
        changes associated with numerical differencing. The default is 500.

    Returns
    -------
    mH : TYPE
        DESCRIPTION.

    """
    
    delta = np.finfo(np.float32).eps * precision_loss
    n = len(x)
    
    mh = np.diag(np.full([n], delta))
    
    # values of y where each y is the result of one x being slightly altered.
    y_u, y_l = compute_y_2sided(fun, x, precision_loss)
    y_m = fun(x)
    
    #Now do the same as above but do it twice while looping over all combinations of x
    #again doing positive and negative values.
    so_y_u = np.zeros_like(mh)
    so_y_l = np.zeros_like(mh)
    
    for i, v in enumerate(mh):
        for j, w in enumerate(mh):
            so_y_u[i,j] = fun(x + v + w)
            so_y_l[i,j] = fun(x - v - w)

    deltasq = delta**2
    mH = np.zeros([n,n])
    
    for i in range(n):
        for j in range(i,n):
            v = (so_y_u[i,j] - y_u[i] - y_u[j] + (2*y_m) - y_l[i] - y_l[j] + so_y_l[i,j])/(deltasq)/2
            mH[i,j] = v
            mH[j,i] = v
    
    return mH

def compute_y_2sided(fun, x, precision_loss = 500):
    
    delta = np.finfo(np.float32).eps * precision_loss
    n = len(x)
    
    #Create a matrix where all variables are incremented one-by-one each row
    mh = np.diag(np.full([n], delta))
    ml = -1*mh
    
    mh = mh + x
    ml = ml + x
    
    y_u = np.array([fun(i) for i in mh])
    y_l = np.array([fun(i) for i in ml])
    
    return y_u, y_l

def gradient_vector(fun, x, precision_loss = 500):
    
    delta = np.finfo(np.float32).eps * precision_loss
    y_u, y_l = compute_y_2sided(fun, x)
    
    return (y_u - y_l) * ((2*delta)**-1)

