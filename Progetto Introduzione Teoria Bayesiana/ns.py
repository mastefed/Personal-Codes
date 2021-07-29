def log_multivariate_point(array, dimensions):
    """
    Given a set of parameters, calculates the logarithm of the
    multivariate normal distribution.
    
    Args:
        array (float): array of parameters' values
        dimensions (int): dimension of the parameters' space
        
    Returns:
        point (float): the logarithm of the multivariate normal distribution given the initial array
    """
    mean_array = np.zeros(shape=dimensions, dtype=float)
    covariance_matrix = np.identity(dimensions)
    exponent = np.dot((array - mean_array), np.dot(covariance_matrix, (array - mean_array)))
    point = - 0.5*exponent - (dimensions/2)*np.log(2*np.pi)
    return point

def proposal(generator, distrib_function,  dimensions, mult_par, loc_value, parameters):
    """
    Generates a n-dimensional point given any continous distribution from the SciPy package.
    The mean value of the distribution can be chosen to be any and the scaling parameter is
    defined as the standard deviation of a set of points already existing in the paramters space
    times a multiplicative parameter. 
    
    Args:
        generator: None or Generator to block the random generator seed
        distrib_function: one of the continous distributions from the SciPy package
        dimensions (int): the dimensions of the parameters' space
        mult_par (float): the multiplicative parameter
        loc_value (array or float): the mean value of the chosen distribution
        parameters (array): set of points in the parameters space
        
    Returns:
        proposed_point (array): the point generated from the chosen distribution
    """
    scaling = mult_par*np.std(parameters)
    proposed_point = distrib_function.rvs(loc=loc_value, scale=scaling, size=(1,dimensions), random_state=generator)
    return proposed_point  
    
def prior_uniform_sampling(generator, dimensions, parameters, mult_par, loc_value, lb_val, hb_val, distrib_function):
    """
    Using the proposal function, generates a point and checks if it stays within the 
    boundaries of the hypercube representing the prior volume. If it does, returns such
    point, if it doesn't generates a new point and do the check again.
    
    Args:
        generator: None or Generator to block the random generator seed
        dimensions (int): the dimensions of the parameters' space
        parameters (array): set of points in the parameters space
        mult_par (float): see the proposal function documentation
        loc_value (array): see the proposal function documentation
        lb_val (float): the low boundary value of the hypercube, e.g. lb_val = -1. --> x = [-1., -1., ..., -1.]
        hb_val (float): the high boundary value of the hypercube, e.g. hb_val = 1. --> x = [1., 1., ..., 1.]
        distrib_function: see the proposal function documentation
    """
    new_point = proposal(generator, distrib_function, dimensions, mult_par, loc_value, parameters)

    # Check if the proposed point is in the boundary
    cond1 = any(axis > hb_val for axis in new_point[0])
    cond2 = any(axis < lb_val for axis in new_point[0])
    
    while (cond1 or cond2) == True:
        # print("Out of boundary")
        new_point = proposal(generator, distrib_function, dimensions, mult_par, loc_value, parameters)
        cond1 = any(axis > hb_val for axis in new_point[0])
        cond2 = any(axis < lb_val for axis in new_point[0])
    
    return new_point

def lowest_likelihood_point(log_like_points):
    """
    After calculating the log_multivariate_points checks which one is the lowest.
    
    Args:
        log_like_points (array): the log_multivariate_points
        
    Returns:
        min_index (int): the index "i" such that log_multivariate_point(parameters[i], dimensions) is the lowest value 
        min_likelihood_value (float): the aforementioned lowest value
    """
    min_index = np.argmin(log_like_points)
    min_likelihood_value = log_like_points[min_index]
    return min_index, min_likelihood_value
    
def define_controlboard():
    """
    Creates the figure that will show some useful plots, called Control Board.
    In the upper part of the figure one can see the normal distribution and the selected lowest value point.
    In the lower part of the figure one can find the samples representing the prior volume and the proposal
    function centered in the parameter which generated the lowest valued log_multivariate_point.
    
    Pay attention: this works only in dimension 1.
    
    Returns:
        fig, axs: figure and axis without content
    """
    plt.ion()
    
    fig = plt.figure(num="CB", figsize=(5,7))
    axs = fig.subplots(nrows=2, ncols=1, sharex=True)
        
    return fig, axs
    
def update_controlboard(parameters, min_index, log_mult_points, mult_param):
    """
    Updates the content of the Control Board during the code execution.
    
    Args:
        parameters (array): set of points in the parameters space
        min_index (int): the index "i" such that log_multivariate_point(parameters[i], dimensions) is the lowest value
        log_mult_points (array): the set of log_mult_points calulated using the log_multivariate_point function
        mult_par (float): the multiplicative parameter needed to plot the proposal function
    """
    plt.suptitle("Control board")
    
    line, = axs[0].plot(parameters, np.exp(log_mult_points), 'x', ms=3.5, label="L values")
    vertline = axs[0].axvline(parameters[min_index], color='r', ls='--', label='Min L value')
    axs[0].set_title("Likelihood")
    axs[0].legend(loc=8)
    
    line = axs[1].hist(parameters, bins=15)
    scal = mult_param*np.std(parameters)
    x = np.linspace(min(parameters), max(parameters), 2000)
    sampler = proposal_function.pdf(x, loc=parameters[min_index], scale=scal)
    line, = axs[1].plot(x, sampler, label="Proposal")
    axs[1].legend(loc=1)
    axs[1].set_xlabel("Parameters")
    axs[1].set_title("Live-points")
    
    plt.show()
    plt.pause(0.00001)
    
    axs[0].cla()
    axs[1].cla()

import os
import time
import argparse

import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from scipy.stats import anglit,logistic, norm, semicircular
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Welcome. This is a benchmark test for a Nested Sampling algorithm. Its purpose is to integrate a Multivariate Normal Distribution with zero mean and the identity matrix as covariance matrix.')
    
    parser.add_argument('dim', metavar='Dimensions', type=int,
                    help='How many dimensions should have the parameters\' space?')
    parser.add_argument('pts', metavar='Points', type=int,
                    help='Choose the number of points for the Prior Sample')
    parser.add_argument('distr', metavar='Sampling Distribution',
                    help='Which distribution would you like to use as a Sampler?')
    parser.add_argument('plot', metavar='Control Board', choices=['True', 'False'],
                    help='Select True if you want an over time updated plot of the Prior Sample and of the Likelihood! Warning: This works only for Dimension = 1!')
    
    args = parser.parse_args()

    term_size = os.get_terminal_size()
    towel = np.random.Generator(np.random.PCG64(42))
    
    plots = args.plot
    
    dimensions   = args.dim
    samples_size = args.pts
    proposal_function = eval(args.distr)
    
    list_of_mult_params = np.arange(1., 5.)
    proposals_param_iterable = it.cycle(list_of_mult_params)

    log_evidence       = -np.inf
    logh                     = -np.inf
    constant_increment = np.log(1 - np.exp(-1/samples_size))
    fraction = .1
    
    low_bound_value    = -4.
    high_bound_value  =   4.
    
    low_bound   = np.full(shape=dimensions, fill_value = low_bound_value, dtype=float)
    high_bound  = np.full(shape=dimensions, fill_value = high_bound_value, dtype=float)
    
    parameters = towel.uniform(low_bound, high_bound, size=(samples_size, dimensions))
    
    t1 = time.time()
    
    if plots == 'True' and dimensions == 1:
        fig, axs = define_controlboard()
    
    i = 0
    
    while True:
        log_mult_points = [log_multivariate_point(parameters[j], dimensions) for j in range(samples_size)]
        
        min_index, min_likelihood_value = lowest_likelihood_point(log_mult_points)
        
        mult_param = next(proposals_param_iterable)
        
        if plots == 'True' and dimensions == 1:
            update_controlboard(parameters, min_index, log_mult_points, mult_param)
        
        a_new_point = prior_uniform_sampling(
            towel, dimensions, parameters,
            mult_param, parameters[min_index], 
            low_bound_value, high_bound_value, 
            proposal_function
            )
        
        if (log_multivariate_point(a_new_point[0], dimensions) > min_likelihood_value) == True:
            t2 = time.time()
            
            log_delta_X = -i/samples_size + constant_increment
            
            #                           This is decreasing                     This is increasing
            if max(log_mult_points) - log_evidence < np.log(fraction) + (i + i**.5)/samples_size:
                print("The mean increment is now too small to make a difference!\n")
                break
            
            print("─"*term_size.columns)
            print("This loop will end when max(ln(L)) - log(Z) is lesser than log(f) + (i + sqrt(i))/N")
            print(f"max(ln(L)) - log(Z)            --> {max(log_mult_points) - log_evidence}")
            print(f"log(f)     + (i + sqrt(i))/N   --> {np.log(fraction) + (i + i**0.5)/samples_size}\n")
            print(f"I found a new likelihood minimum after {(t2 - t1)} seconds!")
            parameters[min_index] = a_new_point[0]
            
            log_evidence = np.logaddexp(log_evidence, (log_delta_X + min_likelihood_value))
            
            loghplus = min_likelihood_value - log_evidence + np.log(min_likelihood_value - log_evidence) + log_delta_X
            logh = np.logaddexp(logh, loghplus)
            error = np.sqrt(np.exp(logh)/samples_size)
            
            print(f"The logarithm of evidence at iteration {i+1} is ------> {log_evidence} +- {error}")
            print("─"*term_size.columns + "\n")
            
            i += 1
        
        else:
            parameters[min_index] = parameters[min_index]

    t3 = time.time()
    
    fin_time = round((t3-t1)/60, 2)
    fin_log_ev = round(log_evidence, 4)
    fin_err = round(error, 4)
    min_log_ev_val = round(log_evidence + error, 4)
    max_log_ev_val = round(log_evidence - error, 4)
    theor_value = round(-dimensions*np.log(high_bound_value - low_bound_value), 4)
    
    print("─"*term_size.columns)
    print(f"The last value of log_evidence is {fin_log_ev} +- {fin_err}, hence ------>")
    print(f"------> the logarithm of the evidence spans from {min_log_ev_val} to {max_log_ev_val}\n ")
    print(f"This result was found in {fin_time} minutes.\n")
    
    print(f"The correct value, for a hypercube of side {2*high_bound_value}, should be {theor_value}")
    print("─"*term_size.columns + "\n")