### Assignment 2 - Advanced Econometric Methods
# Author:           Jan Koolbergen
# Student number:   2667333
# Date:             26-09-2022

### I like to type-hint my functions, so my IDE can perform syntax highlighting
# This is however, a (somewhat) recent addition to python
# I use python v3.10.4, but I have confirmed that this code works on all versions from v3.8.3 onward
# In the case that the code does not run, please uncomment the first import (__future__), as this will fix the issue

### Imports
# from __future__ import annotations
from typing import Callable, NamedTuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import scipy.stats as sts
import math

def compute_NIC(theta, x = np.arange(-5,5.1,.1)):
    
    sig2_start = theta[0]
    mu = theta[1]
    omega = theta[2]
    alpha = theta[3]
    beta = theta[4]
    delta = theta[5]
    lam = theta[6]
    
    ind_f = x < 0
    
    sigmas = []
        
    for i, j in zip(x, ind_f):
        sigma_2_t = omega + (((alpha*(i-mu)**2) + (delta*(i-mu)**2) * j) / (1 + ((i**2)/(lam*sig2_start)))) + (beta * sig2_start)
        sigmas.append(sigma_2_t)
            
    return x, sigmas

def question_1() -> None:
    """Answers question 1: """
    lambdas = [2, 5, 10, 50] 
    deltas = [0, 1, 0.2, 0.4] 
    sig = 1
    omega = 0
    alpha = 0.05
    beta = 0.9
    fig, axs = plt.subplots(round(len(deltas)/2), 2, sharex='col', figsize=[16,12])
    cnt_x = 0
    cnt_y = 0
    for i in deltas:
        for j in lambdas:
            x, sigma = compute_NIC([sig,0,omega,alpha,beta,i,j])
            axs[cnt_y,cnt_x].plot(x, sigma, label = f"lambda = {j}")
        axs[cnt_y,cnt_x].title.set_text(f"NIC for delta={i}")
        handles, labels = axs[cnt_y,cnt_x].get_legend_handles_labels()
        fig.legend(handles, labels, loc = (0.425,0.94), ncol=2)
        fig.text(0.5, 0.075, 'Value of x', ha='center')
        fig.text(0.085,0.5,'News Impact', ha='center', rotation='vertical')
        if cnt_x < 1:
            cnt_x += 1
        else:
            cnt_x = 0
            cnt_y += 1
            
    plt.show()
       
def question_2(data_dict: dict) -> None:
    """Answers question 2: """
    for stock_name, stock_data in data_dict.items():
        print(f"\nSummary statistics for stock {stock_name}:")
        summary_statistics(stock_data)
        
    plot_stock_returns(data_dict)
        
    
def question_3(data_dict: dict):
    """Answers question 3: """
    nolev_dict, lev_dict = {}, {}
    
    for stock_name, stock_data in data_dict.items():
        # No leverage
        print(f"\nResults for {stock_name} without leverage")
        initial_omega = np.var(stock_data.RET.iloc[0:2500]) / 50
        initial_theta = np.array([initial_omega, 0.02, 0.96, 5])
        nolev_results = opt.minimize(GARCH_log_likelihood, initial_theta, args=(stock_data))
        report_optimization_results(nolev_results, stock_name)
        nolev_dict.update({stock_name: nolev_results.x})
        
        # With leverage
        print(f"\nResults for {stock_name} with leverage")
        initial_theta = np.append(initial_theta, [0])
        lev_results = opt.minimize(GARCH_log_likelihood, initial_theta, args=(stock_data, True))
        report_optimization_results(lev_results, stock_name)
        lev_dict.update({stock_name: lev_results.x})
        
    return nolev_dict, lev_dict
        
        
def report_optimization_results(opt_res, stock_name: str, T: int=2500) -> None:
    """Reports the optimization results for question 3."""
    standard_errors = np.sqrt(np.diag(opt_res.hess_inv) / T)
    likelihood = opt_res.fun * -T
    print(f"Parameter estimates:")
    print('')
    print(f"omega  : {opt_res.x[0]:.4f} ({standard_errors[0]:.4f})")
    print(f"alpha  : {opt_res.x[1]:.4f} ({standard_errors[1]:.4f})")
    print(f"beta   : {opt_res.x[2]:.4f} ({standard_errors[2]:.4f})")
    print(f"lambda : {opt_res.x[3]:.4f} ({standard_errors[3]:.4f})")
    if len(opt_res.x) > 4: print(f"delta  : {opt_res.x[4]:.4f} ({standard_errors[4]:.4f})")
    print('')
    print(f"Log likelihood: {likelihood:.3f}")
    print(f"AIC: {-2*likelihood + 2*len(opt_res.x):.3f}")
    print(f"BIC: {-2*likelihood + len(opt_res.x)*np.log(T):.3f}")
    
    
def question_3_test_case(data_dict: dict) -> None:
    """Tests if we get the same output as the examples for question 3."""
    KO = data_dict['KO']
    
    # Test if the likelihood is the same in the case of no-leverage
    test_theta = (0.005665683, 0.09922153, 0.91055318, 5.5415955)
    likelihood = GARCH_log_likelihood(test_theta, KO, return_mean=False, use_leverage=False)
    print(f"Likelihood should be -3774. and is {likelihood:.2f}")
    
    # Test if the likelihood is the same in the case of leverage
    test_theta = (0.007442577, 0.02670845, 0.9217446, 6.056328, 0.11266453)
    likelihood = GARCH_log_likelihood(test_theta, KO, return_mean=False)
    print(f"Likelihood should be -3758. and is {likelihood:.2f}")
    
    
def question_4(data_dict, lev_par, nolev_par, sig0 = 1) -> None:
    """Answers question 4: """
    fig, axs = plt.subplots(4, 2, squeeze=False, figsize=[16,12])
    for i, it in enumerate(zip(lev_par.items(), nolev_par.items())):
        v, w = it
        tick, l = v
        nl = w[1]
        theta_nl = [sig0, 0, nl[0], nl[1], nl[2], 0, nl[3]]
        theta_l = [sig0, 0, l[0], l[1], l[2], l[4], l[3]]
        x, sig_nl = compute_NIC(theta_nl)
        x, sig_l = compute_NIC(theta_l)
        breakpoint()
        axs[i,0].plot(x, sig_nl, label='No leverage')
        axs[i,0].plot(x, sig_l, label='Leveraged')
        axs[i,1].plot(data_dict[tick]['RET'])
        #axs[i,:].title.set_text(f"{tick}")

def question_5(data_dict: dict, nolev_dict: dict, lev_dict: dict) -> None:
    """Answers question 5: """
    N = 1
    horizons = [1, 5, 20]
    levels = [0.01, 0.05, 0.1]
    
    for stock_name, stock_data in data_dict.items():
        if stock_name != 'KO': continue
        time_to_april = len(stock_data[stock_data.date < '2020/04/01'])
        starting_x = stock_data.RET.iloc[time_to_april]
        
        starting_s2 = np.var(stock_data.RET.iloc[0:50])
        for t in range(1, time_to_april+1):
            starting_s2 = GARCH_s2_formula(starting_s2, stock_data.RET.iloc[t-1], nolev_dict[stock_name], use_leverage=False)

        # No leverage
        print(f"\nResults for {stock_name} without leverage")
        sim_returns = np.zeros((N, max(horizons)))
        for i in range(N):
            sim_returns[i,:] = simulate_compound_returns(starting_x, starting_s2, nolev_dict[stock_name], use_leverage=False, T=max(horizons))
        
        for h in horizons:
            for level in levels:
                print(f"Level = {level} - h={h} - VaR = {np.percentile(sim_returns[:,(h-1)], 100*level):.4f}")
                
        # With leverage
        starting_s2 = np.var(stock_data.RET.iloc[0:50])
        for t in range(1, time_to_april):
            starting_s2 = GARCH_s2_formula(starting_s2, stock_data.RET.iloc[t-1], lev_dict[stock_name], use_leverage=True)
            
        print(f"\nResults for {stock_name} with leverage")
        sim_returns = np.zeros((N, max(horizons)))
        for i in range(N):
            sim_returns[i,:] = simulate_compound_returns(starting_x, starting_s2, lev_dict[stock_name], use_leverage=True, T=max(horizons))
        
        for h in horizons:
            for level in levels:
                print(f"Level = {level} - h={h} - VaR = {np.percentile(sim_returns[:,(h-1)], 100*level):.4f}")
    

def simulate_compound_returns(starting_x, starting_s2, theta, use_leverage: bool=False, T: int=20) -> np.ndarray:
    """Simulates one GARCH path using random draws. Returns the compound returns from that path"""
    x = np.zeros(T+1)
    s2 = np.zeros(T+1)
    compound_returns = np.zeros(T)
    compound_return_factor = 1
    x[0] = starting_x
    s2[0] = starting_s2
    
    for t in range(1, T+1):
        s2[t] = GARCH_s2_formula(s2[t-1], x[t-1], theta, use_leverage=use_leverage)
        x[t] = s2[t] * np.random.standard_t(df=theta[3])
        print(x[t])
        compound_return_factor *= (1 + x[t] / 100)
        compound_returns[t-1] = 100 * (compound_return_factor - 1)
        
    return compound_returns
        
    
def question_6(data_dict: dict) -> None:
    """Answers question 6: """
    ...


def GARCH_log_likelihood(theta, stock_data: pd.DataFrame, use_leverage: bool=False, T: int=2500, return_mean: bool=True, return_negative: bool=False) -> float:
    """Calculates the log-likelihood of certain GARCH-model parameters."""
    x = stock_data.RET.iloc[0:T]
    s2 = np.zeros(T)
    s2[0] = np.var(stock_data.RET.iloc[0:50])
    c = 0.0001
    l = theta[3] # for readability, save lambda in l
    
    for t in range(1, T):
        s2[t] = GARCH_s2_formula(s2[t-1], x.iloc[t-1], theta, use_leverage=use_leverage)
        
    if np.any(s2 < 0) or theta[3] < c:
        log_likelihood = -100*T - 10000*T*((l - c)**2 *((l < c)*1) + np.sum((s2**2) * ((s2 < 0)*1)))
    else:
        log_likelihood = math.lgamma(0.5 * (l + 1)) - math.lgamma(0.5*l) - 0.5*np.log(l) - 0.5*np.log(np.pi) - 0.5*(l + 1) * np.log(1 + ((x / np.sqrt(s2))**2) / l) -0.5*np.log(s2)
        
    sign = 1 if return_negative else -1 # Because the average likelihood is already negative
    if return_mean:
        return sign * np.mean(log_likelihood)
    else:
        return sign * np.sum(log_likelihood)


def GARCH_s2_formula(s2: float, x: float, theta, use_leverage: bool=False) -> float:
    """Calculates the next s2 in a GARCH model."""
    # for the parameters: [0] = omega, [1] = alpha, [2] = beta, [3] = lambda, [4] = delta
    d = theta[4] if use_leverage else 0
    return theta[0] + (theta[1] * x**2 + d * x**2 * ((x < 0)*1)) / (1 + x**2 / (theta[3] * s2)) + theta[2] * s2


def plot_stock_returns(data_dict: dict) -> None:
    """Plots the returns for each of the stocks in the dictionary."""
    # Set up for the entire figure
    colors = ['slateblue', 'lightsteelblue', 'lightslategrey', 'cadetblue']
    fig = plt.figure(figsize=(10, 20))
    fig.subplots_adjust(hspace=0.4)
    
    # Create the subplots
    for i, (stock_name, stock_data) in enumerate(data_dict.items()):
        sp = plt.subplot(len(data_dict), 1, i+1)
        sns.lineplot(y=stock_data.RET, x=range(0, len(stock_data.RET)), ax=sp, linewidth=0.5, color=colors[i])
        sp.set_title(f"Stock {stock_name}")
        sp.set_ylabel(r"Returns")
        sp.set_xlabel(r"t")
        
    plt.show()
        
    
def summary_statistics(stock_data: pd.DataFrame) -> None:
    """This function takes in some stock data and reports some summary statistics."""
    returns = stock_data.RET # Just put this in a variable for convenience
    
    no_of_observations = len(returns)
    mean = np.mean(returns)
    
    # medians
    Q1 = np.percentile(returns, 25)
    Q2 = np.percentile(returns, 50)
    Q3 = np.percentile(returns, 75)
    
    standard_deviation = np.std(returns)
    skewness = sts.skew(returns)
    kurtosis = sts.kurtosis(returns)
    
    minimum = np.min(returns)
    maximum = np.max(returns)
    
    # Now we report the results:
    print(f"Number of observations: {no_of_observations}")
    print(f"Mean: {mean:.4f}")
    print(f"Q1: {Q1:.4f}")
    print(f"Q2: {Q2:.4f}")
    print(f"Q3: {Q3:.4f}")
    print(f"Standard deviation: {standard_deviation:.4f}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis {kurtosis:.4f}")
    print(f"Min: {minimum:.4f}")
    print(f"Max: {maximum:.4f}")


def data_preprocess(data_dict: dict) -> dict:
    """Demeans and scales the data by 100."""
    new_dict = {}
    for stock_name, raw_stock_data in data_dict.items():
        stock_data = raw_stock_data.copy()
        
        mean_return = np.mean(stock_data.RET)
        stock_data.RET = stock_data.RET.transform(lambda ret: ret - mean_return)
        stock_data.RET = stock_data.RET.transform(lambda ret: ret * 100)
        
        new_dict.update({stock_name: stock_data})
        
    return new_dict


def read_in_data() -> pd.DataFrame:
    """Reads the data from a csv file into a pandas DataFrame. 
    This function assumes that the current working directory is the same as the file directory."""
    file_path = "Assignment3_dataset.csv"
    return pd.read_csv(filepath_or_buffer=file_path)


def main() -> None:
    """Main function. Runs all code required for the assignment."""
    # Read the data
    raw_data = read_in_data()
    data_dict = {stock: raw_data[raw_data.TICKER == stock] for stock in raw_data.TICKER.unique()}
    scaled_data_dict = data_preprocess(data_dict)
    
# =============================================================================
#     # For testing 5 and 6: TODO remove
#     nolev_dict = {
#         'KO':  np.array([0.00564837, 0.09934079, 0.91048003, 5.54233274]), 
#         'PFE': np.array([0.02897666, 0.11873272, 0.88174828, 6.17490427]), 
#         'JNJ': np.array([0.01349871, 0.15553047, 0.85263037, 5.95413620]), 
#         'MRK': np.array([0.04942468, 0.16251609, 0.83947497, 4.38238453])
#     }
#     
#     lev_dict = {
#         'KO':  np.array([0.00741560, 0.02706942, 0.92155679, 6.01830196, 0.11256638]), 
#         'PFE': np.array([0.01383148, 0.02818548, 0.91973697, 6.00233458, 0.11151245]), 
#         'JNJ': np.array([0.03327439, 0.01545059, 0.88586522, 5.01367745, 0.12399875]), 
#         'MRK': np.array([0.03522076, 0.04217402, 0.88308114, 4.38340737, 0.15444859])
#     }
# =============================================================================
    
    # Go through the test cases
    print("=== Testing question 3 ===")
    # question_3_test_case(scaled_data_dict)
    
    # Work through the questions
    print("=== Question 1 ===")
    question_1()
    print("=== Question 2 ===")
    # question_2(scaled_data_dict)
    print("=== Question 3 ===")
    nolev_dict, lev_dict = question_3(scaled_data_dict)
    print("=== Question 4 ===")
    question_4(data_dict, lev_dict, nolev_dict)
    print("=== Question 5 ===")
    question_5(scaled_data_dict, nolev_dict, lev_dict)
    print("=== Question 6 ===")
    # question_6(data_dict)
    

# Ensures that the file only runs when ran as a script instead of being imported
if __name__ == '__main__':
    np.random.seed(1234)
    main()