# DFA processing code from https://github.com/dokato/dfa

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_rms(x, scale):
    """
    Root Mean Square in windows with linear detrending.
    
    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale* : int
        length of the window in which RMS will be calculaed
    Returns:
    --------
      *rms* : numpy.array
        RMS data in each window with length len(x)//scale
    """
    # making an array with data divided in windows
    shape = (int(x.shape[0]/scale), scale)
    X = np.lib.stride_tricks.as_strided(x,shape=shape)
    # vector of x-axis points to regression
    scale_ax = np.arange(scale)
    rms = np.zeros(X.shape[0])
    for e, xcut in enumerate(X):
        coeff = np.polyfit(scale_ax, xcut, 1)
        xfit = np.polyval(coeff, scale_ax)
        # detrending and computing RMS of each window
        rms[e] = np.sqrt(np.mean((xcut-xfit)**2))
    return rms

def dfa(x, scale_lim=[5,8], scale_dens=0.25, show=False):
    """
    Detrended Fluctuation Analysis - algorithm with measures power law
    scaling of the given signal *x*.
    More details about algorithm can be found e.g. here:
    Hardstone, R. et al. Detrended fluctuation analysis: A scale-free 
    view on neuronal oscillations, (2012).
    
    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale_lim* = [5,9] : list of lenght 2 
        boundaries of the scale where scale means windows in which RMS
        is calculated. Numbers from list are indexes of 2 to the power
        of range.
      *scale_dens* = 0.25 : float
        density of scale divisions
      *show* = False
        if True it shows matplotlib picture
    Returns:
    --------
      *scales* : numpy.array
        vector of scales
      *fluct* : numpy.array
        fluctuation function
      *alpha* : float
        DFA exponent
    """
    # cumulative sum of data with substracted offset
    y = np.cumsum(x - np.mean(x))
    scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
    fluct = np.zeros(len(scales))
    # computing RMS for each window
    for e, sc in enumerate(scales):
        fluct[e] = np.mean(np.sqrt(calc_rms(y, sc)**2))
    # fitting a line to rms data
    coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    if show:
        fluctfit = 2**np.polyval(coeff,np.log2(scales))
        plt.loglog(scales, fluct, 'bo')
        plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f'%coeff[0])
        plt.title('DFA')
        plt.xlabel(r'$\log_{10}$(time window)')
        plt.ylabel(r'$\log_{10}$<F(t)>')
        plt.legend()
        plt.show()
    return scales, fluct, coeff[0]



# DTB convenience method for running a batch
def runDFA(df, signalName="signal", isfractalGaussianNoise=False):
    i=0;   
    exponents = np.zeros(len(df.index.unique(level='trialID')))
    for trialID, trialData in df.groupby(level='trialID'):    
        scales, fluct, esta = dfa(trialData[signalName], show=0)
        exponents[i]=esta
        i+=1
    HurstAvg = exponents.mean()
    # Per Eke et al 2002 - as first step DFA performs cumSum on signal and subtracts mean.
     # if the input is a fGn signal then this will result in a fractional Brownian motion (fBm) time series 
    # and the exponents calculated will approximate the Hurst exponent H
    # if the input is fBm series, then cumSum step results in a summed fBm signal and the exponents calculated
    # will approximate H+1.    
    if isfractalGaussianNoise==False:
        HurstAvg-=1
    print("mean:", exponents.mean(), "std:", exponents.std(), "min/max:", exponents.min(),"/",exponents.max())
    print("mean Hurst exponent:", HurstAvg)
    if exponents.mean() < 1:
        print("integrate input before WTMM")
    else:
        print("no need to integrate input before WTMM")
    return exponents

