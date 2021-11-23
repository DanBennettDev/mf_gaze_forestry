

import datetime
import pandas as pd
import numpy as np
from scipy import integrate, signal, stats
import subprocess
from os import listdir, path
from os.path import isfile, join
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, glob
from pathlib import Path
import glob
import math
from nolitsa import surrogates
import itertools
from tabulate import tabulate
from MFDFA import MFDFA as mfdfalib
from MFDFA import IMFs


SCALECOUNT = 32
# processing functions


def makeUniformLength(df, length=-1, truncateFromStart=0):
    if length==-1:
        minSize = 9999999999999999

        for trialID, trialData in df.groupby(level='trialID'):
            l = len(trialData)
            if(l < minSize):
                minSize = l          
    else:
        minSize = int(length)
    return df.loc[df.index.isin(range(truncateFromStart,minSize), level="sampleNo"), :].copy(), minSize-truncateFromStart



def integrateSignal(df):
    cp = df.copy() 
    cp["signalNoDC"] = 0 
    cp["signalIntegral"]=0
    for trialID, trialData in cp.groupby(level='trialID'):
        cp.loc[cp.index.isin([trialID], level="trialID"), "signalIntegral"] = \
            (trialData["signal"] - trialData["signal"].mean()).cumsum()
    return cp



def importAndAnalyseHolderSpectra(wttm_worklist):
    # import the results
    holderFile = os.path.splitext(wttm_worklist)[0] + "_out.txt"
    cols = ["key", 'h','dh']
    holder = pd.read_csv(holderFile, header=None, skiprows=0, sep=" ", names=cols)
    # parse filenames
    #strip filepath
    holder['key']  = holder['key'].str.split(pat = "/").str[-1]
    # parse filepath into labels
    holder["condition"] = holder['key'].str.split(pat = "_").str[2]
    holder["type"] = 'original'

    holder.loc[holder['key'].str.split(pat = "_").str[-1].str[0] == 's', "type"] = \
                        holder['key'].str.split(pat = "_").str[-1].str[0:4]
    holder['participant'] = holder['key'].str.split(pat = "_").str[0]
    holder['session'] = holder['key'].str.split(pat = "_").str[1]
    holder['key'] = holder['participant'] + "_" + holder['session']
    return analyseHolderSpectra(holder)

    
def analyseHolderSpectra(holder):
    # calculate the spectrum width for each trial and surrogate
    widths = pd.DataFrame(holder.groupby(['key', "type", "condition", "participant", "session"]).apply(lambda x: x['h'].max() - x['h'].min()), columns=['width'])
    widths["surrZindex"] = 0
    widths["outsideConfInterval"] = False

    # for each trial, calculate the degree to which original is differentiated from surrogates
    # Quantified by both z score and confidence interval
    for trialID, dt in widths.groupby(level='key'):
        vals = dt['width'].values
        mean = dt['width'].mean()
        std = dt['width'].std()
        originalWidth = widths.loc[(trialID, 'original'), 'width'][0]
        # widths.loc[(trialID, 'original'), 'surrZindex'] = abs(originalWidth - mean) / std
        widths.loc[(trialID, 'original'), 'surrZindex'] = (originalWidth - mean) / std

        confidenceInterval = stats.t.interval(0.95, len(vals)-1, loc=mean, scale=stats.sem(vals))
        if(originalWidth > confidenceInterval[1] or  originalWidth < confidenceInterval[0]):
            widths.loc[(trialID, 'original'), "outsideConfInterval"] = True
    return widths, holder


def summariseResults(widths):
    # look at rows for originals data only
    idx = pd.IndexSlice
    widths = widths.loc[idx[:, 'original', :], :]
    widths.reset_index(level=1, drop=True, inplace=True)
    ind = widths.index.unique(level='condition').append(pd.Index(["Overall"]))
    
    
    col = ["mean", "std", "t-stat", "confidenceInterval"]

    summary = pd.DataFrame(columns=col, index=ind)

    summary.loc['Overall', 'mean'] = widths.loc[:, 'width'].mean()
    summary.loc['Overall', 'std'] = widths.loc[:, 'width'].std()
    summary.loc['Overall', 't-stat'] = widths.loc[:, 'surrZindex'].sum()
    summary.loc['Overall', 'confidenceInterval'] = \
                float(len(widths.loc[widths.loc[:, 'outsideConfInterval']==True])) / float(len(widths))

    for i in range(0, ind.size-1):
        summary.loc[ind[i], 'mean'] = widths.loc[idx[:, ind[i]], 'width'].mean()
        summary.loc[ind[i], 'std'] = widths.loc[idx[:, ind[i]], 'width'].std()
        summary.loc[ind[i], 't-stat'] = widths.loc[idx[:, ind[i]], 'surrZindex'].sum()
        summary.loc[ind[i], 'confidenceInterval'] = widths.loc[idx[:, ind[i]],'outsideConfInterval']\
                                    .loc[widths.loc[:, 'outsideConfInterval']==True].count() \
                                        / widths.loc[idx[:, ind[i]],'outsideConfInterval'].count()

    return widths, summary





def generateSurrogates(data, surrogateCount, sig):
    idx = pd.IndexSlice
    allData = [] 
    for trialID, trialData in data.groupby(level='trialID'):
        signal =data.loc[idx[trialID, :], sig].values
        for i in range(0,surrogateCount):
            data.loc[idx[trialID, :], "surr"+str(i)] = surrogates.iaaft(signal)[0]
    return data.copy()
    
    

# versions using the mfdfa lib instead of jurica's implementations -
# the fact that he uses old python and allows errors to appear makes me concerned
# that the functions may not be as precise and rigorous as they could be -double checking
# against a peer reviewed library
def calc_mfdfa(signal, qs, scales, order, window):
    # The library does this internally - we do it here to align arrays, for the calculations below.
    qs = qs[(qs < -.1) + (qs > .1)]
    lag, dfa = mfdfalib(signal, lag = scales, q = qs, order = order, extensions = {'window': window})
    model = np.polyfit(np.log(lag), np.log(dfa),1, full=True)
    coeff = model[0]
    Hq = coeff[0]
    ssr = model[1]  # sum of squared residuals
    diff = np.log(dfa) - np.mean(np.log(dfa))
    r2 = 1 - ssr/(diff ** 2).sum()  
    tq = Hq*qs - 1
    hq = np.diff(tq)/(np.diff(qs))
    Dq = (qs[:-1]*hq) - tq[:-1]
    return dfa, Hq, hq, tq, Dq, lag, r2, coeff
    
  

# early idea for scoring improvement in surrogate test - takes into account both difference in differentiation from
# surrogates and consistency across the sample - based on the number that pass the 95% confidence interval 
# that they are distinguishable from the surrogate population.
# Acceptable loss defines an acceptable tradeoff of consistency against absolute improvement.
# It is the proportion of samples the user will accept failing to pass the 95% confidence interval
# in order to gain a 100% overall improvement in differentiation from the surrogate population
def surrogateScore(cumZScore, confidenceIntervalPercent, acceptableLoss=.03):
    return (cumZScore * math.pow(confidenceIntervalPercent, math.log(.5) / math.log(1-acceptableLoss) ) )



def generateScaleList(length, scaleCount = SCALECOUNT, minscale = 2):
    # smallest scale must be order+1 (m points required to fit a polynomial of order m)
    # largest should be <= length/4 (need 4 windows to get adequate statistics)
    smallestScale = math.log(minscale,2)
    largestScale = math.log(length/4,2)
    # generate equally spaced range of scales, rounded to integers. 
    # This will generate duplicates, which we remove to avoid wasted computation
    # so we will get less than the scalecount requested in many cases
    return np.unique(np.logspace(smallestScale, largestScale, scaleCount, base=2.0).astype(int))


#https://github.com/ajschumacher/cles
def cles(lessers, greaters):
    """Common-Language Effect Size
    Probability that a random draw from `greater` is in fact greater
    than a random draw from `lesser`.
    Args:
      lesser, greater: Iterables of comparables.
    """
    if len(lessers) == 0 and len(greaters) == 0:
        raise ValueError('At least one argument must be non-empty')
    # These values are a bit arbitrary, but make some sense.
    # (It might be appropriate to warn for these cases.)
    if len(lessers) == 0:
        return 1
    if len(greaters) == 0:
        return 0
    numerator = 0
    lessers, greaters = sorted(lessers), sorted(greaters)
    lesser_index = 0
    for greater in greaters:
        while lesser_index < len(lessers) and lessers[lesser_index] < greater:
            lesser_index += 1
        numerator += lesser_index  # the count less than the greater
    denominator = len(lessers) * len(greaters)
    return float(numerator) / denominator



def MFDFA_Process2(data, scales, qs, order=1, surrogateCount=100,  sig = 'signal', window = 32, returnOnlyHs = True):
    allHolders = []    
    allTaus = []
    allDiagnostics = []
    idx = pd.IndexSlice    
    for trialID, trialData in data.groupby(level='trialID'):
        signal =data.loc[idx[trialID, :], sig].values
        dfa, Hq, hq, tq, Dq, lag, r2, coeff = calc_mfdfa(signal, qs, scales, order, window)
        hs = pd.DataFrame({'h': hq, 'dh': Dq}, columns=['h', 'dh'])
        condition = trialData["condition"][0]
        hs['condition'] = condition
        hs['key'] = trialID
        hs["type"] = "original"
        allHolders.append(hs)
        if(returnOnlyHs==False):
            ts = pd.DataFrame({'tq': tq, 'Hq': Hq}, columns=['tq', 'Hq'])
            ts['condition'] = condition
            ts['key'] = trialID
            allTaus.append(ts)
            # allDiagnostics.append((trialID,condition, lag, np.array(dfa), np.array(r2), np.array(coeff)))
            diagData = {"trialID":trialID,
                        "condition":condition, 
                        "fq":[dfa], 
                        "r2":[r2], 
                        "coeff":[coeff]}
            diagDF = pd.DataFrame(data=diagData)
            allDiagnostics.append(diagDF)
        for i in range(0,surrogateCount):
            s =  data.loc[idx[trialID, :], "surr"+str(i)].values
            _, _, hq, _, Dq, _, _, _ = calc_mfdfa(s, qs, scales, order, window)
            shs = pd.DataFrame({'h': hq, 'dh': Dq}, columns=['h', 'dh'])
            shs['condition'] = condition
            shs['key'] = trialID
            shs["type"] = "s" + str(i).zfill(3)
            allHolders.append(shs)
    holders = pd.concat(allHolders, sort=False)
    holders['participant'] = holders['key'].str.split(pat = "_").str[0]
    holders['session'] = holders['key'].str.split(pat = "_").str[1]
    if(returnOnlyHs==True):
        return holders
    else:
        taus = pd.concat(allTaus, sort=False)
        taus['participant'] = taus['key'].str.split(pat = "_").str[0]
        taus['session'] = taus['key'].str.split(pat = "_").str[1]
        return [holders, taus, pd.concat(allDiagnostics)]
