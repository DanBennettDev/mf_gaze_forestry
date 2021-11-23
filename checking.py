import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from tabulate import tabulate
import matplotlib.transforms as mtransforms


#   __ = preAnalysisInspection(inputTable=details, 
#                           conditionIdx="condition", 
#                           participantIdx="userID", 
#                           dataCol="VAS_writing", 
#                           within=True
#                          )


def preAnalysisInspection(inputTable, conditionIdx, participantIdx, dataCol, dataColName=None, within = False, bincount=20):
    mpl.style.use("tableau-colorblind10")  
    mpl.rcParams['lines.linewidth'] = 2
    idx = pd.IndexSlice

    conditions = np.append(inputTable.index.get_level_values(conditionIdx).unique().values, "all")
    theIndex = inputTable.index.names
    output = pd.DataFrame(columns=["min", "max", "iqr", "mean", "std", "count", "isNormal", "shapiroVal","Outliers(3IQR)"], index=conditions)
    datasets=[]

    for cond in conditions:
        if cond =="all":
            subset = inputTable[dataCol]
        else:
            subset = inputTable.query(conditionIdx + " == '" +cond + "'")[dataCol]
            datasets.append(subset)
        output.loc[idx[cond],"min"] = subset.min()
        output.loc[idx[cond],"max"] = subset.max()
        output.loc[idx[cond],"iqr"] = np.subtract(*np.percentile(subset, [75, 25]))
        output.loc[idx[cond],"mean"] = subset.mean()
        output.loc[idx[cond],"std"] = subset.std()    
        output.loc[idx[cond],"count"] = int(subset.count())
        stat, p = stats.shapiro(subset)
        output.loc[idx[cond],"shapiroVal"] = p
        output.loc[idx[cond],"isNormal"] = p > 0.05
        outlierMin = np.percentile(subset, 25) - 3 * output.loc[idx[cond],"iqr"]  
        outlierMax = np.percentile(subset, 75) + 3 * output.loc[idx[cond],"iqr"]  
        output.loc[idx[cond],"Outliers(3IQR)"] = int(subset[subset<outlierMin].count() + subset[subset>outlierMax].count())
    
    if within:
        within = pd.DataFrame(columns=["min", "max", "iqr", "mean", "std", "count", "isNormal", "shapiroVal","Outliers(3IQR)"], index=["within"])
        avg = subset.groupby([participantIdx, conditionIdx]).mean()
        c1 = pd.DataFrame(avg).query(conditionIdx + " == '" +conditions[0] + "'")[dataCol].values
        c2 = pd.DataFrame(avg).query(conditionIdx + " == '" +conditions[1] + "'")[dataCol].values
        diffs = pd.DataFrame(c1-c2)[0]
        within.loc[idx["within"],"min"] = diffs.min()
        within.loc[idx["within"],"max"] = diffs.max()
        within.loc[idx["within"],"iqr"] = np.subtract(*np.percentile(diffs, [75, 25]))
        within.loc[idx["within"],"mean"] = diffs.mean()
        within.loc[idx["within"],"std"] = diffs.std()    
        within.loc[idx["within"],"count"] = int(diffs.count())
        stat, p = stats.shapiro(diffs)
        within.loc[idx["within"],"shapiroVal"] = p
        within.loc[idx["within"],"isNormal"] = p > 0.05
        outlierMin = np.percentile(diffs, 25) - 2 * within.loc[idx["within"],"iqr"]  
        outlierMax = np.percentile(diffs, 75) + 2 * within.loc[idx["within"],"iqr"]  
        within.loc[idx["within"],"Outliers(3IQR)"] = int(diffs[diffs<outlierMin].count() + diffs[diffs>outlierMax].count())
        output = output.append(within, sort=False)
        
        print("------------ Within Subject Plots -----------------")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(5)
        fig.set_figwidth(15)
        bins = np.linspace( diffs.min(), diffs.max(), bincount)

        for i, d in enumerate(datasets):
            ax1.hist(diffs, bins= bins, alpha=0.8, edgecolor = 'black')
            
        sns.boxplot(data=diffs, ax=ax2)
        plt.show()   
    
    print("------------ Condition Plots -----------------")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    bins = np.linspace(output.loc[idx['all'], "min"], output.loc[idx['all'], "max"], 20)

    for i, d in enumerate(datasets):
        ax1.hist(d, bins=bins, alpha=0.6, edgecolor = 'black', label =conditions[i])
    # print(d)
    d = pd.DataFrame(pd.concat(datasets))
    d.reset_index(inplace=True)
    sns.boxplot(data=d, ax=ax2, x=conditionIdx, y=dataCol)

    ax1.legend(loc='upper right')
    if dataColName == None:
        ax1.set_xlabel(dataCol)
    else:
        ax1.set_xlabel(dataColName)        
    plt.show()
    
    print(tabulate(output, headers='keys', tablefmt='psql', floatfmt=".4f"))      
    # Levene test for equal variance
    t, p = stats.levene(datasets[0],datasets[1])
    print("Levene test result: t: ","{0:.4f}".format(t) , "p: ", "{0:.4f}".format(p),"- populations ", 'are not' if p < 0.05 else 'are', "of equal variance")
    
    
    return output





def cohen_d(sample1, sample2):
    return(np.mean(sample1) - np.mean(sample2)) / (np.sqrt((np.std(sample1) ** 2 + np.std(sample2) ** 2) / 2))


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
