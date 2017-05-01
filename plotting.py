from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors as mcolors
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from autocorr import *
import os
import sys
import pickle as pkl
import numpy as np

pathtosave = os.getcwd()
def analyze(ans,truevalue,expname,size,burnin=0):
    temp1 = ans[0][burnin:,:]
    temp2 = ans[1][burnin:,:]
    temp3 = ans[2][burnin:]
    temp4 = ans[3][burnin:]
    ansfinal =(temp1,temp2,temp3,temp4)
    plothistogram(ansfinal,size,expname)
    plot_posterior_value(ansfinal,truevalue,0,expname)
    plot_posterior_value(ansfinal,truevalue,1,expname)
    plot_posterior_value(ansfinal,truevalue,2,expname)

def analyze_autocorrelation(ans,expname,size,burnin=0):
    temp1 = ans[0][burnin:,:]
    temp2 = ans[1][burnin:,:]
    temp3 = ans[2][burnin:]
    temp4 = ans[3][burnin:]
    ansfinal =(temp1,temp2,temp3,temp4)
    for ij in range(0,size):
        arr = calculate_autocorrelation(ansfinal[0][:,ij])
        print_autocorrelation_graph(arr,expname,label="A"+str(ij+1))
        act = calculate_autocorrelation_time(arr)
        print "Auto correlation time for A"+str(ij+1)+" is "+str(act)
    for ij in range(0,size):
        arr = calculate_autocorrelation(ansfinal[1][:,ij])
        print_autocorrelation_graph(arr,expname,label="lambda"+str(ij+1))
        act = calculate_autocorrelation_time(arr)
        print "Auto correlation time for lambda"+str(ij+1)+ " is "+str(act)

    print "Auto correlation time for sigma is "+str(act)

def plothistogram(ansfinal,size,expname):

    for ij in range(0,size):
        label = "A"+str(ij+1)
        n, bins, patches = plt.hist(ansfinal[0][:,ij],facecolor='green')
        plt.title("Histogram for "+label)
        plt.savefig(expname+label+"hist.png")
        plt.close()
    for ij in range(0,size):
        label = "lambda"+str(ij+1)
        n, bins, patches = plt.hist(ansfinal[1][:,ij],facecolor='green')
        plt.title("Histogram for "+label)
        plt.savefig(expname+label+"hist.png")
        plt.close()
    label = "sigma"
    n, bins, patches = plt.hist(ansfinal[2],facecolor='green')
    plt.title("Histogram for sigma")
    plt.savefig(expname+label+"hist.png")
    plt.close()
def print_autocorrelation_graph(arr,expname,label):

    plt.figure()
    x = range(0,len(arr))
    y = arr
    plt.fill_between(x,0,y)
    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.title("Autocorrelation graph of "+label)
    plt.plot(x,y,label=label)
    plt.savefig(expname+label+"autocorr.png")
    plt.close()
def plot_posterior_value(ansinput,truevalue,value,expname,burnin=0):
    ans = (ansinput[0][burnin:,:],ansinput[1][burnin:,:],ansinput[2][burnin:],ansinput[3][burnin:])
    colors = ['red','green','blue','yellow','black','gold','grey','indigo','pink','khaki']
    if(value==0):

        for ij in range(0,ans[value][0].shape[0]):
            label='A'+str(ij+1)
            plt.plot(ans[value][:,ij],c=colors[ij])
            plt.axhline(y=truevalue[value][ij],c='b',linestyle='dashed',
            label="True Parameter value")
            plt.xlabel("iterations")
            plt.ylabel("posterior value")
            plt.title("Posterior values of "+label)
            plt.savefig(expname+label+"post.png")
            plt.close()
    if(value==1):
        for ij in range(0,ans[value][0].shape[0]):
            label='lambda'+str(ij+1)
            plt.plot(ans[value][:,ij],c=colors[ij])
            plt.axhline(y=truevalue[value][ij],c='b',linestyle='dashed',
            label="True Parameter value")
            plt.xlabel("iterations")
            plt.ylabel("posterior value")
            plt.title("Posterior values of "+label)
            #plt.legend(bbox_to_anchor=(0., -1.0))
            plt.savefig(expname+label+"post.png")
            plt.close()
    if(value==2):
        label = 'sigma'
        plt.plot(ans[value],c=colors[0],label=label)
        plt.axhline(y=truevalue[value],c='b',linestyle='dashed',
        label="True Parameter value")
        plt.xlabel("iterations")
        plt.ylabel("posterior value")
        plt.title("Posterior values of sigma")
        #plt.legend(bbox_to_anchor=(0., -1.))
        plt.savefig(expname+label+"post.png")
        plt.close()
def main():
    #arg 1 pickle file
    #arg 2 experiment no
    #arg 3 auto_corr = 1 posterior = 0
    #arg 4 size of the parameters
    picklefile = sys.argv[1]
    experiment_name = sys.argv[2]
    auto_corr = int(sys.argv[3])
    size = int(sys.argv[4])
    truevalue_picklefile = sys.argv[5]
    f = open(picklefile)
    ans = pkl.load(f)
    f.close()
    f = open(truevalue_picklefile)
    truevalue = pkl.load(f)
    f.close()
    if(auto_corr==0):
        analyze(ans,truevalue,sys.argv[2],size)
    else:
        analyze_autocorrelation(ans,sys.argv[2],size)

if __name__ == "__main__":
    main()
