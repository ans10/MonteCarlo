import numpy as np
import scipy
from scipy.stats import norm
from copy import copy
from scipy.stats import multivariate_normal as mvnorm
import pickle as pkl
from error import *
import math
def eval_f(A,decay,t):
    m = A.shape[0]
    f = []
    for ti in t:
        temp = 0.
        temp = (A*np.exp(-1.*decay*ti)).sum()
        f.append(temp)
    return np.array(f)

def calculate_likelihood(Y,f,sigma):
    return norm(f,sigma).pdf(Y).prod()

def calculate_log_likelihood(Y,f,sigma):
    np.seterr(invalid='raise')
    if(sigma<=0):
        return -1.0*np.infty
    return norm(f,sigma).logpdf(Y).sum()

def eval_log_likelihood(data_points, r, sigma):
    val = ((r - data_points)*(r - data_points)) / (2 * sigma * sigma )
    val = np.exp(-1 * val)
    norm = math.sqrt(2 * math.pi * sigma * sigma)
    val = val / norm
    val = np.log(val)
    t = 0.0
    for item in val:
        t = t + item
    return t

def get_proposal(method,mu,sigma):
    if(method=="ig"):
        proposal = np.random.normal(mu,sigma)
        if(type(proposal) != np.ndarray):
            proposal = np.array([proposal])
        return proposal
    raise ValidProposalError("The proposal method chosen is not valid")


def get_prior(method,A,decay_factor,domainA,domain_decay,sigma):

    if(method=='c'):
        ans = 0.0
        for ij in range(0,A.shape[0]):
            if(domainA[0]<=A[ij]<=domainA[1] and
               domain_decay[0]<=decay_factor[ij]<=domain_decay[1] and
               0.0<=sigma<=1.0):
               ans = 0.0
            else:
                ans = -1.0*np.infty
                break


        return ans

    raise ValidPriorError("The prior method chosen is not valid")

def generate_fake_data(A,decay_factor,t,sigma):
    f = eval_f(A,decay_factor,t)
    xi = np.random.normal(0,1.0,t.shape[0])
    Y = f + sigma*xi
    return (f,Y)
