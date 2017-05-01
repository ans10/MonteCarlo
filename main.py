import numpy as np
from mcmc import *
from utils import *
import pickle as pkl
import os
from error import *

pathtosave = os.getcwd()
tfilepath = os.getcwd()
#t = np.random.uniform(low=0.0,high=1.0,size=100)
tfile = open(os.path.join(tfilepath,"t1000.pkl"),"rb")
print tfile
t = pkl.load(tfile)
print t.shape
#print t.shape
tfile.close()
size = 3
sigma = 0.1
minA = 0.0
maxA = 1.0
domainA = (minA,maxA)
min_decay = 0.0
max_decay = 1.0
domain_decay = (min_decay,max_decay)
A = np.array([0.3,0.6,0.9])
initialA = np.array([0.2,0.5,0.8])
initialdecay = np.array([0.2,0.5,0.8])
initialsigma = 0.5
decay_factor = np.array([0.3,0.6,0.9])
(f,Y) = generate_fake_data(A,decay_factor,t,sigma)
#print f
#print Y
likelihood = calculate_log_likelihood(Y,f,sigma)
print likelihood
initial = (initialA,initialdecay,initialsigma)
domain = (domainA,domain_decay)
"""
print "Started with experiment: "
exp6a = sampler(Y,t,initial,domain,50000,0.1,verbose=False)
pkl.dump(exp6a,open(os.path.join(pathtosave,"exp6a.pkl"),"wb"))
"""

print "Started with experiment: "
exp11a = sampler(Y,t,initial,domain,1000,0.5,"exp11a.log",verbose=True)
pkl.dump(exp11a,open(os.path.join(pathtosave,"exp11a.pkl"),"wb"))


print "Started with experiment: "
exp11b = sampler(Y,t,initial,domain,1000,0.3,"exp11b.log",verbose=True)
pkl.dump(exp11b,open(os.path.join(pathtosave,"exp11b.pkl"),"wb"))

"""
print "Started with experiment: "
exp8c = sampler(Y,t,initial,domain,samples=500000,proposal_width = 0.5,verbose=True)
pkl.dump(exp8c,open(os.path.join(pathtosave,"exp8c.pkl"),"wb"))
print "Started with experiment: "
exp8d = sampler(Y,t,initial,domain,samples=500000,proposal_width = 0.5,verbose=True)
pkl.dump(exp8d,open(os.path.join(pathtosave,"exp8d.pkl"),"wb"))
"""
