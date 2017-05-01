from utils import *
import numpy as np
from error import *
import logging
def initialize(Y,t,initial,domain,prior='c',verbose=False):
    (currentA,currentdecay,currentsigma) = initial
    (domainA, domaindecay) = domain
    f = eval_f(currentA,currentdecay,t)
    current_likelihood = calculate_log_likelihood(Y,f,currentsigma)
    if(verbose):
        logging.info("current_likelihood "+str(current_likelihood))
    current_prior = get_prior(prior,currentA,currentdecay,domainA,domaindecay,currentsigma)
    current_p = current_likelihood + current_prior
    return current_p

def MCMC_step(Y,t,currentA,currentdecay,currentsigma,
              current_p,domain,proposal_width,
              proposal="ig",prior="c",verbose=False):

    (domainA,domain_decay) = domain
    #finding new parameters
    newA = get_proposal(proposal,currentA,proposal_width)
    newdecay = get_proposal(proposal,currentdecay,proposal_width)
    newsigma = get_proposal(proposal,currentsigma,proposal_width)

    #calculating f with new parameters
    newf = eval_f(newA, newdecay, t)

    #calculating likelihood with new f and the data
    new_likelihood = calculate_log_likelihood(Y,newf,newsigma)

    #calculating the prior for the given values of parameters
    new_prior = get_prior(prior,newA,newdecay,domainA,domain_decay,newsigma)

    #calculating the probability = likelihood + prior
    new_p = new_likelihood + new_prior

    #acceptance condition metropolis hastings
    accept_ratio = min(1.0,np.exp(new_p-current_p))
    random_value = np.random.random()
    if(random_value<accept_ratio):
        if(verbose):
            logging.info("New probability = "+str(new_p))
            logging.info("Old probability = "+str(current_p))

        current_p = new_p
        return (newA,newdecay,newsigma,current_p)
    else:
        return (currentA,currentdecay,currentsigma,current_p)

def sampler(data,t,initial,domain,samples,
            proposal_width, logfilename,proposal="ig", prior="c",verbose=False):
    logging.basicConfig(filename=logfilename,filemode='w',level=logging.INFO)
    A_posterior = []
    decay_posterior = []
    sigma_posterior = []
    samples_accepted = 0
    no_samples = 0
    np.seterr(invalid='raise')
    #initialization step
    current_p = initialize(data,t,initial,domain,verbose=verbose)
    if(verbose):
        logging.info("Current p "+str(current_p))
    (initialA,initialdecay,initialsigma) = initial
    A_posterior.append(initialA)
    decay_posterior.append(initialdecay)
    sigma_posterior.append(initialsigma)
    samples_accepted = samples_accepted + 1
    no_samples = no_samples + 1

    while(no_samples<samples):

        currentA = A_posterior[len(A_posterior)-1]
        currentdecay = decay_posterior[len(decay_posterior)-1]
        currentsigma = sigma_posterior[len(sigma_posterior)-1]
        nextvalue = MCMC_step(data,t,currentA,currentdecay,currentsigma,
                      current_p,domain,proposal_width,
                      proposal="ig", prior="c",verbose=verbose)


        A_posterior.append(nextvalue[0])
        decay_posterior.append(nextvalue[1])
        sigma_posterior.append(nextvalue[2])
        if(nextvalue[3]!=current_p):
            samples_accepted = samples_accepted + 1
        current_p = nextvalue[3]


        if((no_samples%100000)==0 and verbose):
            logging.info(str(no_samples)+"reached!")
        no_samples = no_samples + 1
        if(verbose):
            logging.info("Done with generating "+str(no_samples)+ " samples!")

    logging.info("Number of accepted samples: "+str(samples_accepted))
    logging.info("Acceptance ratio: "+str(float(samples_accepted)/float(no_samples)))
    return np.array(A_posterior),np.array(decay_posterior),np.array(sigma_posterior),np.array(range(no_samples))
