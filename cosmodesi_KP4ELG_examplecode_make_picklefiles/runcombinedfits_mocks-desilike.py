# file to get combined fits to the mocks 

import sys
import os
import numpy as np
import scipy as sp
import pandas as pd
import pickle
from scipy.stats import gaussian_kde 
import emcee 
from getdist import plots, MCSamples
from desilike.samples import Chain, plotting

# get command line variables 
realisation_number = int(sys.argv[1])

if realisation_number == 25: # 0 - 24 are individual realisations 
    realisation_number = 'mean'
    
dataframes_kdes = {} 

# setting up data / variables for running fits 
list_paramswanted = [r'$\alpha$', r'$\epsilon$', r'$\beta_{\phi(N_{\mathrm{eff}})}$', 'weights']    

labels = [
          r'$\alpha_{\mathrm{LRG1}}$', 
          r'$\epsilon_{\mathrm{LRG1}}$',
          r'$\alpha_{\mathrm{LRG2}}$', 
          r'$\epsilon_{\mathrm{LRG2}}$',
          r'$\alpha_{\mathrm{LRG3ELG1}}$', 
          r'$\epsilon_{\mathrm{LRG3ELG1}}$',
           r'$\alpha_{\mathrm{ELG2}}$', 
          r'$\epsilon_{\mathrm{ELG2}}$', 
          r'$\beta_{N_{\mathrm{eff}}}$'
         ]

    
    
print('command line variables and basic variables set up') 


# path to chains for desilike fits - Hugo  
path = '/global/cfs/projectdirs/desi/users/hugoriv/BAO_Neff/chains results/'

    
# LRG1
nchains = 7
burnin=0.3
thin=1
chains = {}

tracers = ['LRG_z0.4-0.6', 'LRG_z0.6-0.8', 'LRG+ELG_z0.8-1.1', 'ELG_LOPnotqso_z1.1-1.6']
nameshort = ['LRG1', 'LRG2', 'LRG3ELG1', 'ELG2']
for i, t in enumerate(tracers): 

    chains_path = (f'/global/cfs/projectdirs/desi/users/hugoriv/desilike/SecondGen_Neff'
    f'/desilike2D_{realisation_number}_sampling_xi_secgenV1_2_c000_recsym_GCcomb_' + t)
    chain_power_mock = Chain.concatenate([Chain.load(chains_path + f'_chain{j}.npy').remove_burnin(burnin)[::thin] for j in range(nchains)])
    # Parameters of the chains
    beta_shift=np.array(chain_power_mock['baoshift'].flatten())
    qpar=np.array(chain_power_mock['qpar'].flatten())
    qper=np.array(chain_power_mock['qper'].flatten())
    qiso = qpar**(1./3.) * qper**(2./3.)
    eps = (qpar / qper)**(1.0/3.0) - 1.0
    chain_mock=np.column_stack((qiso,eps,beta_shift))
    chains[nameshort[i]] = chain_mock

print('desilike data loaded')  


# set up the KDEs (interpolation of the likelihood) for each tracer 

for item in chains: 
    
    chain = chains[item]
    
    df = pd.DataFrame(chain, columns=[r'$\alpha$', r'$\epsilon$', 
                                      r'$\beta_{\phi}$'])

    dataframes_kdes[item] = [] 
    dataframes_kdes[item].append(item) 

    kde = gaussian_kde(np.vstack([
            df[r'$\alpha$'].to_numpy(),
            df[r'$\epsilon$'].to_numpy(),
            df[r'$\beta_{\phi}$'].to_numpy()]))

    dataframes_kdes[item].append(kde) 
    
print(dataframes_kdes)
print(sys.getsizeof(dataframes_kdes)) 

print('kde objects set up') 
# define a function for the log of the combined likelihood of all the tracers    

def log_prob_betaphaseshift(x):
    
    lrg1 = dataframes_kdes['LRG1'][1]([x[0], x[1], x[8]])[0]
    lrg2 = dataframes_kdes['LRG2'][1]([x[2], x[3], x[8]])[0]
    lrg3 = dataframes_kdes['LRG3ELG1'][1]([x[4], x[5], x[8]])[0]
    elg2 = dataframes_kdes['ELG2'][1]([x[6], x[7], x[8]])[0]
        
    if lrg1 <= 0.0 or abs(x[0])-1.0 >= 0.2 or abs(x[1]) >= 0.2 or abs(lrg1) == np.inf: 
        lrg1 = -np.inf 
    else:
        lrg1 = np.log(lrg1)
    
    if lrg2 <= 0.0 or abs(x[2])-1.0 >= 0.2 or abs(x[3]) >= 0.2 or abs(lrg2) == np.inf: 
        lrg2 = -np.inf 
    else:
        lrg2 = np.log(lrg2)
        
    if lrg3 <= 0.0 or abs(x[4])-1.0 >= 0.2 or abs(x[5]) >= 0.2 or abs(lrg3) == np.inf: 
        lrg3 = -np.inf 
    else:
        lrg3 = np.log(lrg3)
    
    if elg2 <= 0.0 or abs(x[6])-1.0 >= 0.2 or abs(x[7]) >= 0.2 or abs(elg2) == np.inf: 
        elg2 = -np.inf 
    else:
        elg2 = np.log(elg2)
        
        
    if abs(lrg1) == np.inf or abs(lrg2) == np.inf or abs(lrg3) == np.inf or abs(elg2) == np.inf: #  or abs(qso) == np.inf or abs(bgs) == np.inf:
        logl = -np.inf 
    elif x[8] > 10 or x[8] < -8.0:
        logl = -np.inf
    else: 
        logl = elg2 + lrg1 + lrg2 + lrg3 # + qso + bgs 
        
    return logl 

print('running MCMC') 

# now run an MCMC fit to the combined likelihood of the KDES (using previously defined function) 
dim = 9
np.random.seed(42)
nwalkers = 32                                                                                          
p0 = np.array([ 
               np.random.uniform(0.99, 1.01, nwalkers),  np.random.uniform(-0.01, 0.01, nwalkers), 
               np.random.uniform(0.99, 1.01, nwalkers),  np.random.uniform(-0.01, 0.01, nwalkers), 
               np.random.uniform(0.99, 1.01, nwalkers),  np.random.uniform(-0.01, 0.01, nwalkers), 
               np.random.uniform(0.99, 1.01, nwalkers),  np.random.uniform(-0.01, 0.01, nwalkers), 
               np.random.uniform(0.99, 1.01, nwalkers)
                 ]).T

# We'll track how the average autocorrelation time estimate changes
max_n = 30000
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf

sampler = emcee.EnsembleSampler(nwalkers, dim, log_prob_betaphaseshift)

# Now we'll sample for up to max_n steps
for sample in sampler.sample(p0, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau

chains_flat = sampler.get_chain(flat=True, discard=100)

df_fit = pd.DataFrame({labels[i]: chains_flat[:,i] for i in np.arange(len(chains_flat[0,:]))})

print('MCMC successful, saving chain to a file') 

# save to a file 
df_fit.to_csv("/pscratch/sd/a/abbew25/combinedfits_secondgen_mocks_v1_2_elgslrgsbaselineDESILIKE/poly_xi_postrecon_desilike" +"_" + str(realisation_number) + ".csv")
