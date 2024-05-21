# file to get combined fits to the mocks 

import sys
import os

# this is Cullan's code to run and plot the second gen mocks with all appropriate settings 

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../Barry/")
from barry.samplers import NautilusSampler
from barry.config import setup
from barry.models import PowerBeutler2017, CorrBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
from barry.fitter import Fitter
import numpy as np
import scipy as sp
import pandas as pd
from barry.models.model import Correction
from barry.utils import weighted_avg_and_cov
import pickle
from scipy.stats import gaussian_kde 
import emcee 


# get command line variables 
realisation_number = int(sys.argv[1])
broadband_method = sys.argv[2]
recon = sys.argv[3]
data = sys.argv[4]
dataframes_kdes = {} 

# setting up data / variables for running fits 
list_paramswanted = [r'$\alpha$', r'$\epsilon$', r'$\beta_{\phi(N_{\mathrm{eff}})}$', 'weights']    
copy_list_BGSQSO = list_paramswanted.copy()
copy_list_BGSQSO.remove(r'$\epsilon$')

labels_pk = [r'$\alpha_{\mathrm{QSO}}$',
          r'$\alpha_{\mathrm{BGS}}$',
          r'$\alpha_{\mathrm{LRG1}}$', 
          r'$\epsilon_{\mathrm{LRG1}}$',
          r'$\alpha_{\mathrm{LRG2}}$', 
          r'$\epsilon_{\mathrm{LRG2}}$',
          r'$\alpha_{\mathrm{LRG3}}$', 
          r'$\epsilon_{\mathrm{LRG3}}$',
           r'$\alpha_{\mathrm{ELG2}}$', 
          r'$\epsilon_{\mathrm{ELG2}}$', 
          r'$\beta_{N_{\mathrm{eff}}}$'
         ]

labels_xi = [r'$\alpha_{\mathrm{QSO}}$',
          r'$\alpha_{\mathrm{BGS}}$',
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

if data == 'pk':
    labels = labels_pk
else:
    labels = labels_xi

    
    
print('command line variables and basic variables set up') 


# read in the fitter objects in order to read in the chains 
path = '/global/u1/a/abbew25/barryrepo/Barry/cosmodesi_KP4ELG_examplecode_make_picklefiles/plots/desi_kp4_SecondGen_'

# BGS - no epsilon fits 
with open(path + 'BGS_z01_04_' + data + '-reducedcov/output/desi_kp4_SecondGen_BGS_z01_04_' + data + '-reducedcov.fitter.pkl', 'rb') as pickle_file:
    fitter_BGS = pickle.load(pickle_file)

# QSOs - no epsilon fits 
with open(path + 'QSOs_z08_21_' + data + '-reducedcov/output/desi_kp4_SecondGen_QSOs_z08_21_' + data + '-reducedcov.fitter.pkl', 'rb') as pickle_file:
    fitter_QSO = pickle.load(pickle_file)
    
# LRG1
with open(path + 'LRGs_z04_06_' + data + '-reducedcov/output/desi_kp4_SecondGen_LRGs_z04_06_' + data + '-reducedcov.fitter.pkl', 'rb') as pickle_file:
    fitter_LRG1 = pickle.load(pickle_file)

# LRG2
with open(path + 'LRGs_z06_08_' + data + '-reducedcov/output/desi_kp4_SecondGen_LRGs_z06_08_' + data + '-reducedcov.fitter.pkl', 'rb') as pickle_file:
    fitter_LRG2 = pickle.load(pickle_file)

# LRG3 or LRG3+ELG1
if data == 'pk':
    with open(path + 'LRGs_z08_11_' + data + '-reducedcov/output/desi_kp4_SecondGen_LRGs_z08_11_' + data + '-reducedcov.fitter.pkl', 'rb') as pickle_file:
        fitter_LRG3 = pickle.load(pickle_file)
else:
    with open(path + 'ELGsLRGscombined_z08_11_' + data + '-reducedcov/output/desi_kp4_SecondGen_ELGsLRGscombined_z08_11_' + data + '-reducedcov.fitter.pkl', 'rb') as pickle_file:
        fitter_LRG3ELG1 = pickle.load(pickle_file)

# ELG2 
with open(path + 'ELGs_z11_16_' + data + '-reducedcov/output/desi_kp4_SecondGen_ELGs_z11_16_' + data + '-reducedcov.fitter.pkl', 'rb') as pickle_file:
    fitter_ELG2 = pickle.load(pickle_file)
    

fitters = {'QSO': fitter_QSO, 'BGS': fitter_BGS, 'LRG1': fitter_LRG1, 'LRG2': fitter_LRG2, 'ELG2': fitter_ELG2}
if data == 'pk': 
    fitters['LRG3'] = fitter_LRG3
else:
    fitters['LRG3ELG1'] = fitter_LRG3ELG1
    
print(fitters)
print(sys.getsizeof(fitters)) 
print('barry fitter objects loaded')  


# set up the KDEs (interpolation of the likelihood) for each tracer 

for item in fitters:
    
    weight, chain, model = fitters[item].load()[realisation_number][1], fitters[item].load()[realisation_number][2], fitters[item].load()[realisation_number][4]
    
    df = pd.DataFrame(chain, columns=model.get_labels())
    df['weights'] = weight

    if item in ['QSO', 'BGS']:
        df = df[copy_list_BGSQSO] 
    else:
        df = df[list_paramswanted] 

    dataframes_kdes[item] = [] 
    dataframes_kdes[item].append(df) 

    if item in ['QSO', 'BGS']:

        kde = gaussian_kde(np.vstack([
            df[r'$\alpha$'].to_numpy(),
            df[r'$\beta_{\phi(N_{\mathrm{eff}})}$'].to_numpy()]), 
            weights=df['weights'].to_numpy())

    else:                  

        kde = gaussian_kde(np.vstack([
            df[r'$\alpha$'].to_numpy(),
            df[r'$\epsilon$'].to_numpy(),
            df[r'$\beta_{\phi(N_{\mathrm{eff}})}$'].to_numpy()]), 
            weights=df['weights'].to_numpy())

    dataframes_kdes[item].append(kde) 
    
print(dataframes_kdes)
print(sys.getsizeof(dataframes_kdes)) 
    
print('kde objects set up') 
# define a function for the log of the combined likelihood of all the tracers    

def log_prob_betaphaseshift(x):
    
    # x is a vector with alpha, epsilon x 5 for each dataset - in a given order 
    # get the likelihood from each KDE 

    qso = dataframes_kdes['QSO'][1]([x[0], x[10]])[0]
    bgs = dataframes_kdes['BGS'][1]([x[1], x[10]])[0]
    lrg1 = dataframes_kdes['LRG1'][1]([x[2], x[3], x[10]])[0]
    lrg2 = dataframes_kdes['LRG2'][1]([x[4], x[5], x[10]])[0]
    if data == 'pk':
        lrg3 = dataframes_kdes['LRG3'][1]([x[6], x[7], x[10]])[0]
    else:
        lrg3 = dataframes_kdes['LRG3ELG1'][1]([x[6], x[7], x[10]])[0]
    elg2 = dataframes_kdes['ELG2'][1]([x[8], x[9], x[10]])[0]
    
    if qso <= 0.0 or abs(x[0])-1.0 >= 0.2 or abs(qso) == np.inf:
        qso = -np.inf 
    else:
        qso = np.log(qso)
    
    if bgs <= 0.0 or abs(x[1])-1.0 >= 0.2 or abs(bgs) == np.inf:
        bgs = -np.inf 
    else:
        bgs = np.log(bgs)
        
    if lrg1 <= 0.0 or abs(x[2])-1.0 >= 0.2 or abs(x[3]) >= 0.2 or abs(lrg1) == np.inf: 
        lrg1 = -np.inf 
    else:
        lrg1 = np.log(lrg1)
    
    if lrg2 <= 0.0 or abs(x[4])-1.0 >= 0.2 or abs(x[5]) >= 0.2 or abs(lrg2) == np.inf: 
        lrg2 = -np.inf 
    else:
        lrg2 = np.log(lrg2)
        
    if lrg3 <= 0.0 or abs(x[6])-1.0 >= 0.2 or abs(x[7]) >= 0.2 or abs(lrg3) == np.inf: 
        lrg3 = -np.inf 
    else:
        lrg3 = np.log(lrg3)
    
    if elg2 <= 0.0 or abs(x[8])-1.0 >= 0.2 or abs(x[9]) >= 0.2 or abs(elg2) == np.inf: 
        elg2 = -np.inf 
    else:
        elg2 = np.log(elg2)
        
        
    if abs(lrg1) == np.inf or abs(lrg2) == np.inf or abs(lrg3) == np.inf or abs(elg2) == np.inf or abs(qso) == np.inf or abs(bgs) == np.inf:
        logl = -np.inf 
    elif x[10] > 10 or x[10] < -8.0:
        logl = -np.inf
    else: 
        logl = elg2 + lrg1 + lrg2 + lrg3 + qso + bgs 
        
    return logl 

print('running MCMC') 

# now run an MCMC fit to the combined likelihood of the KDES (using previously defined function) 
dim = 11
np.random.seed(42)
nwalkers = 32                                                                                          
p0 = np.array([np.random.uniform(0.99, 1.01, nwalkers),  
               np.random.uniform(0.99, 1.01, nwalkers),  
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

chains_flat = sampler.get_chain(flat=True, discard=5000)

df_fit = pd.DataFrame({labels[i]: chains_flat[:,i] for i in np.arange(len(chains_flat[0,:]))})

print('MCMC successful, saving chain to a file') 

# save to a file 
df_fit.to_csv("/pscratch/sd/a/abbew25/combinedfits_secondgen_mocks_v1_2_reducedcov/"+data+"_"+recon+"_"+broadband_method +"_" + str(realisation_number) + ".csv")
