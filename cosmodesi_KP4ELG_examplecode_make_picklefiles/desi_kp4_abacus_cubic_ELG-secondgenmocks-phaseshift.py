import os
import sys
import pickle
import numpy as np

sys.path.append("../../Barry/")     # Change this so that it points to where you have Barry installed

from barry.samplers import NautilusSampler#DynestySampler
from barry.config import setup
from barry.models import PowerBeutler2017, CorrBeutler2017 # Beutler et al 2017 methods for calculation power spectrum, correlation function 
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4 # classes with methods for fitting data? 
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
from barry.fitter import Fitter # manages model fitting 
from barry.models.model import Correction # class for applying corrections to the likelihood function 

if __name__ == "__main__":
    
    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)
    # Set up the Fitting class and Dynesty sampler with 500 live points. 
    # Set remove_output=False to make sure that we don't delete/overwrite existing chains in the same directory.
    fitter = Fitter(dir_name, remove_output=True)
    sampler = NautilusSampler(temp_dir=dir_name, nlive=500)
    
    # The optimal sigma values we found when fitting the mocks with fixed alpha/epsilon
    sigma_nl_par = {None: 8.5, "sym": 6.0}
    sigma_nl_perp = {None: 4.5, "sym": 3.0}
    sigma_s = {None: 2.0, "sym": 2.0}
    
    # Loop over the mocktypes
    allnames = []
    mocknames = ['DESI_SecondGen_sm10_elg_lop_ffa_gccomb_0.8_1.1_default_FKP_xi.pkl']
    for i, mockname in enumerate(mocknames):

        # Loop over pre- and post-recon measurements
        for recon in [None, "sym"]:

            dataset_xi = CorrelationFunction_DESI_KP4(
                recon=recon,
                fit_poles=[0, 2],
                min_dist=50.0,
                max_dist=150.0,
                realisation=None,
                num_mocks=25,
                reduce_cov_factor=25,
                datafile=mockname,
                data_location="/global/u1/a/abbew25/barryrepo/Barry/cosmodesi_KP4ELG_examplecode_make_picklefiles",
            )

            model = CorrBeutler2017(
                recon=dataset_xi.recon,
                isotropic=dataset_xi.isotropic,
                marg="full",
                poly_poles=dataset_xi.fit_poles,
                correction=Correction.NONE,
                vary_phase_shift_neff=True, 
            )

            # Set Gaussian priors for the BAO damping centred on the optimal values 
            # found from fitting with fixed alpha/epsilon and with width 2 Mpc/h
            model.set_default("sigma_nl_par", sigma_nl_par[recon], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma_nl_perp[recon], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            model.set_default("sigma_s", sigma_s[recon], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            
            #pktemplate = np.loadtxt("../prepare_data/DESI_Pk_template.dat")
            pktemplate = np.loadtxt("DESI_Pk_template.dat")
            model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

            name = dataset_xi.name + " mock mean"
            fitter.add_model_and_dataset(model, dataset_xi, name=name)
            allnames.append(name)

            # Now add the individual realisations to the list
            for j in range(len(dataset_xi.mock_data)):
                dataset_xi.set_realisation(j)
                name = dataset_xi.name + f" realisation {j}"
                fitter.add_model_and_dataset(model, dataset_xi, name=name)
                allnames.append(name)

    #print(allnames)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    
    outfile = fitter.temp_dir+pfn.split("/")[-1]+".fitter.pkl"
    with open(outfile, 'wb') as pickle_file:
        pickle.dump(fitter, pickle_file)
                
    # Set the sampler (dynesty) and assign 1 walker (processor) to each. If we assign more than one walker, for dynesty
    # this means running independent chains which will then get added together when they are loaded in.
    fitter.fit(file)
    
   