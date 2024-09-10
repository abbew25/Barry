import sys
import os

# this is Cullan's code to run the second gen mocks with all appropriate settings 

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
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import pickle
from chainconsumer import ChainConsumer


# Config file to fit the abacus cutsky mock means for sigmas
if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=True)
    sampler = NautilusSampler(temp_dir=dir_name, nlive=500)

    colors = [mplc.cnames[color] for color in ["orange", "orangered", "firebrick", "lightskyblue", "steelblue", "seagreen", "black"]]

    tracers = {
        "LRG": [[0.4, 0.6], [0.6, 0.8]],# [0.8, 1.1]],
        "ELG_LOPnotqso": [[1.1, 1.6]], # [0.8, 1.1],
        "QSO": [[0.8, 2.1]],
        "BGS_BRIGHT-21.5": [[0.1, 0.4]],
        "LRG+ELG_LOPnotqso": [[0.8, 1.1]],
    }
    reconsmooth = {"LRG": 10, "ELG_LOPnotqso": 10, "QSO": 30, "BGS_BRIGHT-21.5": 15, "LRG+ELG_LOPnotqso": 15}
    sigma_nl_par = {
        "LRG": [
            [9.0, 6.0],
            [9.0, 6.0],
            [9.0, 6.0],
        ],
        "ELG_LOPnotqso": [[8.5, 6.0], [8.5, 6.0]],
        "QSO": [[9.0, 6.0]],
        "BGS_BRIGHT-21.5": [[10.0, 8.0]],
        "LRG+ELG_LOPnotqso": [[9.0, 6.0]],
        
    }
    sigma_nl_perp = {
        "LRG": [
            [4.5, 3.0],
            [4.5, 3.0],
            [4.5, 3.0],
        ],
        "ELG_LOPnotqso": [[4.5, 3.0], [4.5, 3.0]],
        "QSO": [[3.5, 3.0]],
        "BGS_BRIGHT-21.5": [[6.5, 3.0]],
        "LRG+ELG_LOPnotqso": [[4.5, 3.0]],
        
    }
    sigma_s = {
        "LRG": [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
        "ELG_LOPnotqso": [[2.0, 2.0], [2.0, 2.0]],
        "QSO": [[2.0, 2.0]],
        "BGS_BRIGHT-21.5": [[2.0, 2.0]],
        "LRG+ELG_LOPnotqso": [[2.0, 2.0]],
        
    }

    cap = "gccomb"
    ffa = "ffa"  # Flavour of fibre assignment. Can be "ffa" for fast fiber assign, or "complete"
    rpcut = False  # Whether or not to include the rpcut
    imaging = (
        "default_FKP"
        # What form of imaging systematics to use. Can be "default_FKP", "default_FKP_addSN", or "default_FKP_addRF"
    )
    rp = f"{imaging}_rpcut2.5" if rpcut else f"{imaging}"

    # plotnames = [f"{t}_{zs[0]}_{zs[1]}" for t in tracers for i, zs in enumerate(tracers[t])]
    # datanames = [f"{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}" for t in tracers for i, zs in enumerate(tracers[t])]

    allnames = []
    count = 0
    for t in tracers:
        for i, zs in enumerate(tracers[t]):
            for r, recon in enumerate([None, "sym"]):
                for broadband in ['spline', 'poly']:
                    for vary_beta in [False]: 
#                     # ------------------------------------------------------------------------------------------------
#                     # ------------------------------------------------------------------------------------------------

                        n_poly = [-2, -1, 0] 
                        if broadband == 'spline':
                            n_poly = [0, 2]

                        model = CorrBeutler2017(
                            recon=recon,
                            isotropic=False,
                            marg="full",
                            fix_params=["om"],
                            poly_poles=[0, 2],
                            correction=Correction.NONE,
                            broadband_type=broadband,
                            n_poly=n_poly,
                        )
                        model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.25, max=16.0)
                        model.set_default("beta", 0.4, min=0.01, max=2.0)
                        model.set_default("sigma_nl_par", sigma_nl_par[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
                        model.set_default("sigma_nl_perp", sigma_nl_perp[t][i][r], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
                        model.set_default("sigma_s", sigma_s[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

                        # Load in a pre-existing BAO template
                        pktemplate = np.loadtxt("DESI_Pk_template.dat")
                        model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

                        name = ''

                        name = f"DESI_SecondGen_pickledbyAW_sm{reconsmooth[t]}_{t.lower()}_{cap}_z{zs[0]}-{zs[1]}_default_FKP_xi.pkl"
                        
                        if t == "LRG+ELG_LOPnotqso":
                            name = f"DESI_SecondGen_pickledbyAW_sm15_lrg+elg_lopnotqso_gccomb_z0.8-1.1_default_fkp_lin_nran10_njack0_split20_default_FKP_xi.pkl"

                        dataset = CorrelationFunction_DESI_KP4(
                            recon=model.recon,
                            fit_poles=model.poly_poles,
                            min_dist=50.0,
                            max_dist=150.0,
                            realisation=None,
                            reduce_cov_factor=25,
                            datafile=name,
                            data_location="/global/cfs/cdirs/desi/users/chowlett/barry_inputs/",
                        )

                        name = dataset.name + f" mock mean"
                        fitter.add_model_and_dataset(model, dataset, name=name, color=colors[count])
                        allnames.append(name)

#                     for j in range(len(dataset.mock_data)):
#                         dataset.set_realisation(j)
#                         name = dataset.name + f" realisation {j}"
#                         fitter.add_model_and_dataset(model, dataset, name=name, color=colors[count])
#                         allnames.append(name)


                    # ------------------------------------------------------------------------------------------------
                    # ------------------------------------------------------------------------------------------------

#                         n_poly = [-1, 0, 1, 2, 3]
#                         if broadband == 'spline':
#                             n_poly = 30

#                         model = PowerBeutler2017(
#                             recon=recon,
#                             isotropic=False,
#                             marg="full",
#                             fix_params=["om"],
#                             poly_poles=[0, 2],
#                             correction=Correction.NONE,
#                             broadband_type=broadband,
#                             n_poly=n_poly,
#                             vary_phase_shift_neff=vary_beta,
#                         )

#                         model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.25, max=16.0)
#                         model.set_default("beta", 0.4, min=0.01, max=2.0)
#                         model.set_default("sigma_nl_par", sigma_nl_par[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
#                         model.set_default("sigma_nl_perp", sigma_nl_perp[t][i][r], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
#                         model.set_default("sigma_s", sigma_s[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
#                         model.set_default('beta_phase_shift', default=1.0, min=-8, max=10)

#                         # Load in a pre-existing BAO template
#                         pktemplate = np.loadtxt("DESI_Pk_template.dat")
#                         model.kvals, model.pksmooth, model.pkratio = pktemplate.T

#                         name = f"DESI_SecondGen_pickledbyAW_sm{reconsmooth[t]}_{t.lower()}_{cap}_z{zs[0]}-{zs[1]}_default_FKP_pk.pkl"

#                         dataset = PowerSpectrum_DESI_KP4(
#                             recon=recon,
#                             fit_poles=[0, 2],
#                             min_k=0.02,
#                             max_k=0.30,
#                             realisation=None,
#                             reduce_cov_factor=25,
#                             datafile=name,
#                             data_location="/global/cfs/cdirs/desi/users/chowlett/barry_inputs/",
#                         )
                        
#                         name = dataset.name + f" mock mean"
#                         fitter.add_model_and_dataset(model, dataset, name=name, color=colors[count])
#                         allnames.append(name)

                        # for j in range(len(dataset.mock_data)):
                        #     dataset.set_realisation(j)
                        #     name = dataset.name + f" realisation {j}"
                        #     fitter.add_model_and_dataset(model, dataset, name=name, color=colors[count])
                        #     allnames.append(name)

                
            count += 1

    # Submit all the job. We have quite a few (42), so we'll
    # only assign 1 walker (processor) to each. Note that this will only run if the
    # directory is empty (i.e., it won't overwrite existing chains)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    
    outfile = fitter.temp_dir+pfn.split("/")[-1]+".fitter.pkl"
    with open(outfile, 'wb') as pickle_file:
        pickle.dump(fitter, pickle_file)
        
    fitter.fit(file)
