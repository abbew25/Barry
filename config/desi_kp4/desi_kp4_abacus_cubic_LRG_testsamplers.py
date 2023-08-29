import sys

sys.path.append("..")
sys.path.append("../../")
from barry.samplers import NautilusSampler, ZeusSampler, EnsembleSampler, DynestySampler
from barry.config import setup
from barry.models import PowerBeutler2017, CorrBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
from barry.fitter import Fitter
import numpy as np
import pandas as pd
from barry.models.model import Correction
from barry.utils import weighted_avg_and_cov
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__, "/reduced_cov/")

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitters = [Fitter(dir_name, remove_output=False) for _ in range(4)]
    samplers = [
        NautilusSampler(temp_dir=dir_name),
        ZeusSampler(temp_dir=dir_name),
        EnsembleSampler(temp_dir=dir_name),
        DynestySampler(temp_dir=dir_name),
        DynestySampler(temp_dir=dir_name, dynamic=True),
    ]
    sampler_names = ["Nautilus", "Zeus", "Emcee", "Dynesty"]

    # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
    # First load up mock mean and add it to the fitting list.
    dataset_pk = PowerSpectrum_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_k=0.02,
        max_k=0.30,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
        datafile="desi_kp4_abacus_cubicbox_cv_pk_lrg.pkl",
    )

    dataset_xi = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_dist=52.0,
        max_dist=150.0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
        datafile="desi_kp4_abacus_cubicbox_cv_xi_lrg.pkl",
    )

    # We'll do this test using post-recon measurements only
    model_pk = PowerBeutler2017(
        recon=dataset_pk.recon,
        isotropic=dataset_pk.isotropic,
        fix_params=["om"],
        marg="full",
        poly_poles=dataset_pk.fit_poles,
        correction=Correction.NONE,
        n_poly=6,
    )
    model_pk.set_default("sigma_nl_par", 5.1, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_pk.set_default("sigma_nl_perp", 1.6, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_pk.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model_pk.kvals, model_pk.pksmooth, model_pk.pkratio = pktemplate.T

    model_xi = CorrBeutler2017(
        recon=dataset_xi.recon,
        isotropic=dataset_xi.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=dataset_xi.fit_poles,
        correction=Correction.NONE,
        n_poly=4,
    )
    model_xi.set_default("sigma_nl_par", 5.1, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_xi.set_default("sigma_nl_perp", 1.6, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_xi.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model_xi.parent.kvals, model_xi.parent.pksmooth, model_xi.parent.pkratio = pktemplate.T

    for fitter, sampler, sampler_name in zip(fitters, samplers, sampler_names):

        fitter.add_model_and_dataset(model_pk, dataset_pk, name=dataset_pk.name + f" mock mean {sampler_name}")
        fitter.add_model_and_dataset(model_xi, dataset_xi, name=dataset_xi.name + f" mock mean {sampler_name}")

        # Submit all the jobs to NERSC. We have quite a few (72), so we'll
        # only assign 1 walker (processor) to each. Note that this will only run if the
        # directory is empty (i.e., it won't overwrite existing chains)
        fitter.set_sampler(sampler)
        fitter.set_num_walkers(1)
        fitter.fit(file)

    # Everything below here is for plotting the chains once they have been run. The should_plot()
    # function will check for the presence of chains and plot if it finds them on your laptop. On the HPC you can
    # also force this by passing in "plot" as the second argument when calling this code from the command line.
    if fitters[-1].should_plot():
        import logging

        logging.info("Creating plots")

        # Set up a ChainConsumer instance. Plot the MAP for individual realisations and a contour for the mock average
        datanames = ["Xi_CV", "Pk_CV"]

        c = [
            ChainConsumer(),
            ChainConsumer(),
        ]

        # Loop over all the fitters
        stats = [[] for _ in range(len(datanames))]
        output = {k: [] for k in datanames}
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the realisation number and redshift bin
            recon_bin = 0 if "Prerecon" in extra["name"] else 1
            data_bin = 0 if "Xi" in extra["name"] else 1 if "CV" not in extra["name"] else 2
            sigma_bin = int(extra["name"].split("fixed_type ")[1].split(" ")[0])
            redshift_bin = int(2.0 * len(sigma_sigma) * data_bin + 2.0 * sigma_bin + recon_bin)

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels())

            # Compute alpha_par and alpha_perp for each point in the chain
            alpha_par, alpha_perp = model.get_alphas(df["$\\alpha$"].to_numpy(), df["$\\epsilon$"].to_numpy())
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp
            mean, cov = weighted_avg_and_cov(
                df[["$\\alpha_\\parallel$", "$\\alpha_\\perp$", "$\\Sigma_{nl,||}$", "$\\Sigma_{nl,\\perp}$", "$\\Sigma_s$"]],
                weight,
                axis=0,
            )
            extra.pop("realisation", None)
            if "n_poly=5" in extra["name"]:
                extra["name"] = datanames[data_bin] + f" fixed_type {sigma_bin}"
                c[data_bin].add_chain(df, weights=weight, **extra, plot_contour=True, plot_point=False, show_as_1d_prior=False)

            stats[data_bin].append(
                [
                    sigma_sigma[sigma_bin],
                    model.n_poly,
                    mean[0] - 1.0,
                    mean[1] - 1.0,
                    mean[2] - 5.4,
                    mean[3] - 1.8,
                    mean[4],
                    np.sqrt(cov[0, 0]),
                    np.sqrt(cov[1, 1]),
                    np.sqrt(cov[2, 2]),
                    np.sqrt(cov[3, 3]),
                    np.sqrt(cov[4, 4]),
                ]
            )
            output[datanames[data_bin]].append(
                f"{sigma_sigma[sigma_bin]:6.4f}, {model.n_poly:3d}, {mean[0]:6.4f}, {mean[1]:6.4f}, {mean[2]:6.4f}, {mean[3]:6.4f}, {mean[4]:6.4f}, {np.sqrt(cov[0, 0]):6.4f}, {np.sqrt(cov[1, 1]):6.4f}, {np.sqrt(cov[2, 2]):6.4f}, {np.sqrt(cov[3, 3]):6.4f}, {np.sqrt(cov[4, 4]):6.4f}"
            )

        print(stats)

        for data_bin in range(3):
            if "Pre" in datanames[data_bin]:
                truth = {
                    "$\\alpha_\\perp$": 1.0,
                    "$\\alpha_\\parallel$": 1.0,
                    "$\\Sigma_{nl,||}$": 9.71,
                    "$\\Sigma_{nl,\\perp}$": 4.66,
                    "$\\Sigma_s$": None,
                }
            else:
                truth = {
                    "$\\alpha_\\perp$": 1.0,
                    "$\\alpha_\\parallel$": 1.0,
                    "$\\Sigma_{nl,||}$": 5.29,
                    "$\\Sigma_{nl,\\perp}$": 1.57,
                    "$\\Sigma_s$": None,
                }

            c[data_bin].configure(bins=20, sigmas=[0, 1])
            c[data_bin].plotter.plot(
                filename=["/".join(pfn.split("/")[:-1]) + "/" + datanames[data_bin] + "_contour.png"],
                truth=truth,
                parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$", "$\\Sigma_{nl,||}$", "$\\Sigma_{nl,\\perp}$", "$\\Sigma_s$"],
                legend=True,
                extents=[(0.98, 1.02), (0.98, 1.02)],
            )

            # Plot histograms of the errors and r_off
            plot_alphas(np.array(stats[data_bin]), "/".join(pfn.split("/")[:-1]) + "/" + datanames[data_bin] + "_alphas.png")
            plot_errors(np.array(stats[data_bin]), "/".join(pfn.split("/")[:-1]) + "/errs_" + datanames[data_bin] + "_alphas.png")

            # Save all the numbers to a file
            with open(dir_name + "/Barry_fit_" + datanames[data_bin] + ".txt", "w") as f:
                f.write(
                    "# N_poly, alpha_par, alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof\n"
                )
                for l in output[datanames[data_bin]]:
                    f.write(l + "\n")
