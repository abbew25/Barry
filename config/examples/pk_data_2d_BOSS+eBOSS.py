import sys

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import DynestySampler, Optimiser
from barry.cosmology.camb_generator import getCambGenerator
from barry.config import setup
from barry.utils import weighted_avg_and_cov
from barry.models import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12, PowerSpectrum_eBOSS_LRGpCMASS
from barry.fitter import Fitter
import numpy as np
import pandas as pd
from barry.models.model import Correction
import matplotlib as plt
from matplotlib import cm


# Run an optimisation on each of the post-recon SDSS DR12 mocks. Then compare to the pre-recon mocks
# to compute the cross-correlation between BAO parameters and pre-recon measurements

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, remove_output=False)

    c = getCambGenerator()

    sampler = DynestySampler(temp_dir=dir_name, nlive=500)

    cmap = plt.cm.get_cmap("viridis")

    datasets = [
        PowerSpectrum_SDSS_DR12(
            redshift_bin=1,
            realisation="data",
            galactic_cap="ngc",
            recon="iso",
            isotropic=False,
            fit_poles=[0, 2, 4],
            min_k=0.02,
            max_k=0.30,
            num_mocks=999,
        ),
        PowerSpectrum_SDSS_DR12(
            redshift_bin=1,
            realisation="data",
            galactic_cap="sgc",
            recon="iso",
            isotropic=False,
            fit_poles=[0, 2, 4],
            min_k=0.02,
            max_k=0.30,
            num_mocks=999,
        ),
        PowerSpectrum_SDSS_DR12(
            redshift_bin=2,
            realisation="data",
            galactic_cap="ngc",
            recon="iso",
            isotropic=False,
            fit_poles=[0, 2, 4],
            min_k=0.02,
            max_k=0.30,
            num_mocks=999,
        ),
        PowerSpectrum_SDSS_DR12(
            redshift_bin=2,
            realisation="data",
            galactic_cap="sgc",
            recon="iso",
            isotropic=False,
            fit_poles=[0, 2, 4],
            min_k=0.02,
            max_k=0.30,
            num_mocks=999,
        ),
        PowerSpectrum_eBOSS_LRGpCMASS(
            realisation="data", galactic_cap="ngc", recon="iso", isotropic=False, fit_poles=[0, 2, 4], min_k=0.02, max_k=0.30, num_mocks=999
        ),
        PowerSpectrum_eBOSS_LRGpCMASS(
            realisation="data", galactic_cap="sgc", recon="iso", isotropic=False, fit_poles=[0, 2, 4], min_k=0.02, max_k=0.30, num_mocks=999
        ),
    ]

    # Standard Beutler Model
    model = PowerBeutler2017(
        recon="iso", isotropic=False, fix_params=["om"], poly_poles=[0, 2, 4], correction=Correction.HARTLAP, marg="full"
    )

    for d in datasets:
        fitter.add_model_and_dataset(model, d, name=d.name + " " + str(i), realisation=i)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(5)
    fitter.set_num_concurrent(100)
    fitter.fit(file)

    # Everything below is nasty plotting code ###########################################################
    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        from chainconsumer import ChainConsumer

        counter = 0
        c = ChainConsumer()
        for i, (posterior, weight, chain, evidence, model, data, extra) in enumerate(fitter.load()):

            fitname = " ".join(extra["name"].split()[:4])

            color = plt.colors.rgb2hex(cmap(float(i / len(datasets))))

            model.set_data(data)
            r_s = model.camb.get_data()["r_s"]

            df = pd.DataFrame(chain, columns=model.get_labels())
            alpha = df["$\\alpha$"].to_numpy()
            epsilon = df["$\\epsilon$"].to_numpy()
            alpha_par, alpha_perp = model.get_alphas(alpha, epsilon)
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp

            extra.pop("realisation", None)
            c.add_chain(df, weights=weight, color=color, posterior=posterior, plot_contour=True, **extra)

            max_post = posterior.argmax()
            chi2 = -2 * posterior[max_post]

            params = model.get_param_dict(chain[max_post])
            for name, val in params.items():
                model.set_default(name, val)

            new_chi_squared, dof, bband, mods, smooths = model.plot(params, display=False)