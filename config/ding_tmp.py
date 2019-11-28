import sys


sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.utils import weighted_avg_and_std, get_model_comparison_dataframe
from barry.models import PowerDing2018, PowerBeutler2017
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter
import numpy as np
import pandas as pd


# Check if B17 and D18 results change if we apply the BAO extractor technique.
# Spoiler: They do not.
if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    r = True
    data = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r)
    model = PowerDing2018(recon=r)
    model.set_data(data.get_data())
    from barry.cosmology.PT_generator import getCambGeneratorAndPT

    cosmo = model.cosmology
    c, pt = getCambGeneratorAndPT(
        redshift=cosmo["z"], h0=cosmo["h0"], ob=cosmo["ob"], ns=cosmo["ns"], smooth_type="hinton2017", recon_smoothing_scale=cosmo["reconsmoothscale"]
    )
    ptd = pt.get_data(0.3)
    keys = ["sigma_nl", "sigma_dd_nl", "sigma_sd_nl", "sigma_ss_nl"]
    for key in keys:
        newv = model.get_pregen(key, 0.3)
        oldv = ptd[key]
        print(key, newv, oldv)

    if False:
        fitter = Fitter(dir_name, remove_output=True)

        c = getCambGenerator()
        r_s = c.get_data()["r_s"]
        p = BAOExtractor(r_s)

        sampler = DynestySampler(temp_dir=dir_name, nlive=200)

        for r in [True]:  # , False]:
            t = "Recon" if r else "Prerecon"
            ls = "-" if r else "--"

            d = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r)

            ding = PowerDing2018(recon=r)
            fitter.add_model_and_dataset(ding, d, name=f"D18", linestyle=ls, color="p")

        fitter.set_sampler(sampler)
        fitter.set_num_walkers(1)
        fitter.set_num_concurrent(700)

        fitter.fit(file)

        if fitter.should_plot():
            from chainconsumer import ChainConsumer

            c = ChainConsumer()
            names2 = []
            for posterior, weight, chain, evidence, model, data, extra in fitter.load():
                name = extra["name"]
                c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
            c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4)
            extents = None
            c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982})
            c.plotter.plot(filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982})