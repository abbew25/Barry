import sys

from chainconsumer import ChainConsumer

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import DynestySampler
from barry.config import setup
from barry.models import CorrBeutler2017
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESIMockChallenge_Post
from barry.fitter import Fitter
import numpy as np
import pandas as pd
from barry.models.model import Correction
from barry.utils import weighted_avg_and_cov, break_vector_and_get_blocks
import matplotlib as plt
from matplotlib import cm

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    print(pfn)
    fitter = Fitter(dir_name, remove_output=True)

    sampler = DynestySampler(temp_dir=dir_name, nlive=1000)

    names = [
        ["PostRecon Yuyu Iso Fix ", "PostRecon Yuyu Iso NonFix "],
        ["PostRecon Yuyu Ani Fix ", "PostRecon Yuyu Ani NonFix "],
    ]
    cmap = plt.cm.get_cmap("viridis")

    smoothtypes = [1, 2, 3, 4]  # [5, 10, 15, 20] Mpc/h
    smins = [35, 45, 55]

    allnames = []
    counter = 0
    for i, recon in enumerate(["iso", "ani"]):
        for j, covtype in enumerate(["fix", "nonfix"]):
            for smoothtype in smoothtypes:
                for fit_poles in [[0, 2], [0, 2, 4]]:
                    for smin in smins:
                        data = CorrelationFunction_DESIMockChallenge_Post(
                            isotropic=False,
                            recon=recon,
                            realisation="data",
                            fit_poles=fit_poles,
                            min_dist=smin,
                            max_dist=157.5,
                            num_mocks=998,
                            smoothtype=smoothtype,
                            covtype=covtype,
                            tracer="elg",
                        )
                        model = CorrBeutler2017(
                            recon=data.recon,
                            isotropic=data.isotropic,
                            marg="full",
                            fix_params=["om", "beta"],
                            poly_poles=fit_poles,
                            correction=Correction.NONE,
                        )

                        # model.set_data(data.get_data())
                        # p = model.get_defaults()
                        # model.plot(model.get_param_dict(p))

                        smoothnames = [" 5", " 10", " 15", " 20"]
                        hexname = " Hexa " if 4 in fit_poles else " No-Hexa "
                        name = names[i][j] + recon + smoothnames[smoothtype - 1] + hexname + str(r"$s_{min}=%3.1lf$" % smin)
                        allnames.append(name)
                        fitter.add_model_and_dataset(model, data, name=name)
                        counter += 1

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    # Everything below is nasty plotting code ###########################################################
    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        from chainconsumer import ChainConsumer

        output = {}
        for name in allnames:
            fitname = " ".join(name.split()[:7])
            output[fitname] = []

        c = ChainConsumer()
        counter = 0
        truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0, "$\\alpha_\\perp$": 1.0, "$\\alpha_\\parallel$": 1.0}
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            smin = extra["name"].split(" ")[-1][9:-1]
            fitname = " ".join(extra["name"].split()[:7])
            print(fitname, smin)

            color = plt.colors.rgb2hex(cmap(float(counter) / (len(smins) - 1)))

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

            figname = pfn + "_" + fitname + "_bestfit.pdf" if counter == 2 else None
            new_chi_squared, dof, bband, mods, smooths = model.plot(params, display=False, figname=figname)

            ps = chain[max_post, :]
            best_fit = {}
            for l, p in zip(model.get_labels(), ps):
                best_fit[l] = p

            mean, cov = weighted_avg_and_cov(
                df[
                    [
                        "$\\alpha_\\parallel$",
                        "$\\alpha_\\perp$",
                        "$\\Sigma_s$",
                        "$\\Sigma_{nl,||}$",
                        "$\\Sigma_{nl,\\perp}$",
                    ]
                ],
                weight,
                axis=0,
            )

            corr = cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])
            if "No-Hexa" in fitname:
                output[fitname].append(
                    f"{smin:3s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {new_chi_squared:7.3f}, {dof:4d}, {mean[3]:7.3f}, {mean[4]:7.3f}, {mean[2]:7.3f}, {bband[0]:7.3f}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:8.1f}, {bband[4]:8.1f}, {bband[5]:8.1f}, {bband[6]:8.1f},  {bband[7]:8.1f}"
                )
            else:
                output[fitname].append(
                    f"{smin:3s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {new_chi_squared:7.3f}, {dof:4d}, {mean[3]:7.3f}, {mean[4]:7.3f}, {mean[2]:7.3f}, {bband[0]:7.3f}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:8.1f}, {bband[4]:8.1f}, {bband[5]:8.1f}, {bband[6]:8.1f}, {bband[7]:8.1f}, {bband[8]:8.1f}, {bband[9]:8.1f}, {bband[10]:8.1f}, {bband[11]:8.1f}"
                )

            counter += 1
            if counter >= 3:
                counter = 0
                c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4, legend_location=(0, -1), plot_contour=True)
                c.plotter.plot(
                    filename=[pfn + "_" + fitname + "_contour.pdf"],
                    truth=truth,
                    parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                )
                c = ChainConsumer()

        for name in output.keys():

            with open(dir_name + "/Queensland_bestfit_" + name.replace(" ", "_") + ".txt", "w") as f:
                if "No-Hexa" in name:
                    f.write(
                        "# smin, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, b0, a0_1, a0_2, a0_3, b2, a2_1, a2_2, a2_3\n"
                    )
                else:
                    f.write(
                        "# smin, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, b0, a0_1, a0_2, a0_3, b2, a2_1, a2_2, a2_3, a4_1, a4_2, a4_3\n"
                    )
                for l in output[name]:
                    f.write(l + "\n")
