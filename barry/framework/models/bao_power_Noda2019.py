import logging
import numpy as np
from scipy.interpolate import splev, splrep
import sys
sys.path.append("../../..")
from barry.framework.models.bao_power import PowerSpectrumFit
from barry.framework.cosmology.PT_generator import PTGenerator

class PowerNoda2019(PowerSpectrumFit):

    def __init__(self, fit_omega_m=False, fit_growth=False, fit_gamma=False, gammaval=4.0, smooth_type="hinton2017", recon=False, recon_smoothing_scale=21.21, name="BAO Power Spectrum Ding 2018 Fit"):
        self.recon = recon
        self.recon_smoothing_scale = recon_smoothing_scale
        self.fit_growth = fit_growth
        self.fit_gamma = fit_gamma
        self.gammaval = gammaval
        super().__init__(fit_omega_m=fit_omega_m, smooth_type=smooth_type, name=name)

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)

        self.PT = PTGenerator(self.camb, smooth_type=self.smooth_type, recon_smoothing_scale=self.recon_smoothing_scale)
        if not self.fit_omega_m:
            self.pt_data = self.PT.get_data(om=self.omega_m)
            if not self.fit_growth:
                self.growth = self.omega_m ** 0.55
                self.damping = -np.outer((1.0 + (2.0 + self.growth) * self.growth * self.mu ** 2) * self.pt_data["sigma_dd_rs"] + (self.growth * self.mu ** 2 * (self.mu ** 2 - 1.0)) * self.pt_data["sigma_ss_rs"], self.camb.ks ** 2)
                if not self.fit_gamma:
                    if self.recon:
                        self.damping /= self.gammaval

    def declare_parameters(self):
        super().declare_parameters()
        if self.fit_growth:
            self.add_param("f", r"$f$", 0.01, 1.0)  # Growth rate of structure
        if self.fit_gamma:
            self.add_param("gamma", r"$\gamma_{rec}$", 1.0, 8.0)  # Describes the sharpening of the BAO post-reconstruction
        self.add_param("A", r"$A$", 0.01, 30.0)  # Fingers-of-god damping

    def compute_power_spectrum(self, k, p):
        """ Computes the power spectrum model at k/alpha using the Ding et. al., 2018 EFT0 model
        
        Parameters
        ----------
        k : np.ndarray
            Array of wavenumbers to compute
        p : dict
            dictionary of parameter names to their values
            
        Returns
        -------
        array
            pk_final - The power spectrum at the dilated k-values
        
        """

        from scipy import integrate

        # Get the basic power spectrum components
        ks = self.camb.ks
        pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(p)
        if self.fit_omega_m:
            pt_data = self.PT.get_data(om=p["om"])
        else:
            pt_data = self.pt_data

        # Compute the growth rate depending on what we have left as free parameters
        if self.fit_growth:
            growth = p["f"]
        else:
            if self.fit_omega_m:
                growth = p["om"]**0.55
            else:
                growth = self.growth

        # Compute the BAO damping
        if self.fit_growth or self.fit_omega_m:
            damping = -np.outer((1.0 + (2.0 + growth) * growth * self.mu ** 2) * pt_data["sigma_dd_rs"] + (
                        growth * self.mu ** 2 * (self.mu ** 2 - 1.0)) * pt_data["sigma_ss_rs"], ks ** 2)
            if self.fit_gamma:
                damping /= p["gamma"]
            else:
                if self.recon:
                    damping /= self.gammaval
        else:
            damping = self.damping
        damping = np.exp(damping)

        # Compute the propagator
        if self.recon:
            # Compute the smoothing kernel (assumes a Gaussian smoothing kernel)
            smoothing_kernel = np.exp(-ks ** 2 * self.recon_smoothing_scale ** 2 / 4.0)
            kaiser_prefac = 1.0 + np.outer(growth / p["b"] * self.mu ** 2, 1.0-smoothing_kernel)
        else:
            kaiser_prefac = 1.0 + np.tile(growth / p["b"] * self.mu ** 2, (len(ks), 1)).T
        propagator = kaiser_prefac**2*damping

        # Compute the smooth model
        fog = np.exp(-p["A"]*ks**2)
        pk_smooth = p["b"]**2*pk_smooth_lin*fog

        # Compute the non-linear SPT correction to the smooth power spectrum
        pk_spt = pt_data["I00"] + pt_data["J00"] + 2.0*np.outer(growth/p["b"]*self.mu**2, pt_data["I01"] + pt_data["J01"]) + np.outer((growth/p["b"]*self.mu**2)**2, pt_data["I11"] + pt_data["J11"])

        # Integrate over mu
        pk1d = integrate.simps(pk_smooth*(1.0 + pk_ratio*propagator + pk_spt), self.mu, axis=0)

        pk_final = splev(k / p["alpha"], splrep(ks, pk1d))

        return pk_final

    def get_model(self, data, p):
        # Get the generic pk model
        pk_generated = self.compute_power_spectrum(data["ks_input"], p)

        # Morph it into a model representative of our survey and its selection/window/binning effects
        pk_windowed, mask = self.adjust_model_window_effects(pk_generated)
        return pk_windowed[mask]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    model_pre = PowerNoda2019(fit_omega_m=True, recon=False)
    model_post = PowerNoda2019(fit_omega_m=True, recon=True, gammaval=4.0)

    from barry.framework.datasets.mock_power import MockPowerSpectrum
    dataset = MockPowerSpectrum(step_size=2)
    data = dataset.get_data()
    model_pre.set_data(data)
    model_post.set_data(data)
    p = {"om": 0.3, "alpha": 1.0, "A":20.0, "b": 1.6}

    import timeit
    n = 100

    def test():
        model_post.get_likelihood(p)

    print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if True:
        ks = data["ks"]
        pk = data["pk"]
        pk2 = model_pre.get_model(data, p)
        model_pre.smooth_type = "eh1998"
        pk3 = model_pre.get_model(data, p)
        import matplotlib.pyplot as plt
        plt.errorbar(ks, pk, yerr=np.sqrt(np.diag(data["cov"])), fmt="o", c='k', label="Data")
        plt.plot(ks, pk2, '.', c='r', label="hinton2017")
        plt.plot(ks, pk3, '+', c='b', label="eh1998")
        plt.xlabel("k")
        plt.ylabel("P(k)")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        model_pre.smooth_type = "hinton2017"
        pk_smooth_lin, _ = model_pre.compute_basic_power_spectrum(p)
        growth = p["om"]**0.55
        pt_data = model_pre.PT.get_data(om=p["om"])
        pk_spt = pt_data["I00"] + pt_data["J00"] + 2.0/3.0*growth/p["b"]*(pt_data["I01"] + pt_data["J01"]) + 1.0/5.0*(growth/p["b"])**2*(pt_data["I11"] + pt_data["J11"])
        pk_smooth_interp = splev(data["ks_input"], splrep(model_pre.camb.ks, pk_smooth_lin*(1.0+pk_spt)))
        pk_smooth_lin_windowed, mask = model_pre.adjust_model_window_effects(pk_smooth_interp)
        pk2 = model_pre.get_model(data, p)
        pk3 = model_post.get_model(data, p)
        import matplotlib.pyplot as plt
        plt.plot(ks, pk2/pk_smooth_lin_windowed[mask], '.', c='r', label="pre-recon")
        plt.plot(ks, pk3/pk_smooth_lin_windowed[mask], '+', c='b', label="post-recon")
        plt.xlabel("k")
        plt.ylabel(r"$P(k)/P_{sm}(k)$")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(0.4, 3.0)
        plt.legend()
        plt.show()