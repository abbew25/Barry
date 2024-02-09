import sys

sys.path.append("../..")
from barry.models import PowerBeutler2017
from barry.models.bao_correlation import CorrelationFunctionFit
import numpy as np


class CorrBeutler2017(CorrelationFunctionFit):
    """xi(s) model inspired from Beutler 2017 that treats alphas in the same way as P(k)."""

    def __init__(
        self,
        name="Corr Beutler 2017",
        fix_params=("om",),
        smooth_type=None,
        recon=None,
        smooth=False,
        correction=None,
        isotropic=False,
        poly_poles=(0, 2),
        marg=None,
        dilate_smooth=False,
        vary_neff=False,
        vary_phase_shift_neff=False,
        use_classorcamb='CAMB',
        fog_wiggles=False,
        include_binmat=True,
        broadband_type="spline",
        **kwargs,
    ):

        self.dilate_smooth = dilate_smooth
        self.fog_wiggles = fog_wiggles

        super().__init__(
            name=name,
            fix_params=fix_params,
            smooth_type=smooth_type,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            poly_poles=poly_poles,
            marg=marg,
            vary_neff=vary_neff,
            vary_phase_shift_neff=vary_phase_shift_neff,
            use_classorcamb=use_classorcamb,
            include_binmat=include_binmat,
            broadband_type=broadband_type,
            **kwargs,
        )
        self.parent = PowerBeutler2017(
            fix_params=fix_params,
            smooth_type=smooth_type,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            marg=marg,
            dilate_smooth=dilate_smooth,
            vary_neff=vary_neff,
            vary_phase_shift_neff=vary_phase_shift_neff,
            use_classorcamb=use_classorcamb,
            fog_wiggles=fog_wiggles,
            broadband_type=None,
        )
        
        fix_params = [param for param in fix_params]
        if not vary_neff:
            fix_params.append("Neff")
        if not vary_phase_shift_neff:
            fix_params.append("beta_phase_shift")
            
        fix_params = tuple(fix_params) 

        self.set_marg(fix_params, do_bias=False, marg_bias=0)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("sigma_s", r"$\Sigma_s$", 0.0, 20.0, 10.0)  # Fingers-of-god damping
        if self.isotropic:
            self.add_param("sigma_nl", r"$\Sigma_{nl}$", 0.0, 20.0, 10.0)  # BAO damping
        else:
            self.add_param("beta", r"$\beta$", 0.01, 4.0, None)  # RSD parameter f/b
            self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.0, 20.0, 8.0)  # BAO damping parallel to LOS
            self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.0, 20.0, 4.0)  # BAO damping perpendicular to LOS

#     def compute_correlation_function(self, dist, p, smooth=False, vary_neff=False):
#         """Computes the correlation function model using the Beutler et. al., 2017 power spectrum
#             and 3 bias parameters and polynomial terms per multipole

#         Parameters
#         ----------
#         dist : np.ndarray
#             Array of distances in the correlation function to compute
#         p : dict
#             dictionary of parameter name to float value pairs
#         smooth : bool, optional
#             Whether or not to generate a smooth model without the BAO feature

#         Returns
#         -------
#         sprime : np.ndarray
#             distances of the computed xi
#         xi : np.ndarray
#             the model monopole, quadrupole and hexadecapole interpolated to sprime.
#         poly: np.ndarray
#             the additive terms in the model, necessary for analytical marginalisation

#         """

#         ks, pks, _ = self.parent.compute_power_spectrum(self.parent.camb.ks, p, smooth=smooth, nopoly=True, vary_neff=vary_neff)
#         xi_comp = np.array([self.pk2xi_0.__call__(ks, pks[0], dist), np.zeros(len(dist)), np.zeros(len(dist))])

#         if not self.isotropic:
#             xi_comp[1] = self.pk2xi_2.__call__(ks, pks[2], dist)
#             xi_comp[2] = self.pk2xi_4.__call__(ks, pks[4], dist)

#         xi, poly = self.add_poly(dist, p, xi_comp)

#         return dist, xi, poly


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_correlation_function import (
        CorrelationFunction_DESI_KP4,
    )
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    dataset = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_dist=52.0,
        max_dist=150.0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
    )

    model = CorrBeutler2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=dataset.fit_poles,
        correction=Correction.HARTLAP,
        n_poly=[0, 2],
    )
    model.set_default("sigma_nl_par", 5.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model.set_default("sigma_nl_perp", 2.0, min=0.0, max=20.0, sigma=1.0, prior="gaussian")
    model.set_default("sigma_s", 2.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

    model.sanity_check(dataset)
