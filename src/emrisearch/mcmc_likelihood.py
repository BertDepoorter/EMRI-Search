"""
Full matched-filter MCMC likelihood for EMRI parameter estimation.

This module provides:

:class:`FastEMRILikelihood`
    Matched-filter log-likelihood using ``fastemriwaveforms``
    (GPU-accelerated FEW with cupy-cuda12x) and ``fastlisaresponse``
    for LISA TDI channel generation.

:func:`prepare_mcmc_handoff`
    Convert a :class:`~emrisearch.iterative_track_search.SearchResult`
    (intrinsic-parameter MAP + Fisher bounds) into the full prior and
    initialisation needed by the MCMC sampler, including extrinsic
    parameters and the FEW ``(p0, e0)`` parameterisation.

Pipeline
--------
::

    # 1. Run iterative track search
    from emrisearch.iterative_track_search import IterativeTrackSearch
    result = IterativeTrackSearch(flux_data, data_sfts, t_obs).run(f_obs, fdot_obs)

    # 2. Prepare MCMC inputs
    from emrisearch.mcmc_likelihood import FastEMRILikelihood, prepare_mcmc_handoff
    handoff = prepare_mcmc_handoff(result)

    # 3. Build likelihood (requires fastemriwaveforms + fastlisaresponse + cupy)
    like = FastEMRILikelihood(data_td_A, data_td_E, T_data, dt)
    log_like = like(params)  # params = handoff.param_vector_from_physical(...)

Notes on parameterisation
--------------------------
``FastEMRILikelihood`` operates in the FEW parameter space:

    ``[log₁₀M, log₁₀μ, a, p₀, e₀, x₀, d_L, qS, φS, qK, φK, Φ_φ₀, Φ_θ₀, Φ_r₀]``

The intrinsic parameters ``(log₁₀M, log₁₀μ, a, p₀, e₀)`` are tightly
constrained by the identification stage; the extrinsic parameters
``(d_L, qS, φS, qK, φK, Φ_φ₀, Φ_θ₀, Φ_r₀)`` use broad uninformative priors.

The ``T_plunge`` ↔ ``(p₀, e₀)`` conversion is performed by
:func:`tplunge_to_p0e0` using the fewtrax backward integration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from few.utils.constants import YRSID_SI
except ImportError:
    YRSID_SI = 3.15581497632e7  # fallback

# ---------------------------------------------------------------------------
# Parameter space constants
# ---------------------------------------------------------------------------

# Default broad prior bounds for extrinsic parameters
# [d_L (Gpc), qS (rad), phiS (rad), qK (rad), phiK (rad),
#  Phi_phi0 (rad), Phi_theta0 (rad), Phi_r0 (rad)]
EXTRINSIC_BOUNDS_DEFAULT = np.array([
    [0.01,  30.0],          # d_L [Gpc]
    [0.0,   np.pi],         # qS  (ecliptic latitude, colatitude convention)
    [0.0,   2.0 * np.pi],   # phiS
    [0.0,   np.pi],         # qK
    [0.0,   2.0 * np.pi],   # phiK
    [0.0,   2.0 * np.pi],   # Phi_phi0
    [0.0,   2.0 * np.pi],   # Phi_theta0
    [0.0,   2.0 * np.pi],   # Phi_r0
])

EXTRINSIC_NAMES = [
    "d_L [Gpc]", "qS [rad]", "phiS [rad]",
    "qK [rad]", "phiK [rad]",
    "Phi_phi0 [rad]", "Phi_theta0 [rad]", "Phi_r0 [rad]",
]

# Intrinsic parameter names (FEW convention)
INTRINSIC_NAMES_FEW = ["log10_M", "log10_mu", "a", "p0", "e0"]


# ---------------------------------------------------------------------------
# Parameterisation helpers
# ---------------------------------------------------------------------------

def tplunge_to_p0e0(
    M: float,
    mu: float,
    a: float,
    T_plunge: float,
    e_f: float,
    flux_data=None,
    x0: float = 1.0,
    dense_steps: int = 500,
) -> tuple[float, float]:
    """Convert plunge-time parameterisation to FEW initial conditions.

    Runs the fewtrax backward integration from the separatrix for
    ``T_plunge`` years and reads off ``(p, e)`` at the far end — the
    orbital elements at the start of the LISA observation window.

    Parameters
    ----------
    M, mu : float   Primary and secondary masses [M☉].
    a : float       Dimensionless BH spin.
    T_plunge : float  Years until plunge at the reference epoch.
    e_f : float     Final eccentricity at the separatrix.
    flux_data : fewtrax.data.FluxData, optional
        If None, ``fewtrax.data.load_flux_data()`` is called automatically.
    x0 : float      Inclination sign (+1 prograde, -1 retrograde).
    dense_steps : int  Trajectory resolution.

    Returns
    -------
    p0 : float  Semi-latus rectum at the start of observation.
    e0 : float  Eccentricity at the start of observation.
    """
    try:
        from fewtrax.trajectory import EMRIInspiral
        from fewtrax.data import load_flux_data
    except ImportError as exc:
        raise ImportError("tplunge_to_p0e0 requires fewtrax.") from exc

    if flux_data is None:
        flux_data = load_flux_data()

    traj = EMRIInspiral(flux_data)
    t_b, p_b, e_b, _, _, _ = traj(
        p0=float(10.0),
        e0=float(e_f),
        T=float(T_plunge),
        a=float(a),
        M=float(M),
        mu=float(mu),
        backward=True,
        e_f=float(e_f),
        dense_steps=dense_steps,
    )
    return float(np.array(p_b)[-1]), float(np.array(e_b)[-1])


# ---------------------------------------------------------------------------
# MCMCHandoff
# ---------------------------------------------------------------------------

@dataclass
class MCMCHandoff:
    """Ready-to-use inputs for the MCMC follow-up stage.

    Produced by :func:`prepare_mcmc_handoff` from a
    :class:`~emrisearch.iterative_track_search.SearchResult`.

    Attributes
    ----------
    intrinsic_bounds_tplunge : np.ndarray, shape (5, 2)
        Tight bounds for ``[log₁₀M, log₁₀μ, a, T_plunge, e_f]``
        from the Fisher matrix (``n_sigma``-σ credible region).
    intrinsic_bounds_few : np.ndarray, shape (5, 2)
        Corresponding bounds in FEW convention ``[log₁₀M, log₁₀μ, a, p₀, e₀]``.
    intrinsic_map_tplunge : np.ndarray, shape (5,)
        MAP estimate in T_plunge parameterisation.
    intrinsic_map_few : np.ndarray, shape (5,)
        MAP estimate in FEW convention ``[log₁₀M, log₁₀μ, a, p₀, e₀]``.
    extrinsic_bounds : np.ndarray, shape (8, 2)
        Broad prior bounds for extrinsic parameters
        ``[d_L, qS, φS, qK, φK, Φ_φ₀, Φ_θ₀, Φ_r₀]``.
    confirmed_modes : list of (m, k, n)
    mode_det_stats : dict
    sigma_intrinsic : np.ndarray, shape (5,)
        1-σ intrinsic parameter uncertainties from Fisher matrix.
    """

    intrinsic_bounds_tplunge: np.ndarray
    intrinsic_bounds_few: np.ndarray
    intrinsic_map_tplunge: np.ndarray
    intrinsic_map_few: np.ndarray
    extrinsic_bounds: np.ndarray
    confirmed_modes: list
    mode_det_stats: dict
    sigma_intrinsic: np.ndarray

    @property
    def full_bounds_few(self) -> np.ndarray:
        """Combined (13, 2) bounds in FEW convention."""
        return np.vstack([self.intrinsic_bounds_few, self.extrinsic_bounds])

    @property
    def n_intrinsic(self) -> int:
        return len(self.intrinsic_map_few)

    @property
    def n_extrinsic(self) -> int:
        return len(self.extrinsic_bounds)

    @property
    def n_params(self) -> int:
        return self.n_intrinsic + self.n_extrinsic

    def summary(self) -> str:
        lines = ["MCMCHandoff (FEW convention)"]
        lines.append("=" * 56)
        lines.append("Intrinsic parameters (tight Fisher bounds):")
        for i, name in enumerate(INTRINSIC_NAMES_FEW):
            lo, hi = self.intrinsic_bounds_few[i]
            lines.append(
                f"  {name:<14s}: {self.intrinsic_map_few[i]:.5g}"
                f"  ±{self.sigma_intrinsic[i]:.3g}"
                f"  [{lo:.5g}, {hi:.5g}]"
            )
        lines.append("Extrinsic parameters (broad priors):")
        for i, name in enumerate(EXTRINSIC_NAMES):
            lo, hi = self.extrinsic_bounds[i]
            lines.append(f"  {name:<22s}: [{lo:.4g}, {hi:.4g}]")
        return "\n".join(lines)


def prepare_mcmc_handoff(
    result,  # SearchResult
    extrinsic_bounds: Optional[np.ndarray] = None,
    n_sigma: Optional[float] = None,
    flux_data=None,
) -> MCMCHandoff:
    """Build MCMC inputs from a :class:`~emrisearch.iterative_track_search.SearchResult`.

    Converts the intrinsic MAP estimate from ``(M, μ, a, T_plunge, e_f)``
    to FEW convention ``(log₁₀M, log₁₀μ, a, p₀, e₀)`` and constructs
    tight prior bounds for the intrinsic parameters and broad bounds for
    the extrinsic parameters.

    Parameters
    ----------
    result : SearchResult
        Output of :class:`~emrisearch.iterative_track_search.IterativeTrackSearch`.
    extrinsic_bounds : np.ndarray, shape (8, 2), optional
        Override the default extrinsic prior bounds.
    n_sigma : float, optional
        Override the σ-width used for the intrinsic bounds.  Defaults to
        the ``n_sigma_narrow`` used when ``result.narrow_bounds`` was
        computed.
    flux_data : fewtrax.data.FluxData, optional
        Pre-loaded fewtrax flux data; loaded automatically if None.

    Returns
    -------
    MCMCHandoff
    """
    M, mu, a, T_plunge, e_f = result.theta_map
    sigma = result.sigma_params

    if extrinsic_bounds is None:
        extrinsic_bounds = EXTRINSIC_BOUNDS_DEFAULT.copy()

    # --- Intrinsic bounds in T_plunge convention (already computed) --------
    if n_sigma is not None:
        # Recompute with requested n_sigma
        from .track_optimizer import DEFAULT_BOUNDS
        lo = result.theta_map - n_sigma * sigma
        hi = result.theta_map + n_sigma * sigma
        lo = np.maximum(lo, DEFAULT_BOUNDS[:, 0])
        hi = np.minimum(hi, DEFAULT_BOUNDS[:, 1])
        intrinsic_bounds_tplunge = np.column_stack([lo, hi])
    else:
        intrinsic_bounds_tplunge = result.narrow_bounds.copy()

    # --- MAP in FEW convention --------------------------------------------
    p0 = result.p0
    e0 = result.e0
    intrinsic_map_few = np.array([
        np.log10(M), np.log10(mu), a, p0, e0
    ])

    # --- Intrinsic bounds in FEW convention --------------------------------
    # Propagate T_plunge / e_f bounds to p0 / e0 by evaluating the
    # backward trajectory at the bound corners.
    # Simple approach: use ±sigma_T_plunge and ±sigma_e_f to get (p0,e0) range.
    sigma_T = float(sigma[3])
    sigma_e = float(sigma[4])
    n = intrinsic_bounds_tplunge[3][1] - intrinsic_bounds_tplunge[3][0]  # full range

    p0_samples, e0_samples = [], []
    for dT in [intrinsic_bounds_tplunge[3][0], T_plunge, intrinsic_bounds_tplunge[3][1]]:
        for de in [intrinsic_bounds_tplunge[4][0], e_f, intrinsic_bounds_tplunge[4][1]]:
            try:
                _p0, _e0 = tplunge_to_p0e0(M, mu, a, dT, de, flux_data=flux_data)
                p0_samples.append(_p0)
                e0_samples.append(_e0)
            except Exception:
                pass

    if p0_samples:
        p0_lo = max(0.0, min(p0_samples))
        p0_hi = max(p0_samples)
        e0_lo = max(0.0, min(e0_samples))
        e0_hi = min(0.95, max(e0_samples))
    else:
        p0_lo, p0_hi = p0 * 0.9, p0 * 1.1
        e0_lo, e0_hi = max(0.0, e0 - 0.1), min(0.9, e0 + 0.1)
        warnings.warn("Could not compute (p0, e0) bounds from backward trajectory.")

    # Full 5-parameter bounds in FEW convention
    lo_tpl = intrinsic_bounds_tplunge[:, 0]
    hi_tpl = intrinsic_bounds_tplunge[:, 1]
    intrinsic_bounds_few = np.array([
        [np.log10(lo_tpl[0]), np.log10(hi_tpl[0])],  # log10 M
        [np.log10(lo_tpl[1]), np.log10(hi_tpl[1])],  # log10 mu
        [lo_tpl[2], hi_tpl[2]],                       # a
        [p0_lo, p0_hi],                               # p0
        [e0_lo, e0_hi],                               # e0
    ])

    # Sigma in FEW convention (log10 transforms for M and mu)
    log10_e = np.log10(np.e)
    sigma_few = np.array([
        sigma[0] / (M * np.log(10)),   # sigma_log10M = sigma_M / (M * ln10)
        sigma[1] / (mu * np.log(10)),  # sigma_log10mu
        sigma[2],                       # sigma_a unchanged
        abs(p0_hi - p0_lo) / (2.0 * n if n > 0 else 1.0),  # rough sigma_p0
        abs(e0_hi - e0_lo) / (2.0 * abs(sigma[4]) if sigma[4] > 0 else 1.0),  # rough sigma_e0
    ])

    return MCMCHandoff(
        intrinsic_bounds_tplunge=intrinsic_bounds_tplunge,
        intrinsic_bounds_few=intrinsic_bounds_few,
        intrinsic_map_tplunge=result.theta_map.copy(),
        intrinsic_map_few=intrinsic_map_few,
        extrinsic_bounds=extrinsic_bounds,
        confirmed_modes=result.confirmed_modes,
        mode_det_stats=result.mode_det_stats,
        sigma_intrinsic=sigma,
    )


# ---------------------------------------------------------------------------
# FastEMRILikelihood
# ---------------------------------------------------------------------------

class FastEMRILikelihood:
    """Full matched-filter EMRI log-likelihood using fastemriwaveforms + fastlisaresponse.

    Requires:
    - ``fastemriwaveforms`` (FEW with cupy-cuda12x GPU backend)
    - ``fastlisaresponse``
    - ``cupy-cuda12x``

    Parameter vector convention
    ---------------------------
    ``params = [log₁₀M, log₁₀μ, a, p₀, e₀, x₀,
                d_L, qS, φS, qK, φK, Φ_φ₀, Φ_θ₀, Φ_r₀]``

    where the first 6 are intrinsic and the remaining 8 extrinsic.

    Parameters
    ----------
    data_A, data_E : np.ndarray, shape (n_samples,)
        TDI A and E channel time-domain data.  Matched to waveform duration T.
    T : float
        Observation duration [years].
    dt : float
        Sampling interval [seconds].
    psd_A, psd_E : callable or np.ndarray, optional
        PSD for each TDI channel evaluated at ``np.fft.rfftfreq(n, dt)``.
        If None, a default LISA sensitivity is used.
    use_gpu : bool
        Use GPU acceleration (requires cupy).  Default True.
    waveform_kwargs : dict, optional
        Additional keyword arguments forwarded to
        ``FastKerrEccentricEquatorialFlux``.
    response_kwargs : dict, optional
        Additional keyword arguments forwarded to ``ResponseWrapper``.
    """

    def __init__(
        self,
        data_A: np.ndarray,
        data_E: np.ndarray,
        T: float,
        dt: float,
        psd_A: Optional[np.ndarray] = None,
        psd_E: Optional[np.ndarray] = None,
        use_gpu: bool = True,
        waveform_kwargs: Optional[dict] = None,
        response_kwargs: Optional[dict] = None,
    ):
        self._T = float(T)
        self._dt = float(dt)
        self._use_gpu = use_gpu and self._check_gpu()

        self._data_A = np.asarray(data_A, dtype=np.float64)
        self._data_E = np.asarray(data_E, dtype=np.float64)
        n = len(self._data_A)
        self._freqs = np.fft.rfftfreq(n, dt)

        # Pre-compute FFT of data
        self._data_A_fd = np.fft.rfft(self._data_A)
        self._data_E_fd = np.fft.rfft(self._data_E)

        # PSD (shape: n_freq)
        if psd_A is None or psd_E is None:
            try:
                from emrisearch.da_utils import psd as lisa_psd
                self._psd_A = lisa_psd(self._freqs)
                self._psd_E = self._psd_A.copy()
            except ImportError:
                warnings.warn("Could not load default PSD; set psd_A and psd_E explicitly.")
                self._psd_A = np.ones_like(self._freqs)
                self._psd_E = np.ones_like(self._freqs)
        else:
            self._psd_A = np.asarray(psd_A, dtype=np.float64)
            self._psd_E = np.asarray(psd_E, dtype=np.float64)

        # Build waveform + response
        wf_kw = {"sum_kwargs": {"pad_output": True}, **(waveform_kwargs or {})}
        resp_kw = {"t0": 100.0 * dt, **(response_kwargs or {})}
        self._wf_gen, self._response = self._build_response(wf_kw, resp_kw)

    # ------------------------------------------------------------------

    @staticmethod
    def _check_gpu() -> bool:
        try:
            import cupy
            return True
        except ImportError:
            warnings.warn("cupy not available; falling back to CPU.")
            return False

    def _build_response(self, wf_kw: dict, resp_kw: dict):
        """Construct FEW + fastlisaresponse pipeline."""
        try:
            from few.waveform import FastKerrEccentricEquatorialFlux
            from fastlisaresponse import ResponseWrapper
        except ImportError as exc:
            raise ImportError(
                "FastEMRILikelihood requires fastemriwaveforms and fastlisaresponse.\n"
                "Install with: pip install fastemriwaveforms fastlisaresponse"
            ) from exc

        wf_gen = FastKerrEccentricEquatorialFlux(
            use_gpu=self._use_gpu, **wf_kw
        )
        T_s = self._T * YRSID_SI
        n = len(self._data_A)
        # ResponseWrapper wraps the waveform generator with the LISA orbit
        # and TDI combination.  sky angle indices: qS at index 7, phiS at 8
        # (0-indexed in the full [M, mu, a, p0, e0, x0, d_L, qS, phiS, qK, phiK,
        #   Phi_phi0, Phi_theta0, Phi_r0] vector → indices 7 and 8 are 0-based
        #   into the full param vector but fastlisaresponse expects them after
        #   removing log transforms, so index_lambda=7, index_beta=8).
        response = ResponseWrapper(
            wf_gen,
            T_s,
            self._dt,
            index_lambda=7,    # phiS index in [M,mu,a,p0,e0,x0,d_L,qS,phiS,...]
            index_beta=8,      # qS index
            t0=resp_kw.pop("t0", 100.0 * self._dt),
            flip_hx=True,
            use_gpu=self._use_gpu,
            remove_sky_coords=True,
            is_ecliptic_latitude=False,  # qS is colatitude
            remove_garbage=True,
            **resp_kw,
        )
        return wf_gen, response

    def _generate_waveform(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate TDI (A, E) waveform for given parameters.

        Parameters
        ----------
        params : shape (14,)
            ``[log₁₀M, log₁₀μ, a, p₀, e₀, x₀,
               d_L, qS, φS, qK, φK, Φ_φ₀, Φ_θ₀, Φ_r₀]``

        Returns
        -------
        A, E : complex arrays in frequency domain, shape (n_freq,)
        """
        log10_M, log10_mu, a, p0, e0, x0 = params[:6]
        d_L, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params[6:]

        M = 10.0 ** log10_M
        mu = 10.0 ** log10_mu

        try:
            A_td, E_td, _ = self._response(
                M, mu, a, p0, e0, x0,
                d_L, qS, phiS, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0,
                T=self._T, dt=self._dt,
            )
            if self._use_gpu:
                import cupy as cp
                A_td = cp.asnumpy(A_td)
                E_td = cp.asnumpy(E_td)
            n = len(self._data_A)
            A_fd = np.fft.rfft(np.real(A_td[:n]))
            E_fd = np.fft.rfft(np.real(E_td[:n]))
            return A_fd, E_fd
        except Exception as exc:
            warnings.warn(f"Waveform generation failed: {exc}")
            return None, None

    def _inner_product(
        self, h_fd: np.ndarray, d_fd: np.ndarray, psd: np.ndarray
    ) -> float:
        """Noise-weighted inner product (h | d) = 4 Re Σ h* d / S_n Δf."""
        df = self._freqs[1] - self._freqs[0] if len(self._freqs) > 1 else 1.0
        valid = (psd > 0) & (self._freqs > 0)
        ip = 4.0 * np.real(
            np.sum(h_fd[valid].conj() * d_fd[valid] / psd[valid])
        ) * df
        return float(ip)

    def log_likelihood(self, params: np.ndarray) -> float:
        """Evaluate the noise-weighted matched-filter log-likelihood.

        ``log L(θ) = (h | d) - ½ (h | h)``

        Parameters
        ----------
        params : np.ndarray, shape (14,)
            ``[log₁₀M, log₁₀μ, a, p₀, e₀, x₀,
               d_L, qS, φS, qK, φK, Φ_φ₀, Φ_θ₀, Φ_r₀]``

        Returns
        -------
        float
            Log-likelihood.  Returns ``-1e50`` on waveform failure.
        """
        A_fd, E_fd = self._generate_waveform(params)
        if A_fd is None:
            return -1.0e50

        hd_A = self._inner_product(A_fd, self._data_A_fd, self._psd_A)
        hd_E = self._inner_product(E_fd, self._data_E_fd, self._psd_E)
        hh_A = self._inner_product(A_fd, A_fd, self._psd_A)
        hh_E = self._inner_product(E_fd, E_fd, self._psd_E)

        return float(hd_A + hd_E - 0.5 * (hh_A + hh_E))

    def __call__(self, params: np.ndarray) -> float:
        """Alias for :meth:`log_likelihood`."""
        return self.log_likelihood(params)
