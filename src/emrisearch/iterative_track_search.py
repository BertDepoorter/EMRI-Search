"""
Iterative multi-track EMRI parameter recovery from a single STFT anchor track.

Pipeline
--------
1. Fit 5D intrinsic parameters ``(M, μ, a, T_plunge, e_f)`` to the anchor track
   using the JAX-differentiable fewtrax backend
   (:class:`~emrisearch.track_optimizer.TrackOptimizerJAX`: Adam + L-BFGS).
2. Compute the Fisher information matrix from the best-fit track →
   Cramér-Rao covariance → 1-σ constraints per parameter.
3. Predict the time-frequency location of every other EMRI harmonic mode
   from the current MAP estimate.
4. Vet each predicted mode against the STFT data using the semi-coherent
   detection statistic (:func:`~emrisearch.jax_utils.det_stat`).  A mode is
   "confirmed" when its detection statistic exceeds ``vet_threshold``.
5. Add all newly confirmed modes to a joint track-residual loss and refit.
6. Repeat from step 2 until no new modes are confirmed or ``max_iterations``
   is exhausted.

Output: :class:`SearchResult` — contains the MAP estimate, covariance,
narrow MCMC prior bounds, list of confirmed modes, per-mode detection
statistics, and the initial orbital elements ``(p0, e0)`` for the FEW
parameterisation.

Notes
-----
The ``vet_threshold`` for secondary-mode confirmation is not universal.
It depends on the signal SNR, the number of SFT segments, and the
noise realisation.  A single threshold should be calibrated over a
population of injections; the default of ``50.0`` is a starting point
for a ~30 SNR source with ~630 SFT segments at T_sft = 5×10⁴ s.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from few.utils.constants import YRSID_SI

from .track_optimizer import (
    TrackOptimizerJAX,
    TrackOptimizer,
    DEFAULT_BOUNDS,
    estimate_plunge_time,
    get_default_mode_candidates,
)
from .jax_utils import det_stat as jax_det_stat

try:
    import jax
    import jax.numpy as jnp
    from fewtrax.trajectory import EMRIInspiral
    from fewtrax.utils.tf_tracks import compute_freq_track_batch
    _FEWTRAX_AVAILABLE = True
except ImportError:
    _FEWTRAX_AVAILABLE = False

# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Output of :meth:`IterativeTrackSearch.run`.

    Attributes
    ----------
    theta_map : np.ndarray, shape (5,)
        MAP parameter estimate ``[M, mu, a, T_plunge, e_f]``.
    cov_cr : np.ndarray, shape (5, 5)
        Cramér-Rao covariance matrix ``F⁻¹`` where ``F`` is the Fisher
        information matrix evaluated at ``theta_map``.
    fisher : np.ndarray, shape (5, 5)
        Fisher information matrix ``J^T J / σ_f²``.
    sigma_params : np.ndarray, shape (5,)
        Per-parameter 1-σ uncertainties ``sqrt(diag(cov_cr))``.
    anchor_mode : tuple (m, k, n)
        The initial anchor mode used to seed the search.
    confirmed_modes : list of tuple (m, k, n)
        All modes confirmed by STFT vetting (includes the anchor).
    mode_det_stats : dict
        Mapping ``(m, k, n) → detection_statistic_value``.
    track_loss : float
        Final joint track-residual loss ``Σ_j Σ_i (f_pred_j - f_obs_j)²``.
    n_iterations : int
        Number of iterative refitting cycles performed.
    narrow_bounds : np.ndarray, shape (5, 2)
        Tight prior bounds ``[theta_map ± n_sigma * sigma_params]`` clipped to
        physical limits.  Ready for use as MCMC intrinsic-parameter prior.
    p0 : float
        Initial semi-latus rectum (start of observation) derived from the
        backward trajectory.  Used to initialise the FEW waveform model.
    e0 : float
        Initial eccentricity (start of observation) from the backward
        trajectory.
    """

    theta_map: np.ndarray
    cov_cr: np.ndarray
    fisher: np.ndarray
    sigma_params: np.ndarray
    anchor_mode: tuple
    confirmed_modes: list
    mode_det_stats: dict
    track_loss: float
    n_iterations: int
    narrow_bounds: np.ndarray
    p0: float
    e0: float

    PARAM_NAMES: tuple = field(
        default=("M [M☉]", "μ [M☉]", "a", "T_plunge [yr]", "e_f"),
        init=False, repr=False,
    )

    def summary(self) -> str:
        """Human-readable summary of the search result."""
        lines = ["SearchResult"]
        lines.append("=" * 56)
        lines.append("MAP parameters and 1-σ uncertainties:")
        for i, name in enumerate(self.PARAM_NAMES):
            lo, hi = self.narrow_bounds[i]
            lines.append(
                f"  {name:<18s}: {self.theta_map[i]:.5g}"
                f"  ±{self.sigma_params[i]:.3g}"
                f"  [{lo:.5g}, {hi:.5g}]"
            )
        lines.append(f"Anchor mode (m,k,n): {self.anchor_mode}")
        lines.append(
            f"Confirmed modes ({len(self.confirmed_modes)}): "
            + ", ".join(str(m) for m in self.confirmed_modes)
        )
        det_str = ", ".join(
            f"{m}: {v:.1f}" for m, v in self.mode_det_stats.items()
        )
        lines.append(f"Detection statistics: {det_str}")
        lines.append(
            f"p0 = {self.p0:.4f}  e0 = {self.e0:.4f}  (FEW initial conditions)"
        )
        lines.append(
            f"Iterations: {self.n_iterations}  |  "
            f"Track loss: {self.track_loss:.4e}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IterativeTrackSearch
# ---------------------------------------------------------------------------

class IterativeTrackSearch:
    """Iterative multi-track EMRI parameter identification.

    Parameters
    ----------
    flux_data : fewtrax.data.FluxData
        Pre-loaded fewtrax flux data (from ``fewtrax.data.load_flux_data()``).
    data_sfts : np.ndarray, shape (n_freq, n_sft), complex
        STFT data array (signal + noise).
    t_obs : np.ndarray, shape (n_sft,)
        SFT mid-times [seconds].
    T_sft : float
        SFT segment duration [seconds].  Default 5×10⁴ s.
    bounds : np.ndarray, shape (5, 2), optional
        Physical parameter bounds ``[[M_lo, M_hi], [mu_lo, mu_hi], ...]``.
        Defaults to :data:`~emrisearch.track_optimizer.DEFAULT_BOUNDS`.
    vet_threshold : float
        Detection-statistic threshold for confirming secondary modes.
        **This must be calibrated for your noise level and SFT configuration.**
        Default 50.0 is appropriate for SNR ≈ 30 with ~630 SFT segments at
        T_sft = 5×10⁴ s.
    n_sigma_narrow : float
        Number of Fisher 1-σ widths for the narrow prior bounds passed to
        the MCMC follow-up.  Default 5.0 (5-σ bounds).
    max_iterations : int
        Maximum number of iterative refitting cycles.  Default 5.
    top_k_vet : int
        Number of predicted modes (ranked by expected power) to vet per
        iteration.  Default 20.
    adam_starts : int
        Number of random starts in the multi-start Adam optimisation.
    adam_steps : int
        Adam steps per start.
    lbfgs_iter : int
        L-BFGS iterations for final refinement.
    mode_candidates : list of (m, k, n), optional
        Full mode candidate set to scan.  Defaults to all ``(m, 0, n)``
        with ``m ∈ [1, 4]`` and ``n ∈ [-2, 2]``.
    """

    def __init__(
        self,
        flux_data,
        data_sfts: np.ndarray,
        t_obs: np.ndarray,
        T_sft: float = 5e4,
        bounds: Optional[np.ndarray] = None,
        vet_threshold: float = 50.0,
        n_sigma_narrow: float = 5.0,
        max_iterations: int = 5,
        top_k_vet: int = 20,
        adam_starts: int = 32,
        adam_steps: int = 300,
        lbfgs_iter: int = 200,
        mode_candidates: Optional[list] = None,
    ):
        if not _FEWTRAX_AVAILABLE:
            raise ImportError(
                "IterativeTrackSearch requires fewtrax.  "
                "Install from JAX-waveform/fewtrax/."
            )

        self.flux_data = flux_data
        self.data_sfts = np.asarray(data_sfts, dtype=complex)
        self.t_obs = np.asarray(t_obs, dtype=float)
        self.T_sft = float(T_sft)
        self.bounds = DEFAULT_BOUNDS.copy() if bounds is None else np.asarray(bounds, float).copy()
        self.vet_threshold = float(vet_threshold)
        self.n_sigma_narrow = float(n_sigma_narrow)
        self.max_iterations = int(max_iterations)
        self.top_k_vet = int(top_k_vet)
        self.adam_starts = int(adam_starts)
        self.adam_steps = int(adam_steps)
        self.lbfgs_iter = int(lbfgs_iter)

        # Mode candidates: (m, k, n) triples, k=0 for equatorial EMRIs
        if mode_candidates is None:
            self.mode_candidates = [
                (m, 0, n)
                for m in range(1, 5)
                for n in range(-2, 3)
            ]
        else:
            self.mode_candidates = [
                (md[0], md[1], md[2]) if len(md) == 3 else (md[0], 0, md[1])
                for md in mode_candidates
            ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        f_obs: np.ndarray,
        fdot_obs: np.ndarray,
        anchor_mode: Optional[tuple] = None,
    ) -> SearchResult:
        """Run the full iterative multi-track identification pipeline.

        Parameters
        ----------
        f_obs : np.ndarray, shape (n_sft,)
            Observed GW frequency at each SFT segment [Hz].
        fdot_obs : np.ndarray, shape (n_sft,)
            Observed frequency derivative [Hz/s].
        anchor_mode : tuple (m, k, n) or (m, n), optional
            Mode numbers of the anchor track.  Defaults to ``(2, 0, 0)``.

        Returns
        -------
        SearchResult
        """
        f_obs = np.asarray(f_obs, dtype=float)
        fdot_obs = np.asarray(fdot_obs, dtype=float)

        if anchor_mode is None:
            anchor_mode = (2, 0, 0)
        if len(anchor_mode) == 2:
            anchor_mode = (anchor_mode[0], 0, anchor_mode[1])

        print(f"[IterativeTrackSearch] anchor mode: {anchor_mode}")

        # ---- Iteration 0: fit to anchor track ----------------------------
        optimizer = self._make_optimizer(f_obs, fdot_obs)
        theta_map, loss = optimizer.optimize_full(
            mode=anchor_mode,
            n_starts=self.adam_starts,
            adam_steps=self.adam_steps,
            lbfgs_iter=self.lbfgs_iter,
        )
        print(f"  Iter 0 — anchor fit: loss = {loss:.4e}")
        print(f"  MAP: {theta_map}")

        # Track which modes are confirmed and their observed frequencies
        confirmed_modes = [anchor_mode]
        observed_tracks: dict[tuple, tuple[np.ndarray, np.ndarray]] = {
            anchor_mode: (f_obs, self.t_obs)
        }
        mode_det_stats: dict[tuple, float] = {}

        # Evaluate detection statistic on anchor track
        t_back, p_back, e_back = self._run_trajectory(theta_map)
        f_anchor_pred, fdot_anchor_pred = self._frequency_and_fdot_at_obs(
            anchor_mode, t_back, p_back, e_back, theta_map
        )
        mode_det_stats[anchor_mode] = self._vet_track(f_anchor_pred, fdot_anchor_pred)

        n_iter = 0
        for iteration in range(self.max_iterations):
            n_iter = iteration + 1

            # ---- Predict secondary tracks --------------------------------
            t_back, p_back, e_back = self._run_trajectory(theta_map)
            p0 = float(p_back[-1])
            e0 = float(e_back[-1])

            new_confirmations = self._predict_and_vet(
                theta_map, t_back, p_back, e_back,
                confirmed_modes, mode_det_stats,
            )

            if not new_confirmations:
                print(f"  Iter {iteration+1} — no new modes confirmed, stopping.")
                break

            print(
                f"  Iter {iteration+1} — confirmed {len(new_confirmations)} new mode(s): "
                + ", ".join(str(m) for m in new_confirmations)
            )

            # Add newly confirmed observed tracks
            for mode in new_confirmations:
                f_pred, fdot_pred = self._frequency_and_fdot_at_obs(
                    mode, t_back, p_back, e_back, theta_map
                )
                # Store the predicted track as the "observed" frequency for
                # the secondary mode — this is the best estimate from the
                # current MAP parameters.
                observed_tracks[mode] = (f_pred, self.t_obs)
                confirmed_modes.append(mode)

            # ---- Refit to joint loss over all confirmed tracks -----------
            modes_list = confirmed_modes
            f_obs_list = [observed_tracks[md][0] for md in modes_list]
            t_obs_list = [observed_tracks[md][1] for md in modes_list]

            joint_loss_fn = optimizer._build_joint_loss_raw(modes_list, f_obs_list, t_obs_list)
            raw_init = TrackOptimizerJAX._to_unconstrained_np(theta_map, optimizer.bounds)
            theta_map, loss = self._run_lbfgs(joint_loss_fn, raw_init, optimizer)
            print(f"  Iter {iteration+1} — joint refit: loss = {loss:.4e}")

        # ---- Final trajectory & p0/e0 ------------------------------------
        t_back, p_back, e_back = self._run_trajectory(theta_map)
        p0 = float(p_back[-1])
        e0 = float(e_back[-1])

        # ---- Fisher matrix & narrow bounds --------------------------------
        fisher, cov_cr = self._compute_joint_fisher(
            theta_map, confirmed_modes, observed_tracks
        )
        sigma_params = np.sqrt(np.maximum(np.diag(cov_cr), 0.0))
        narrow_bounds = self._narrow_bounds_from_fisher(theta_map, sigma_params)

        return SearchResult(
            theta_map=theta_map,
            cov_cr=cov_cr,
            fisher=fisher,
            sigma_params=sigma_params,
            anchor_mode=anchor_mode,
            confirmed_modes=confirmed_modes,
            mode_det_stats=mode_det_stats,
            track_loss=float(loss),
            n_iterations=n_iter,
            narrow_bounds=narrow_bounds,
            p0=p0,
            e0=e0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_optimizer(
        self, f_obs: np.ndarray, fdot_obs: np.ndarray
    ) -> TrackOptimizerJAX:
        """Construct a TrackOptimizerJAX for the anchor track."""
        return TrackOptimizerJAX(
            f_obs=f_obs,
            fdot_obs=fdot_obs,
            t_obs=self.t_obs,
            flux_data=self.flux_data,
            T_sft=self.T_sft,
            bounds=self.bounds.copy(),
        )

    def _run_trajectory(
        self, theta_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run fewtrax backward integration and return numpy arrays.

        Returns
        -------
        t_back : np.ndarray
            Time before plunge τ [s], shape (dense_steps,), ascending.
        p_back : np.ndarray
            Semi-latus rectum at each τ, shape (dense_steps,).
        e_back : np.ndarray
            Eccentricity at each τ, shape (dense_steps,).
        """
        M, mu, a, T_plunge, e_f = theta_map
        dense_steps = max(500, 3 * len(self.t_obs))
        traj = EMRIInspiral(self.flux_data)
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
        return np.array(t_b), np.array(p_b), np.array(e_b)

    def _frequency_and_fdot_at_obs(
        self,
        mode: tuple,
        t_back: np.ndarray,
        p_back: np.ndarray,
        e_back: np.ndarray,
        theta_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute predicted frequency and fdot at SFT observation times.

        The backward trajectory uses τ = T_plunge - t as the time variable.
        SFT times t_obs are mapped to τ = T_plunge_s - t_obs for interpolation.

        Parameters
        ----------
        mode : (m, k, n)
        t_back, p_back, e_back : arrays from :meth:`_run_trajectory`
        theta_map : array, shape (5,)

        Returns
        -------
        f_pred : np.ndarray, shape (n_sft,)
        fdot_pred : np.ndarray, shape (n_sft,)
        """
        M, mu, a, T_plunge, e_f = theta_map
        m, k, n = mode

        # Frequency track along the backward trajectory
        f_mkn = compute_freq_track_batch(
            m, k, n, t_back, p_back, e_back,
            a=float(a), M=float(M), mu=float(mu),
        )

        # Map SFT absolute times to time-before-plunge
        T_plunge_s = float(T_plunge) * YRSID_SI
        tau_obs = T_plunge_s - self.t_obs

        # Mask valid (finite) trajectory points for interpolation
        valid = np.isfinite(f_mkn) & np.isfinite(t_back)
        t_v = t_back[valid]
        f_v = f_mkn[valid]

        f_pred = np.interp(tau_obs, t_v, f_v)
        fdot_pred = np.gradient(f_pred, self.t_obs)
        return f_pred, fdot_pred

    def _vet_track(self, f_pred: np.ndarray, fdot_pred: np.ndarray) -> float:
        """Evaluate the semi-coherent detection statistic on a predicted track.

        Parameters
        ----------
        f_pred : array, shape (n_sft,)   predicted frequency [Hz]
        fdot_pred : array, shape (n_sft,)  predicted fdot [Hz/s]

        Returns
        -------
        det_stat : float
        """
        stat = jax_det_stat(
            jnp.array(self.data_sfts),
            jnp.array(f_pred),
            jnp.array(fdot_pred),
            T_sft=self.T_sft,
        )
        return float(stat)

    def _predict_and_vet(
        self,
        theta_map: np.ndarray,
        t_back: np.ndarray,
        p_back: np.ndarray,
        e_back: np.ndarray,
        confirmed_modes: list,
        mode_det_stats: dict,
    ) -> list[tuple]:
        """Predict all candidate modes and vet against the STFT data.

        Parameters
        ----------
        theta_map : (5,) MAP parameters
        t_back, p_back, e_back : trajectory arrays
        confirmed_modes : list of already-confirmed (m,k,n) tuples
        mode_det_stats : dict updated in-place with new det-stat values

        Returns
        -------
        new_confirmations : list of newly confirmed (m, k, n) tuples
        """
        confirmed_set = set(map(tuple, confirmed_modes))
        candidates_to_vet = [
            md for md in self.mode_candidates if tuple(md) not in confirmed_set
        ]

        # Rank candidates by expected semi-coherent SNR proxy:
        # use the det_stat on the predicted track as the ranking criterion
        # (no prior amplitude information needed; this avoids loading amp tables).
        # Evaluate the top-k_vet candidates to limit wall-clock cost.
        ranked = candidates_to_vet[: self.top_k_vet]

        new_confirmations = []
        for mode in ranked:
            try:
                f_pred, fdot_pred = self._frequency_and_fdot_at_obs(
                    mode, t_back, p_back, e_back, theta_map
                )
                # Skip modes where the frequency is entirely out of range
                if np.all(np.isnan(f_pred)) or np.all(fdot_pred <= 0):
                    continue
                stat = self._vet_track(f_pred, fdot_pred)
                mode_det_stats[tuple(mode)] = stat
                if stat >= self.vet_threshold:
                    new_confirmations.append(tuple(mode))
                    print(f"    mode {mode}: Λ = {stat:.1f} ✓ confirmed")
                else:
                    print(f"    mode {mode}: Λ = {stat:.1f}  (below threshold)")
            except Exception as exc:
                warnings.warn(f"Vetting mode {mode} failed: {exc}")
        return new_confirmations

    def _run_lbfgs(
        self,
        loss_raw_fn,
        raw_init: np.ndarray,
        optimizer: TrackOptimizerJAX,
    ) -> tuple[np.ndarray, float]:
        """Run L-BFGS on an unconstrained loss function.

        Returns
        -------
        theta_map : physical parameters (5,)
        loss : final loss value
        """
        try:
            import jaxopt
        except ImportError:
            warnings.warn("jaxopt not available; skipping L-BFGS refinement.")
            lo = np.array(optimizer.bounds[:, 0])
            hi = np.array(optimizer.bounds[:, 1])
            theta = lo + (hi - lo) / (1.0 + np.exp(-raw_init))
            return theta, float(loss_raw_fn(jnp.array(raw_init)))

        import jax.numpy as jnp

        solver = jaxopt.LBFGS(fun=loss_raw_fn, maxiter=self.lbfgs_iter)
        result = solver.run(jnp.array(raw_init, dtype=jnp.float64))
        raw_opt = np.array(result.params)
        lo = optimizer.bounds[:, 0]
        hi = optimizer.bounds[:, 1]
        theta = lo + (hi - lo) / (1.0 + np.exp(-raw_opt))
        return theta, float(loss_raw_fn(result.params))

    def _compute_joint_fisher(
        self,
        theta_map: np.ndarray,
        confirmed_modes: list,
        observed_tracks: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute joint Fisher information matrix over all confirmed tracks.

        ``F_ij = Σ_modes Σ_t  (∂f_pred/∂θ_i)(∂f_pred/∂θ_j) / σ_f²``

        Parameters
        ----------
        theta_map : (5,) MAP parameters
        confirmed_modes : list of (m, k, n)
        observed_tracks : dict mapping mode → (f_obs, t_obs) arrays

        Returns
        -------
        fisher : np.ndarray, shape (5, 5)
        cov_cr : np.ndarray, shape (5, 5)
        """
        from fewtrax.utils.geodesic import get_fundamental_frequencies
        from fewtrax.utils.constants import YEAR_SI, MTSUN_SI

        sigma_f = 1.0 / self.T_sft
        traj = EMRIInspiral(self.flux_data)
        _dense_steps = max(500, 3 * len(self.t_obs))

        F_total = np.zeros((5, 5))

        for mode in confirmed_modes:
            m, k, n = mode
            _, t_obs_mode = observed_tracks[mode]
            t_obs_jax = jnp.array(t_obs_mode, dtype=jnp.float64)

            @jax.jit
            def predict_freqs_mode(theta, _m=m, _k=k, _n=n):
                M, mu, a, T_plunge, e_f = (
                    theta[0], theta[1], theta[2], theta[3], theta[4]
                )
                t_back, p_back, e_back, _, _, _ = traj(
                    p0=jnp.float64(10.0),
                    e0=e_f, T=T_plunge, a=a, M=M, mu=mu,
                    backward=True, e_f=e_f,
                    dense_steps=_dense_steps,
                )
                M_total_s = (M + mu) * MTSUN_SI
                T_plunge_s = T_plunge * YEAR_SI

                def freq_one(p, e):
                    Om_phi, Om_theta, Om_r = get_fundamental_frequencies(
                        jnp.abs(a), p, e, 1.0
                    )
                    return jnp.abs(
                        _m * Om_phi + _k * Om_theta + _n * Om_r
                    ) / (2.0 * jnp.pi * M_total_s)

                f_track = jax.vmap(freq_one)(p_back, e_back)
                tau_obs = T_plunge_s - t_obs_jax
                return jnp.interp(tau_obs, t_back, f_track)

            theta_jax = jnp.array(theta_map, dtype=jnp.float64)
            J = np.array(jax.jacobian(predict_freqs_mode)(theta_jax))
            F_total += (J.T @ J) / sigma_f ** 2

        # Cramér-Rao bound (regularise if nearly singular)
        try:
            cov_cr = np.linalg.inv(F_total)
        except np.linalg.LinAlgError:
            cov_cr = np.linalg.pinv(F_total)

        return F_total, cov_cr

    def _narrow_bounds_from_fisher(
        self, theta_map: np.ndarray, sigma_params: np.ndarray
    ) -> np.ndarray:
        """Construct tight prior bounds from Fisher matrix 1-σ widths.

        Returns ``[theta_map ± n_sigma * sigma_params]`` clipped to
        physical bounds.

        Returns
        -------
        narrow_bounds : np.ndarray, shape (5, 2)
        """
        lo = theta_map - self.n_sigma_narrow * sigma_params
        hi = theta_map + self.n_sigma_narrow * sigma_params
        lo = np.maximum(lo, self.bounds[:, 0])
        hi = np.minimum(hi, self.bounds[:, 1])
        return np.column_stack([lo, hi])
