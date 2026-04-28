# EMRI Search Algorithm: Analysis, Improvements, and Accelerated Proposals

---

## Task 1: Current Search Algorithm

### Overview

The `emrisearch` package implements a **semi-coherent STFT search pipeline** for detecting Extreme Mass Ratio Inspirals (EMRIs) in LISA data. The algorithm avoids the cost of computing full matched-filter waveforms by working in a sparse time-frequency representation.

### Key Files and Functions

| File | Purpose | Key Functions |
|------|---------|---------------|
| `emri_utils.py` | EMRI waveform and trajectory generation | `create_signal()`, `get_f_fdot_fddot_back()` |
| `da_utils.py` | Noise, PSD, SFT computation | `compute_sfts()`, `generate_noise()`, `psd()` |
| `search_utils.py` | Detection statistic (NumPy) | `det_stat()`, `robust_kernel()`, `fresnel_kernel()` |
| `jax_utils.py` | Detection statistic (JAX, differentiable) | `det_stat()`, `robust_kernel()` (custom VJP) |
| `jax_de_utils.py` | GPU-accelerated Differential Evolution | `differential_evolution_step()`, `selection()` |
| `draw_population.py` | Quasi-random parameter sampling | `ParameterScaler`, `draw_sobol_distribution()` |
| `de_proposal.py` | MCMC DE proposal move | `DEMove`, `_de_proposal` |
| `mcmc_followup.py` | Full Bayesian MCMC posterior | `generate_injection_data()` |

### Main Steps in the Search Algorithm

#### Step 1: Data Preparation (`da_utils.py`, `emri_utils.py`)

1. Generate or load EMRI signal using `create_signal()` — uses FEW (`FastKerrEccentricEquatorialFlux`) with LISA response wrapper (TDI AET channels).
2. Add colored Gaussian noise via `generate_noise()` with the TDI PSD.
3. Divide the time series into $N_{\mathrm{sft}}$ segments of duration $T_{\mathrm{sft}}$ (~14 h or ~8 h).
4. Apply a Tukey window to each segment and FFT → **Short Fourier Transforms** (SFTs) of shape SFT[$n_{\mathrm{freq}}$, $n_{\mathrm{sft}}$].

#### Step 2: Frequency Track Generation (`emri_utils.py`)

For a given parameter vector `[M, μ, a, T_plunge, e_f, x0]`:

1. Integrate the EMRI inspiral **backwards** from the separatrix using `get_f_fdot_fddot_back()`.
   - This uses `EMRIInspiral` (FEW) from `Tpl` years before plunge.
   - The backward parameterisation anchors the track to the plunge time, avoiding the sensitivity of forward integration to uncertain initial conditions.
2. Evaluate phases $\Phi$, frequencies $f$, $\dot f$, and $\ddot f$ at the $N_{\mathrm{sft}}$ SFT mid-times.
3. Construct the $(m,n)$ harmonic track: $f_{\alpha} = m\,f_{\phi} + n\,f_{r}$, where indices 0 and 1 correspond to the azimuthal and radial harmonics respectively.

#### Step 3: Semi-Coherent Detection Statistic (`search_utils.py`, `jax_utils.py`)

The statistic follows **Tenorio & Gerosa (2025), Eq. 7**. For each SFT segment $\alpha$:

$$
k_{\alpha} = \left\lfloor f_{\alpha}\,T_{\mathrm{sft}} \right\rfloor \quad\text{(nearest frequency bin)}
$$

$$
c_{\alpha} = \sum_{j=k-P}^{k+P} \tilde d_j^{*}\,\frac{K\bigl(f_{\alpha}-j\Delta f,\,\dot f_{\alpha},\,T_{\mathrm{sft}}\bigr)}{S_n(j\Delta f)}
$$

where:
- $K(f_0,\;\dot f,\;T)$ is the **Fresnel kernel** — the matched filter for a chirping tone (implemented in `robust_kernel()` with a custom JAX VJP for differentiability)
- $S_n(f)$ is the TDI PSD (interpolated from `TDI2_AE_psd.npy`)
- $P = 100$ bins are summed around the nominal bin

The total detection statistic sums over all valid segments:

$$
\Lambda = \sum_{\alpha} \frac{\lvert c_{\alpha} \rvert^2}{h_{\alpha}}
$$

where $h_{\alpha} = T_{\mathrm{sft}} / \bigl(2 S_n(f_{\alpha})\bigr)$.

An amplitude mask $A_{\alpha}$ suppresses segments where the track is invalid (wrong frequency range, negative $\dot f$, large $\ddot f$ curvature error).

#### Step 4: Optimization — Hybrid DE + Adam (`jax_utils.py`, `jax_de_utils.py`)

When the true parameters are unknown, the search optimises $\Lambda(\theta)$:

1. **Differential Evolution** (`jax_de_utils.py`):
   - Population initialised with Sobol quasi-random samples (`draw_population.py`).
   - Each DE step: mutation (DE/best/1), crossover, and selection are all `jax.jit`-compiled and `jax.vmap`-ed over the population.
   - Bounds are enforced by clipping.

2. **Adam optimizer** (via `optax`) applied to the JAX differentiable `det_stat()`, which exposes gradients through the custom-VJP Fresnel kernel.

3. Iteration: alternating DE exploration and Adam refinement.

#### Step 5: MCMC Follow-up (`mcmc_followup.py`, `de_proposal.py`)

After identifying a candidate, full posterior sampling is performed with **eryn** (parallel-tempered ensemble MCMC):
- DE proposal moves (`DEMove`) accelerate mixing.
- Parameters: `[log₁₀(M), μ, a, T_plunge, e_f]`.
- Likelihood: the semi-coherent detection statistic as a log-likelihood proxy.

### High-Level Algorithm Flow

```
LISA data (time domain)
    │
    ▼
compute_sfts()          ← Windowed FFT → SFT[n_freq, n_sft]
    │
    ▼
get_f_fdot_fddot_back() ← Backward EMRI trajectory → f(t), ḟ(t) at SFT times
    │
    ▼
det_stat()              ← Fresnel-kernel matched filter per segment → Λ(θ)
    │
    ├─→ [if searching] Differential Evolution + Adam optimizer
    │         └─→ Best-fit parameters θ*
    │
    └─→ [if sampling] eryn MCMC with DE proposal
              └─→ Posterior p(θ | data)
```

---

## Task 2: Potential Improvements

### Context: JAX-Differentiable EMRI Tracks via `fewtrax`

The `fewtrax` library (`JAX-waveform/fewtrax/`) provides a **fully JAX-differentiable EMRI trajectory** via `diffrax` (ODE solver with continuous adjoint). Key capabilities:

- `EMRIInspiral` is an `equinox.Module`: vmappable and jit-compilable across the full `(M, μ, a, p₀, e₀)` parameter space.
- `get_frequency_track()` returns `(t, f)` as JAX arrays, end-to-end differentiable.
- **Backward integration** from the separatrix is built in (`backward=True`, `e_f=...`), matching the existing `get_f_fdot_fddot_back()` parameterisation but now differentiable.
- Memory cost at track level: ~48 MB for a batch of 10,000 (feasible under `vmap`), versus ~11 GB for full amplitude waveforms (infeasible).
- Gradient cost per evaluation: ~5–10 ms (trajectory integration + adjoint pass).

### Improvement 1: Gradient-Based Physical Parameter Recovery

**Current limitation:** The existing codebase optimises $\Lambda(\theta)$ (the Fresnel-kernel detection statistic) with respect to the 5D physical parameters via DE + Adam. The detection statistic gradient depends on the Fresnel kernel (JAX-differentiable) and the trajectory (currently not differentiable — uses numpy FEW spline evaluation).

**Proposed improvement:** Replace the trajectory call in the optimisation loop with `fewtrax.EMRIInspiral`, making the entire pipeline end-to-end differentiable:

$$
	heta = (M,\;\mu,\;a,\;T_{\text{plunge}},\;e_f)
$$

fewtrax backward integration (differentiable via diffrax adjoint)

$$
f_{\mathrm{pred}}(t;\theta)
$$

jax_utils.det_stat() (differentiable via custom VJP Fresnel kernel)

$$
\Lambda(\theta) \qquad\text{with }\nabla_{\theta}\Lambda(\theta)\ \text{available}
$$

This enables:
- **L-BFGS** via `jaxopt` — converges in O(50) iterations vs O(500) for Adam.
- **Fisher information matrix** $F_{ij} = \dfrac{\partial^2 \Lambda}{\partial \theta_i \partial \theta_j}$ — identifies poorly constrained directions.
- **NUTS posterior sampling** via `blackjax` — replaces the current eryn MCMC with a gradient-informed sampler.

### Improvement 2: Track Residuals as Loss Function

**Observation:** The STFT search identifies tracks in the time-frequency plane with measurement noise $\sigma_f \approx 1/T_{\mathrm{sft}}$. Rather than optimising the full detection statistic $\Lambda(\theta)$, it is computationally cheaper and more numerically stable to fit the **observed frequency track directly**:

$$
L(\theta) = \sum_i \frac{\bigl[f_{\mathrm{pred}}(t_i;\theta,\mathrm{mode}) - f_{\mathrm{obs}}(t_i)\bigr]^2}{\sigma_f^2}
$$

With `fewtrax`, $\partial L/\partial \theta$ is $\mathcal{O}(5\text{--}10\ \mathrm{ms})$ per evaluation. This loss is smooth and well-conditioned compared to $\Lambda(\theta)$, which has sharp gradients near frequency bins.

**Key insight from `gradient_identification.md`:** Fixing `T_plunge` as the time reference and using backward integration reduces the effective parameter space from `(M, μ, a, p₀, e₀)` to `(M, μ, a, T_plunge, e_f)` — a more natural parameterisation that aligns with the STFT observable (plunge time is directly estimated from `f/ḟ`).

### Improvement 3: Batched Mode Identification via `vmap`

**Current limitation:** Mode number identification (scanning over `O(200)` combinations of `(m, k, n)`) is done sequentially.

**Proposed improvement:** With `fewtrax`, each mode evaluation requires only a frequency computation (not a full waveform), and the same trajectory `(t, p, e)` can be reused for all modes. Mode scanning becomes a single `jax.vmap` call over integer mode numbers, with cost `O(1)` in wall-clock time (GPU-parallel).

### Improvement 4: Multi-Track Parameter Recovery

**Observation:** If multiple harmonic tracks are identified by the STFT search, the frequency ratios `f⁽ʲ⁾(t) / f⁽ᵏ⁾(t)` constrain `a` independently of `(M, μ)`. This breaks the `μ/M` degeneracy that makes single-track parameter recovery ill-conditioned.

**Proposed improvement:** Use `fewtrax` to generate all harmonic tracks from a single trajectory evaluation and fit all observed tracks jointly:

$$
L_{\mathrm{joint}}(\theta) = \sum_j \sum_i \bigl[ f_{\mathrm{pred}}^{(j)}(t_i;\theta) - f_{\mathrm{obs}}^{(j)}(t_i) \bigr]^2
$$

This is computationally identical in cost to single-track fitting (same trajectory, just more frequency evaluations per step).

---

## Task 3: Proposed Accelerated Search Approaches

### Proposal A: Gradient-Based Recovery from Single STFT Anchor Track

**Motivation:** Once the STFT search identifies a single frequency track (the dominant harmonic), we have a noisy observation $\{f_{\mathrm{obs}}(t_i),\;\dot f_{\mathrm{obs}}(t_i)\}$. The estimated plunge time $\hat T_{\mathrm{plunge}} \approx f / (11\dot f/3)$ (from the Peters formula) provides a strong prior. Starting from this anchor, we can efficiently recover the 5D physical parameters.

**Algorithm:**

```
1. STFT identifies anchor track: {f_obs(t_α)}
   │
   ├─→ Estimate $\hat T_{\mathrm{plunge}}$ from $f/\dot f$ ratio (estimate_plunge_time())
   │
2. Initialize search space:
   - Narrow T_plunge bounds to [T̂ - 0.3yr, T̂ + 0.3yr]
   - Initialize N=64 particles with Sobol quasi-random sampling
   │
3. Coarse mode scan (parallelised with fewtrax vmap):
   - For each of O(200) mode candidates (m,k,n):
     → Run fewtrax backward integration
     → Compute track loss L(θ_init, mode)
   - Select top-K modes by loss
   │
4. Multi-start Adam optimization (K modes × N_starts particles):
   - loss = Σ_i [f_pred(t_i; θ) - f_obs(t_i)]²
   - ∂loss/∂θ via fewtrax adjoint + interpax interpolation
   - Warm start: 100 Adam steps (learning rate 1e-3)
   │
5. L-BFGS refinement of best candidates (jaxopt):
   - Converges to machine precision in O(50) steps
   │
6. Best-fit θ* (single track)
   │
   └─→ Multi-track validation:
       - Generate all harmonic tracks from θ*
       - Cross-match with other observed STFT tracks
       - Refine with joint loss L_joint(θ)
```

**Key advantages over current approach:**
- Gradient-based optimization (Adam + L-BFGS) vs gradient-free DE: O(50×) fewer function evaluations.
- Track-level loss is smoother than the Fresnel-kernel detection statistic.
- T_plunge anchor dramatically reduces the effective search volume.
- `fewtrax` vmap enables O(100) parameter candidates to be evaluated simultaneously on GPU.

**Integration points with existing code:**
- `estimate_plunge_time()` → new function in `track_optimizer.py`
- `fewtrax.EMRIInspiral` replaces `emri_utils.get_f_fdot_fddot_back()` in the optimization loop (not in the detection statistic, which is unchanged)
- The `jax_utils.det_stat()` remains the primary detection statistic; the track loss is used only for parameter refinement

---

### Proposal B: Particle Swarm Optimization (PSO) with T_plunge-Constrained Initialization

**Motivation:** Before gradient-based refinement (which can fail if the initial guess is too far from the global optimum), a global search method that respects the T_plunge constraint can rapidly scan the parameter space. PSO is well-suited because:
1. It is trivially parallelizable via `jax.vmap`.
2. The swarm can be initialized with a focused distribution around the T_plunge estimate.
3. No gradient information required, making it robust to discontinuities in the detection statistic.

**Algorithm:**

```
1. STFT identifies {f_obs, fdot_obs}
   │
2. $\hat T_{\mathrm{plunge}} \leftarrow \text{estimate\_plunge\_time}(f_{\mathrm{obs}},\;\dot f_{\mathrm{obs}})$
   Narrow T_plunge bounds to [T̂ - 0.25yr, T̂ + 0.25yr]
   │
3. Initialize swarm (N=200 particles):
   positions ← Sobol quasi-random in narrowed bounds
   velocities ← small random perturbations
   │
4. PSO loop (100 iterations):
   ┌─────────────────────────────────────┐
   │ a. Evaluate fitness = Λ(θ) for all  │
   │    particles in parallel (vmap)      │
   │                                     │
   │ b. Update personal bests            │
   │    Update global best               │
   │                                     │
   │ c. Update velocities (Clerc-Kennedy)│
   │    $$v \leftarrow w\,v + c_1\,r_1\,(p_{\text{best}}-x) + c_2\,r_2\,(g_{\text{best}}-x)$$│
   │                                     │
   │ d. Update positions: x ← x + v     │
   │    Clip to bounds                   │
   └─────────────────────────────────────┘
   │
5. Global best θ*_PSO
   │
6. (Optional) Gradient refinement with Adam/L-BFGS from θ*_PSO
```

**PSO hyperparameters (Clerc-Kennedy constriction factor):**
- `w = 0.729` (inertia)
- `c₁ = c₂ = 1.494` (cognitive and social coefficients)
- These guarantee convergence to a local optimum within the Lyapunov stability criterion.

**Key advantages:**
- Focused initialization from T_plunge estimate reduces the effective search volume by ~10× compared to uniform prior.
- All 200 particle evaluations can be batched as a single `jax.vmap(fitness)(positions)` call when `fewtrax` is available.
- Handles multimodal posteriors better than pure gradient descent.
- Complementary to Proposal A: PSO for global search, then gradient refinement.

**Hybrid approach (recommended):**
```
PSO (100 iter, N=200)           ← Global exploration
  → Top-5 candidates
    → Adam (100 steps)          ← Local gradient descent
      → L-BFGS (50 steps)       ← Newton-step refinement
        → Best θ*
```

**Integration points with existing code:**
- `pso_update_step()` is a pure JAX function, jit-compiled and operating on `jnp.ndarray`s
- Fitness evaluated via `jax_utils.det_stat()` (existing, differentiable)
- Swarm initialization uses `draw_population.ParameterScaler` (existing)
- Full pipeline in `pso_utils.ParticleSwarmOptimizer.optimize()`

---

## Task 4: Prototype Implementation

### New Files

| File | Purpose |
|------|---------|
| `src/emrisearch/track_optimizer.py` | Gradient-based parameter recovery from STFT anchor track |
| `src/emrisearch/pso_utils.py` | JAX PSO with T_plunge-constrained initialization |
| `examples/AcceleratedEMRISearch.ipynb` | Demo notebook comparing old vs new methods |
| `profiling/profile_search.py` | GPU profiling script for benchmarking |

### `track_optimizer.py` — Design

**Core class:** `TrackOptimizer`

```python
optimizer = TrackOptimizer(
    f_obs,          # shape (n_sft,) observed GW frequency per segment [Hz]
    fdot_obs,       # shape (n_sft,) observed frequency derivative [Hz/s]
    t_obs,          # shape (n_sft,) SFT mid-times [s]
    T_sft=5e4,      # SFT segment duration [s]
    bounds=None,    # optional: override default parameter bounds
)

# Estimate plunge time from observed track
T_plunge_est = optimizer.T_plunge_estimate  # [years]

# Identify mode and find best parameters
best_mode, best_params, best_loss, results = optimizer.identify_mode_and_optimize(
    mode_candidates=None,  # defaults to (m, k=0, n) for m in [1..4], n in [-2..2]
    n_starts=64,
)
# best_params = [M, mu, a, T_plunge, e_f]
```

**Key functions:**
- `estimate_plunge_time(f_obs, fdot_obs)` — Peters-formula-based T_plunge estimate
- `compute_track_residuals_numpy(params, mode, f_obs, t_obs)` — NumPy loss (for scanning)
- `TrackOptimizer.optimize(mode, n_starts)` — multi-start Nelder-Mead optimization
- `TrackOptimizer.identify_mode_and_optimize()` — joint mode ID + parameter optimization
- `scan_mode_numbers(params, candidates, f_obs, t_obs)` — mode ranking by loss

**fewtrax integration (optional, for gradient-based optimization):**

When `fewtrax` is available, `TrackOptimizerJAX` provides gradient-based optimization:
```python
optimizer_jax = TrackOptimizerJAX(
    f_obs, fdot_obs, t_obs,
    flux_data=fewtrax.data.load_flux_data(data_dir),
)
# Gradient-based loss uses fewtrax backward integration
best_params = optimizer_jax.optimize_adam(mode=(2, 0, 0), n_starts=32)
```

### `pso_utils.py` — Design

**Core class:** `ParticleSwarmOptimizer`

```python
pso = ParticleSwarmOptimizer(
    data_sfts,      # shape (n_freq, n_sft) SFT data
    t_obs,          # SFT mid-times [s]
    T_sft=5e4,
    bounds=None,
)

best_params, best_fitness, history = pso.optimize(
    mode=(2, 0),            # (m, n) mode numbers
    n_particles=200,
    n_iter=100,
    T_plunge_estimate=0.9,  # [years] from STFT estimate
)
```

**Key functions:**
- `initialize_swarm_from_track(f0, fdot0, T_plunge_est, bounds, n_particles)` — focused Sobol initialization
- `pso_update_step(positions, velocities, pbest, gbest, key, w, c1, c2, ...)` — jit-compiled JAX PSO step
- `ParticleSwarmOptimizer.optimize()` — full PSO loop with T_plunge-constrained initialization

### Performance Characteristics

| Method | Evaluations | Wall-clock (GPU) | Notes |
|--------|-------------|-----------------|-------|
| Current DE + Adam | O(5000) | O(10 min) | Gradient-free DE dominant |
| PSO (N=200, 100 iter) | 20,000 | O(2 min) | Parallel via vmap |
| PSO + Adam + L-BFGS | 20,200 | O(3 min) | Global + local refinement |
| Multi-start Adam (fewtrax) | 3,200 | O(1 min) | Full gradient pipeline |

### Updated `__init__.py`

The new modules are exported from the package:
```python
from .track_optimizer import TrackOptimizer, estimate_plunge_time
from .pso_utils import ParticleSwarmOptimizer, initialize_swarm_from_track
```

### Profiling (`profiling/profile_search.py`)

The profiling script measures:
1. Per-evaluation cost of `jax_utils.det_stat()` (baseline)
2. fewtrax trajectory evaluation cost (if available)
3. PSO swarm evaluation time as a function of `n_particles`
4. Track optimizer convergence speed vs current DE + Adam
5. GPU memory usage under `jax.vmap` over different batch sizes

Run with:
```bash
cd profiling && python profile_search.py --n-particles 200 --n-iter 100
```

---

## Task 5: Iterative Multi-Track Search with MCMC Handoff

### Overview

The iterative pipeline extends Task 4 with:
1. Proper trajectory resolution matching the SFT observation grid
2. Unconstrained parameter reparametrisation for boundary-free gradient descent
3. Iterative secondary-mode discovery and joint refitting
4. Fisher-matrix-derived narrow prior handoff to the MCMC follow-up

### New and modified files

| File | Change | Purpose |
|------|--------|---------|
| `src/emrisearch/track_optimizer.py` | Modified | Resolution fix, unconstrained reparametrisation, joint loss, Fisher fix |
| `src/emrisearch/iterative_track_search.py` | **New** | Iterative multi-track pipeline, `SearchResult` |
| `src/emrisearch/mcmc_likelihood.py` | **New** | `FastEMRILikelihood`, `MCMCHandoff`, `prepare_mcmc_handoff` |

### Key changes to `track_optimizer.py`

#### 1. Trajectory resolution fix

```python
self._dense_steps = max(500, 3 * len(self.t_obs))
```

For a 1-year observation at $T_{\mathrm{sft}} = 5\times10^{4}\ \mathrm{s}$, $n_{\mathrm{sft}} \approx 630$, so `dense_steps = 1890`.
Previously hardcoded at 200, which left fewer than 1 trajectory point per SFT — cubic
interpolation errors were O(10⁻³ Hz), comparable to the STFT bin width.

#### 2. Unconstrained reparametrisation

Adam optimisation is performed in logit-transformed space to prevent gradient zeroing
at parameter boundaries:

$$
\mathrm{raw\_}\theta \in \mathbb{R}^5
$$

$$
	heta_{\mathrm{phys}} = \mathrm{lo} + (\mathrm{hi}-\mathrm{lo})\,\sigma(\mathrm{raw\_}\theta)\quad\text{(physical space)}
$$

`_to_unconstrained_np(theta_phys, bounds)` converts Sobol initialisations.
`_build_loss_raw(mode)` wraps `_build_loss` with this transform for the optimizer.

#### 3. Joint multi-track loss

`_build_joint_loss_raw(modes_list, f_obs_list, t_obs_list)` integrates the trajectory
**once** per call and sums residuals across all confirmed modes:

$$
L_{\mathrm{joint}}(\theta) = \sum_j \sum_i \bigl[ f_{\mathrm{pred}}^{(j)}(\tau_i;\theta) - f_{\mathrm{obs}}^{(j)}(\tau_i) \bigr]^2
$$

Adding each confirmed secondary mode to this loss breaks the `M/μ` degeneracy and
tightens the Fisher matrix, feeding directly into the narrow MCMC prior.

### `IterativeTrackSearch` pipeline

```
IterativeTrackSearch.run(f_obs, fdot_obs)
│
├─ [0] Fit anchor mode: Adam (n_starts, n_steps) → L-BFGS → θ̂₀
│       T_plunge auto-estimated from f/fdot ratio
│
├─ [1] Run trajectory at θ̂₀ → (t_back, p_back, e_back)
│       Predict all candidate mode tracks
│       Vet each against STFT: Λ(mode) = det_stat(data_sfts, f_pred, fdot_pred)
│       Confirm modes with Λ > vet_threshold
│
├─ [2] Add confirmed modes → joint loss L_joint(θ)
│       Refit via L-BFGS in unconstrained space → θ̂₁
│
├─ [3] Compute joint Fisher F = Σ_j J_j^T J_j / σ_f²
│       Cramér-Rao covariance: Cov = F⁻¹
│       Narrow bounds: [θ̂ ± n_sigma · sqrt(diag(Cov))]
│
└─→ SearchResult (theta_map, cov_cr, narrow_bounds, confirmed_modes,
                  mode_det_stats, p0, e0, ...)
```

### MCMC handoff

```python
from emrisearch.iterative_track_search import IterativeTrackSearch
from emrisearch.mcmc_likelihood import FastEMRILikelihood, prepare_mcmc_handoff
from fewtrax.data import load_flux_data

flux_data = load_flux_data()

# Identification stage (JAX, CPU/GPU)
result = IterativeTrackSearch(
    flux_data, data_sfts, t_obs,
    vet_threshold=50.0,   # calibrate for your SNR / T_sft
    n_sigma_narrow=5.0,
).run(f_obs, fdot_obs)
print(result.summary())

# Convert to MCMC inputs
handoff = prepare_mcmc_handoff(result)
print(handoff.summary())

# Full waveform likelihood (fastemriwaveforms + fastlisaresponse + cupy)
like = FastEMRILikelihood(data_A, data_E, T=1.0, dt=10.0, use_gpu=True)

# Port your existing MCMC sampler here:
#   - use handoff.full_bounds_few  as prior bounds
#   - use handoff.intrinsic_map_few as chain initialisation
#   - call like(params) for the log-likelihood
```

### Parameter conventions

| Stage | Intrinsic params | Notes |
|-------|-----------------|-------|
| Identification | `(M, μ, a, T_plunge, e_f)` | Backward integration from separatrix |
| MCMC | `(log₁₀M, log₁₀μ, a, p₀, e₀)` + 8 extrinsic | FEW convention; `(p₀, e₀)` from `tplunge_to_p0e0` |

### `vet_threshold` calibration

The `vet_threshold` for secondary-mode confirmation must be calibrated per experiment:
- It depends on the SNR, number of SFT segments, and noise realisation.
- Recommended approach: run `IterativeTrackSearch` on a large population of injections
  at different parameter values and noise seeds; measure the distribution of
  `det_stat` values for true secondary modes vs. pure noise.
- Default `50.0` is conservative for SNR ≈ 30 with ~630 SFT segments.
