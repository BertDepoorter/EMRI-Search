# Semi-coherent search statistic as a likelihood

Goal: study of secondaries when considering only single tracks. We have a fully differentiable semi-coherent likelihood in terms of the EMRI parameters. We can study the structure of this likelihood as a function of the EMRI parameters, which is very interesting in its own right.

## Formalize the likelihood
In a Bayesian formalism, we have a set of observed data $\mathcal{D}$. We then assume the semi-coherent likelihood 
$$
\mathcal{L}(\mathcal{D}|\vec{\theta}) = \Lambda(\theta)
$$

The standard matched Whittle loglikelihood writes 
$$
\mathcal{L}_{\text{whittle}} = -\frac{1}{2}\langle\mathcal{D} - h|\mathcal{D} - h\rangle \\
\text{where} \langle a | a \rangle = 4\Re \int_0^{\infty} df \frac{|\tilde{a}(f)|^2}{S_n(f)}.
$$
### Discretizing the likelihood
When numerically evaluated, we use the DFT to transform the discrete time series data $\mathcal{D}$ defined on a regularly spaced time grid with $\Delta t = T/N_t$ into discrete frequency bins spaced as $\{0 <f_n = n\Delta f <= N\Delta f | N_f = N_t //2 \}$ We discard the DC component and stop at the Nyquist frequency. The Nyquist frequency is the highest frequency allowed by the time series: $f_{Nyquist} = 1/(2\Delta t)$. 

We know that the smallest possible frequency step is then defined as 
$\Delta f = \frac{1}{2N_t}$ since 
$f_{nyquist} = \frac{1}{2\Delta t} = \frac{N_t}{2T} = N_f\Delta f $ and since for a time-series of even length we have $N_f = \frac{1}{2}N_t$, this gives $\Delta f = \frac{1}{T}$, which makes perfect sense. 

With this in mind we can write down the discretized Whittle loglikelihood:
$$
\mathcal{L} = 4\Re \sum_{n=1}^{N_t //2} \Delta f \frac{|\tilde{a}[f_n]|^2}{S_n[f_n]}
$$
Where the square brackets denote discrete series instead of functions. The power spectral density is simply defined as $S_n[f_n] = S_n(f_n)$. 

## Semi-coherent likelihood
When we do a broad search, we break the data into segments of length $T_{seg}$, and we approximate the signal by a slowly chirping approximation for the fundamental mode. This means the following:
$$
\mathcal{L}_{\text{semi-coherent}} = \sum_{k=1}^{N_s} \mathcal{L}_k
$$
where $\mathcal{L}_k$ is the Whittle loglikelihood for the $k$-th segment. Here we make use a new approximation, being that the signal can be written as a slowly chirping sinusoid:
$$
h_\alpha(f) = A_\alpha e^{i\phi_\alpha + i\Delta \phi_\alpha(t)} \\
\text{where  } \Delta\phi_\alpha(t) = 2\pi f (t-t_\alpha) + \pi \dot{f}_\alpha(t-t_\alpha)^2 + \frac{2\pi}{3}\ddot{f}_\alpha(t-t_\alpha)^3 + \dots
$$
Where we cut have essentiallly truncated the Taylor expanded frequency track after the third order. Now our local waveform is a function of $\{A_\alpha, \phi_\alpha, f_\alpha, \dot{f}_\alpha, \ddot{f}_\alpha\}$ We can now plug this into the Whittle liklihood evaluated at a single segment $k$ and mximize over the amplitude and phase:
$$
\mathcal{L}_\alpha \propto\langle h_\alpha |h_\alpha\rangle
= 4\Re \sum_{n=1}^{N_t //2} \Delta f \frac{|\tilde{h}_\alpha[f_n]|^2}{S_n[f_n]} = 4\Re \sum_{n=1}^{N_t //2} \Delta f \frac{|A_\alpha|^2}{S_n[f_n]} = 4\Re |A_\alpha|^2 \sum_{n=1}^{N_t //2} \Delta f \frac{1}{S_n[f_n]}
$$  

This is now what we consider the semi-coherent likelihood for a single track against the data. We can then sample over the EMRI parameters $\vec{\theta}$ and study the distributions and sampler behaviour. Are there many secondary modes? What is the expected with of th dominant mode? is there clear gaussian behaviour around the dominant mode? Presumably, the more eccentric the system, the less gaussian the likelihood will be for signals with the same matched filtering SNR, since power is more spread out over the harmonics.

We can also define a metric on the parameter space by studying the curvature of the likelihood around the dominant mode. This will give us an idea of how well we can constrain the parameters, and how many templates we need to cover the parameter space. Defining a metric can also help in identifying secindaries and their structure - distribution, width, etc.

## Follow-up: planned studies

The object we now have in hand is a single differentiable scalar field on the
EMRI parameter space,
$$
\Lambda(\vec\theta), \qquad \vec\theta = (M,\,\mu,\,a,\,T_{\rm plunge},\,e_f),
$$
built by chaining a `fewtrax` backward trajectory (differentiable through the
`diffrax` adjoint) into the single-harmonic semi-coherent statistic of
[`multi-harmonic-search-statistic.md`](multi-harmonic-search-statistic.md),
whose Fresnel kernel carries a custom VJP. Because the per-segment statistic
$2|c_\alpha|^2/h_\alpha$ is the matched-filter log-likelihood *ratio* already
maximized over the unknown per-segment amplitude and phase, the natural Bayesian
object is the posterior
$$
p(\vec\theta\mid\mathcal D)\;\propto\;\pi(\vec\theta)\,\exp\!\Big[\tfrac12\,\Lambda(\vec\theta)\Big],
$$
i.e. $\log\mathcal L = \tfrac12\Lambda$ up to a $\vec\theta$-independent constant.
This is the likelihood we will sample. The point of the studies below is not yet
parameter recovery for its own sake — it is to *characterize the geometry of
this likelihood* before we commit it to the full detection pipeline as the
identification-stage likelihood feeding `parismc-jax` and, eventually, the WDM
full-PE stage.

All four studies share one infrastructure investment in week 1 and one compute
campaign in week 2. The whole programme is deliberately sized so that the
implementation is a thin wrapper over code that already exists
(`emrisearch.jax_utils.det_stat`, `TrackOptimizerJAX`, `fewtrax.EMRIInspiral`)
and the science comes from *running* it widely rather than from new algorithms.

### Week 1 — implementation and validation

The shared deliverable is a `parismc`-ready likelihood module,
`semi_coherent_likelihood.py`, exposing

```python
logL, grad_logL = value_and_grad_logL(theta, mode, data_sfts, t_obs)
```

with `logL = 0.5 * det_stat(...)` and `theta` in the unconstrained
(logit-transformed) coordinates already used by `TrackOptimizerJAX`. Concretely:

1. **Glue and gradient check (≈2 days).** Wire `fewtrax` backward integration →
   $\{f_\alpha,\dot f_\alpha\}$ on the SFT mid-time grid → `det_stat`. Validate
   $\nabla_{\vec\theta}\Lambda$ against finite differences across the parameter
   box, and confirm `jit`+`vmap` over a batch of $\theta$ and over the harmonic
   mode reproduces the serial result. This reuses the dense-trajectory
   resolution fix (`dense_steps = max(500, 3 N_{\rm sft})`).
2. **Injection harness (≈1 day).** A generator that produces SFT data for a
   requested $(\vec\theta_{\rm inj}, \text{SNR}, \text{seed})$ by rescaling the
   distance to hit a target *matched-filter* SNR, so that eccentric and circular
   injections are compared at fixed optical depth rather than fixed distance.
3. **Curvature / metric tooling (≈1 day).** `jax.hessian` of $\tfrac12\Lambda$ at
   a point gives the observed Fisher matrix $F_{ij}=-\partial_i\partial_j\log\mathcal L$;
   wrap it with eigen-decomposition and Cramér–Rao covariance
   $\Sigma = F^{-1}$. Expose a `mismatch(θ, dθ) ≈ ½ dθᵀ F dθ` helper for the
   template-density study.
4. **Sampler smoke test (≈1 day).** Run `parismc-jax` (PARIS global multimodal
   sampler) on one mid-eccentricity injection at SNR 30 to fix step sizes,
   temperatures, chain count and convergence diagnostics (split-$\hat R$,
   ESS), and to confirm the gradient-driven proposals behave on the real
   likelihood surface.

This leaves the differentiable likelihood, an injection factory, a Fisher tool,
and a tuned sampler config — exactly the inputs the week-2 campaign consumes.

### Week 2 — the PE campaign

A single grid of `parismc` runs, launched in parallel via `vmap`/batched chains
on GPU, spanning

- **eccentricity** $e_f \in \{0.0,\,0.1,\,0.3,\,0.5,\,0.7\}$ (the control
  variable for non-Gaussianity),
- **SNR** $\in \{20,\,30,\,50\}$ (controls how deep secondaries sit relative to
  the dominant mode), and
- **3 noise realizations** per cell, plus one **zero-noise** run per cell to
  separate likelihood structure from noise scatter.

That is $5\times3\times(3+1)=60$ posterior runs at a fixed reference mass/spin,
each cheap because the likelihood is a frequency-only track statistic with no
amplitude evaluation. The four studies are then read off the *same* 60-run
output set.

#### Study A — Secondary-mode census

Run PARIS globally over the full prior box (not seeded at the injection) and ask
the questions raised above directly: how many resolvable modes does
$\Lambda(\vec\theta)$ support, where do they sit, and what is their height
relative to the dominant mode? We label modes by clustering the cooled chains,
record their $(\Delta\Lambda,\ \text{location},\ \text{integer }(m,n)$ track
they correspond to$)$, and test the standing hypothesis that the dominant
single-track degeneracy is the $M\!-\!\mu$ ridge (amplitude breaks $M$, the track
shape constrains the combination), with discrete secondaries arising from
harmonic mislabelling. **Deliverable:** a mode-count and mode-separation table
vs $(e_f,\text{SNR})$, and an annotated 2D $\Lambda$ slice showing the secondaries
for a representative eccentric case.

#### Study B — Width and Gaussianity of the dominant mode

For each run, compare the *sampled* posterior covariance within the dominant
mode against the Fisher prediction $\Sigma=F^{-1}$ computed at the MAP. We
quantify non-Gaussianity with (i) the Mahalanobis residual of the samples under
the Fisher Gaussian, (ii) marginal skewness/excess-kurtosis, and (iii) a
symmetrized KL between the sampled mode and its Gaussian approximation. The
zero-noise runs isolate the *intrinsic* curvature; the noisy runs show how much
of the spread is noise scatter. **Deliverable:** Fisher-vs-sampled overlay
corner plots and a non-Gaussianity scalar per grid cell.

#### Study C — Eccentricity drives non-Gaussianity (the main physics result)

This is the headline test of the conjecture in the note: *at fixed matched-filter
SNR, more eccentric systems should yield a less Gaussian, more
structured likelihood, because power is spread across more harmonics.* Using the
fixed-SNR injection harness, we plot the Study-B non-Gaussianity scalar and the
Study-A secondary count as functions of $e_f$. A clean monotonic trend would be
a publishable, intuitive statement about why eccentric EMRIs are harder for
single-track identification and why the multi-harmonic statistic is needed.
**Deliverable:** non-Gaussianity and mode-multiplicity vs $e_f$ at fixed SNR,
with the circular case as the Gaussian baseline.

#### Study D — An information metric and template-bank density

From the same Fisher matrices we define the parameter-space metric
$g_{ij}=F_{ij}$ and the proper volume $\int\sqrt{\det g}\,d^5\theta$. We report
the principal axes (which directions are well/poorly constrained — expected to
expose the $M\!-\!\mu$ near-degeneracy), the eccentricity dependence of the
metric volume, and a back-of-the-envelope template count for a target $3\%$
mismatch using $\tfrac12\,\delta\theta^\top g\,\delta\theta = $ mismatch. This
connects directly to the broader pipeline: it tells us how densely PARIS must
seed anchors and whether the metric can pre-screen secondaries before sampling.
**Deliverable:** metric eigenstructure and template-count estimates vs $(e_f,
\text{SNR})$.

### Scope, risks, and what is explicitly out

The campaign is held to two weeks by three deliberate restrictions: a *single
harmonic track* (no Viterbi, no multi-voice — those live in
[`multi-harmonic-search-statistic.md`](multi-harmonic-search-statistic.md)), a
*frequency-only* likelihood (no amplitude/response modelling), and a *fixed
reference $(M,\mu,a)$* with only $(e_f,\text{SNR},\text{seed})$ scanned. The main
risk is sampler tuning on a genuinely multimodal surface eating into week 2; the
mitigation is that PARIS is purpose-built for multimodality and the week-1 smoke
test front-loads that tuning. The natural sequel — repeating Study C with the
*multi-harmonic* $\Lambda_{\rm MH}$ to show that the secondaries collapse once
the comb is included — is the obvious follow-on but is out of scope here.
