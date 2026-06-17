## Multi-harmonic search statistic
We follow the derivation of th search statistic in Speri et al. 2025 (ab uno disce omnes: single harmonic search for extreme mass-ratio inspirals in LISA)

### Goal
The single harmonic search statistis is semi-coherent, maximized of phase and amplitude and only relies on the slow chirp approximation within one single SFT bin. We aim to extend this search statistic to two sets of f, fdot parameters in each time bin corresponding to the two fundamental frequency tracks of the EMRI signal (azimuthal and radial). Higher mode tracks can then be build from integer linear combinations of these two basis function.

### Method
We start by proposing two sets of f, fdot in the first time bin and then integrate along the next two tracks. We can rely on the following observations to reduce the cost of this statistic: 
- fdot should be positive: the signal chirps up in frequency
- The track is slowly-evolving and must be continuous: the next f, fdot pair must be close to the previous one. We can use this to limit the number of proposed tracks in the next time bin.

Intuitively, this method works as follows. Pick some f, fdot pair to start from and then integrate along the next time bins. As the integration proceeds, some tracks will deviate from the true signal and start averaging out due to the noise statistics, while the tracj closely following the true signal will keep accumulating power. By maximizing over the proposed tracks, we can identify the one that best matches the true signal. The search statistic is then an approximate version of the mathced filtering statistic for the multi-harmonic signal, but only relies on the frequency evolution and is maximzed over the amplitudes and phases. This is the key to reducing computational cost while still being sensitive to the multi-harmonic structure of the signal.

The purpose of this search statistic is to provide a fast and robust way of identifying a signal in the data which follows this frequency evolution and the relation between the two tracks. The result from this search can be used in a follow-up stage to perform a more refined identification stage to map back to the true EMRI parameters. This is similar to the way the single harmonic search statistic is used in the current pipeline, but should be more sensitive to the multi-harmonic structure and therefore to higher modes.

### Mathematical formulation

**SFT data model.** The strain time series $d(t)$ is split into $N$ segments of length $T_{\mathrm{sft}}$ with mid-times $t_\alpha$, $\alpha=0,\dots,N-1$. Each windowed segment is Fourier transformed to give the SFT $\tilde d_{j,\alpha}$ on the frequency grid $f_j = j\,\Delta f$, with $\Delta f = 1/T_{\mathrm{sft}}$. Within a single segment an EMRI harmonic is well approximated by a linearly chirping tone (the *slow-chirp* approximation),

$$
h_\alpha(t) \approx A\,\exp\!\Big[\,2\pi i\big(f_\alpha t + \tfrac12 \dot f_\alpha t^2\big) + i\phi\,\Big],\qquad t\in[0,T_{\mathrm{sft}}],
$$

valid while $|\ddot f_\alpha|\,T_{\mathrm{sft}}^3/6 \lesssim 1$.

**Per-segment matched filter (Fresnel kernel).** The overlap of one chirping tone with the segment is the Fresnel kernel

$$
K(f_0,\dot f,T_{\mathrm{sft}}) = \int_0^{T_{\mathrm{sft}}} e^{\,2\pi i\,(f_0 t + \frac12 \dot f t^2)}\,dt
= \frac{e^{-i\pi f_0^2/\dot f}}{\sqrt{2\dot f}}\,\big[\,\mathcal C(u)+i\,\mathcal S(u)\,\big]_{u_-}^{u_+},
$$

with $u_\pm = \sqrt{2/\dot f}\,(f_0 + \dot f t)\big|_{t=0,\,T_{\mathrm{sft}}}$ and $\mathcal C,\mathcal S$ the Fresnel integrals; for $\dot f\to 0$ it reduces to the Dirichlet kernel $T_{\mathrm{sft}}\,e^{i\pi f_0 T_{\mathrm{sft}}}\,\mathrm{sinc}(f_0 T_{\mathrm{sft}})$.

**Semi-coherent single-harmonic statistic** (Tenorio & Gerosa 2025, Eq. 7; Speri et al. 2025). For a track $\{f_\alpha,\dot f_\alpha\}$, project the data in each segment onto the kernel within $\pm P$ bins of the central bin $k_\alpha=\lfloor f_\alpha T_{\mathrm{sft}}\rfloor$:

$$
c_\alpha = \Delta f \sum_{j=k_\alpha-P}^{k_\alpha+P} \frac{\tilde d_{j,\alpha}^{*}}{S_n(f_j)}\,K\!\big(f_\alpha - f_j,\,\dot f_\alpha,\,T_{\mathrm{sft}}\big).
$$

Maximizing analytically over the unknown per-segment amplitude $A$ and phase $\phi$ yields the segment statistic $2|c_\alpha|^2/h_\alpha$ with $h_\alpha = T_{\mathrm{sft}}/\big(2 S_n(f_\alpha)\big)$, and the semi-coherent (incoherent sum over segments) statistic

$$
\Lambda\big(\{f_\alpha,\dot f_\alpha\}\big) = \sum_\alpha A_\alpha\,\frac{2\,|c_\alpha|^2}{h_\alpha},
$$

where $A_\alpha\in\{0,1\}$ masks segments outside the valid band, with $\dot f_\alpha\le 0$, or breaking the slow-chirp condition.

**Two fundamental tracks.** An eccentric EMRI emits a comb of harmonics labelled by azimuthal and radial integers $(m,n)$, each following

$$
f_{mn}(t) = m\,f_\phi(t) + n\,f_r(t),\qquad
\dot f_{mn}(t) = m\,\dot f_\phi(t) + n\,\dot f_r(t).
$$

The whole comb is therefore fixed by the *two fundamental tracks* $f_\phi(t)$ (azimuthal) and $f_r(t)$ (radial). Discretized on the SFT grid these are two $(f,\dot f)$ sequences; by continuity and slow evolution each is anchored by its first-bin value and propagated,

$$
f^{(\alpha+1)} \approx f^{(\alpha)} + \dot f^{(\alpha)}\,T_{\mathrm{sft}},\qquad \dot f^{(\alpha)} > 0 .
$$

**Multi-harmonic statistic.** Summing the single-harmonic statistic over the harmonic set $\mathcal H = \{(m,n):\,0\le m\le 10,\ |n|\le 50\}$ (matching the Kerr eccentric equatorial mode content) gives the proposed detection statistic

$$
\boxed{\;\Lambda_{\mathrm{MH}}\big(f_\phi,f_r\big) = \sum_{(m,n)\in\mathcal H} \Lambda\!\Big(\big\{\,m f_\phi^{(\alpha)} + n f_r^{(\alpha)},\ m\dot f_\phi^{(\alpha)} + n\dot f_r^{(\alpha)}\,\big\}\Big)\;}
$$

which depends only on the two fundamental frequency tracks, is maximized over all per-harmonic amplitudes and phases, and reduces to the single-harmonic statistic when $\mathcal H=\{(2,0)\}$. The search maximizes $\Lambda_{\mathrm{MH}}$ over the first-bin anchors $(f_\phi^{(0)},\dot f_\phi^{(0)},f_r^{(0)},\dot f_r^{(0)})$; positivity ($\dot f>0$) and continuity restrict the candidate proposals at each subsequent bin.

**Differentiability and cost.** Each $\Lambda$ is differentiable in $(f_\alpha,\dot f_\alpha)$ through the custom-VJP Fresnel kernel, so $\Lambda_{\mathrm{MH}}$ is differentiable in the four anchors and is compatible with `vmap` over $\mathcal H$ and over proposals. The SFT $\tilde d_{j,\alpha}$ is computed **once**; only the inexpensive integer frequency combinations $m f_\phi + n f_r$ vary across the $|\mathcal H|$ harmonics, which is what makes the statistic cheap under `jit`+`vmap`.

### Implementation
The implementation of this mult-harmonic search statistic should be a JAX function that takes as input the SFT data and a set of proposed f, fdot pairs for the first time bin. It should then integrate along the next two tracks and return the maximum statistic value. We can use JAX's vmap and jit to optimize the performance of this function. Note that we should only compute the SFT of the full data once and store this. We restrict the function now to integer combinations up to azimuthal<= 10, radial <=50. This is equal to the parameter space coverage for the Kerr eccentric equatoral waveform model. Using JIT this function should be very fast to compute, even with the large summation. Also, we only rely on frequency evolution and not on the amplitudes, which are more expensive to compute. The statistic should be end-to-end differentiable and compatible with vmap, so that we can use it in a gradient-based sampler like parismc.

### Relation to the Viterbi algorithm

The semi-coherent statistic above is, structurally, a *Viterbi-type* track finder with the dynamic-programming step suppressed. As currently formulated, each track is propagated deterministically from its first-bin anchor, $f^{(\alpha+1)}\approx f^{(\alpha)}+\dot f^{(\alpha)}T_{\mathrm{sft}}$, and the search maximizes over the anchors $(f_\phi^{(0)},\dot f_\phi^{(0)},f_r^{(0)},\dot f_r^{(0)})$. The Viterbi step is what one adds to let each track bend segment-by-segment under a local smoothness constraint, instead of being locked to a constant-$\dot f$ extrapolation that accumulates track error over a long inspiral. This section makes the connection more clear and sketches a multi-voice-constrained Viterbi tracker.

**Properties of Viterbi structure.** Three properties of $\Lambda$ are exactly the preconditions a Viterbi / hidden-Markov-model (HMM) tracker requires:

1. *Additivity over segments.* $\Lambda=\sum_\alpha A_\alpha\,2|c_\alpha|^2/h_\alpha$ is a sum of per-segment, per-state terms. This is the only structural requirement for the Bellman recursion to apply.
2. *Semi-coherence is key.* Maximizing over the per-segment amplitude $A$ and phase $\phi$ independently removes all cross-segment phase bookkeeping and leaves a Markov-decomposable objective. A fully coherent matched filter does **not** factor into a Viterbi sum — the phase ties all segments together. Semi-coherence is therefore not merely a cost saving here; it is the enabling assumption for any dynamic-programming track finder.
3. *A local transition rule already exists.* The continuity and positivity constraints ("the next $f,\dot f$ pair must be close to the previous one", $\dot f>0$) are precisely the HMM transition support.

This is the same structure exploited in continuous-wave (CW) searches: for example the SOAP pipeline (Bayley, Messenger & Woan) run Viterbi over an SFT time–frequency map with emission = per-segment power / $\mathcal F$-statistic and transition = "frequency may hop at most a few bins per segment."

**Single-track Viterbi.** Discretize the per-segment state as $s_\alpha=(f,\dot f)$. The Bellman recursion is

$$
V_\alpha(s) = E_\alpha(s) \;+\; \max_{s'\,\to\, s\ \text{allowed}} V_{\alpha-1}(s'),
\qquad
E_\alpha(s) = \frac{2\,|c_\alpha(s)|^2}{h_\alpha},
$$

where the transition support enforces $f = f' + \dot f'\,T_{\mathrm{sft}}$, a jerk bound $|\dot f-\dot f'|\le\delta$, and $\dot f>0$. Backtracking the argmax recovers the optimal track at cost $\mathcal O(N\cdot|\mathcal S|\cdot|\text{transitions}|)$ rather than exponential in $N$. This already improves on a single global anchor with constant-$\dot f$ propagation, because Viterbi lets $\dot f$ drift to follow the true curved track while penalizing implausible jumps.

**Multi-voice-constrained Viterbi.** The feature specific to the EMRI decomposition is that the emission is not a single time–frequency cell but the whole comb, coupling two fundamentals:

$$
\textbf{state}\quad s_\alpha = \big(f_\phi,\dot f_\phi,\;f_r,\dot f_r\big),
\qquad
E_\alpha(s_\alpha)=\!\!\sum_{(m,n)\in\mathcal H}\!\! A_\alpha^{(mn)}\,\frac{2\,\big|c_\alpha\!\big(m f_\phi+n f_r,\;m\dot f_\phi+n\dot f_r\big)\big|^2}{h_\alpha}.
$$

This is a *factorial / coupled HMM*: two latent tracks with a single joint emission. One cannot run two independent Viterbis on $f_\phi$ and $f_r$, because the score at each segment depends on the joint state through the integer combinations $m f_\phi + n f_r$. The transitions, however, factor per fundamental (each obeys its own continuity + positivity), which keeps the recursion clean:

$$
V_\alpha(s)=E_\alpha(s)+\max_{\substack{s'_\phi\to s_\phi\\ s'_r\to s_r}}V_{\alpha-1}(s').
$$

This is precisely "Viterbi constrained by the multi-voice decomposition." Tying every voice $(m,n)$ to integer combinations of two fundamentals is what collapses an otherwise intractable multi-track search — each harmonic wandering independently, with an enormous path space and large noise pickup — into a four-dimensional latent state. One keeps Viterbi's flexibility (curved tracks, robustness to waveform-model error) while the comb constraint controls the look-elsewhere / noise penalty that flexibility normally costs.

**Making the 4D state tractable.** The naive joint state space $|\mathcal S_\phi|\times|\mathcal S_r|$ is the cost concern. Three reductions, in decreasing order of impact:

-> I don't think this dimensionality will be that big of a problem. Even when discretizing each dimension into 1e4 bins, we get 1e16 states which is huge but we can expect very fast and batched evaluation of the search statistic. Suppose one evaluation takes 1ms then we can do 1e3 per second serially. When batching over 1e6 states we can do 1e9 per second which is much more manageable. Also, the Viterbi step will reduce the number of states we need to consider at each step by only keeping the top states that are close to the previous ones.

> **Reply:** Agreed, and stronger than that — with Viterbi the full $10^{16}$ product grid is *never materialized as a compute space*, only as an indexing space. At each segment you evaluate the emission only for the $B$ surviving beam candidates times their local transition fan-out. Continuity makes $f$ nearly determined ($f\approx f'+\dot f'\,T_{\mathrm{sft}}$) and lets $\dot f$ move only within a jerk band, so the fan-out per track is a handful of states; squared over the two coupled tracks it is still $\sim\mathcal O(10^2)$. Net per-segment work is $\sim B\times(\text{fan-out})\times|\mathcal H|\times(2P{+}1)$ kernel evaluations. The number to fold into your throughput estimate is $|\mathcal H|\approx 11\times101\approx1100$: that comb sum sits *inside* every single emission evaluation, so the effective "1ms" should already include it. The brute-force grid only bites at the very first segment, where you have no predecessor to prune against and must seed anchors across the prior box.

- *Physical coupling of the two fundamentals (the dominant lever).* For an adiabatic EMRI, $f_\phi(t)$ and $f_r(t)$ are not independent: both are driven by the same radiation-reaction evolution of $\sim$two intrinsic quantities (e.g. $(p,e)$). The physically allowed $(f_\phi,f_r,\dot f_\phi,\dot f_r)$ states therefore lie near a low-dimensional manifold rather than the full 4D grid. Parametrizing the state by position along a precomputed inspiral track — e.g. interpolating $\dot f_\phi,\dot f_r$ as functions of $(f_r,e)$ from the FEW trajectory module — turns the transition support into "advance along the inspiral manifold" and shrinks the effective state to $\sim$2D. This is also the natural bridge back to the true EMRI parameters in the follow-up stage.
    - I am fairly confident that the identification of the tracks back into physical parameters will not prove to be the issue. We will have batched and differentiable trajectories available which can be used to solve the inverse problem in a Bayesian way. 
        - > **Reply:** Agreed. With differentiable batched FEW trajectories, inverting the recovered $\big(f_\phi(t),f_r(t)\big)$ tracks to intrinsic parameters $(M,\mu,a,p_0,e_0,\dots)$ is a smooth least-squares / nested-sampling fit with the Viterbi track as data — cheap relative to the search. The only thing to keep an eye on is *voice labelling*: which $(m,n)$ each recovered ridge corresponds to. But the integer-comb constraint largely fixes this, since the spacing and slopes of the ridges already encode $m$ and $n$.
- *Beam pruning.* Keep only the top-$B$ states per segment. This is the "tracks that deviate average out and get dropped" intuition made into an explicit beam-search Viterbi, bounding cost to $\mathcal O(N\cdot B\cdot|\text{transitions}|)$.
    - This is exactly what we want to do: keep only the top states that are close to the previous ones. This is a standard technique in Viterbi implementations and should work well here given the continuity constraints. We can experiment with different beam widths to find a good balance between sensitivity and computational cost. The sorting should happen at fixed points, with e.g. one month of data buildup per segment, so that we can keep track of the best candidates as we go along. 
        - > **Reply:** Periodic ("block") pruning every $\sim$1 month rather than every segment is the right instinct: it matches the timescale over which a true track accumulates enough SNR to separate from noise, so you don't cull the real path on a single unlucky segment. The tradeoff to keep in mind is symmetric — prune too rarely and you carry a large candidate set between sort points (memory and per-step cost grow); prune too often and you risk killing a track that momentarily dips below the beam and would have recovered. A month is a sensible middle. Two knobs to expose: the beam width $B$ and the prune cadence; $B$ can be set from a target false-dismissal probability for the true path given the per-block SNR.
- *Hierarchical localization.* Run cheap single-track Viterbi on the two or three strongest expected voices to localize $(f_\phi,f_r)$ at bin 0, then perform the joint multi-voice Viterbi only in that neighborhood.
    - This is a good idea to further reduce tha state space, but may be less efficient for the eccentric systems we aim to target. The single track will just not carry enough SNR to localize in the first place and we may end up having to run the single-voice Viterbi on a large grid anyway. We can experiment with this idea but I would not make it a priority at this stage.
        - > **Reply:** Agreed, deprioritize. Your reasoning is exactly right: for eccentric systems the power is spread across many voices, so no single voice carries enough SNR to localize — the premise of single-track localization fails precisely in the regime we care about. If we ever want a cheap front-end, the natural middle ground is to localize with a *small harmonic sum* (the handful of strongest expected $(m,n)$, e.g. $(2,0),(3,1),(4,2)\dots$) rather than one voice: cheaper than the full $\sim1100$-term comb, but with enough combined SNR to seed the joint search. Not a priority now.

**JAX implementation notes.** The forward pass is a `jax.lax.scan` over the $N$ segments, carrying $V_\alpha$ over the state grid; the emission reuses the existing differentiable Fresnel-kernel $c_\alpha$, `vmap`ped over $(m,n)\in\mathcal H$ and over states. Backtracking requires stored backpointers (argmax indices) per segment, pushed onto the scan output stack and recovered by a reverse scan. Backtracking is a discrete argmax and is therefore *not* differentiable; for gradient-based use (parismc) one differentiates the **soft** version, replacing $\max$ with `logsumexp` / a temperature-softmax (the standard soft-Viterbi / forward algorithm), which yields a smooth $\Lambda_{\mathrm{MH}}$ over the anchors with track-bending built in. The SFT-once-store-once design is unchanged — Viterbi only alters how the per-segment statistics are aggregated, not how they are computed.

**Caveats and trade-offs.**

- *Sensitivity vs. robustness.* The current parametric statistic uses a strong prior (the whole track fixed by four anchors → few degrees of freedom → less noise pickup) and is *more* sensitive when the constant-$\dot f$-per-segment model holds over the full observation. Viterbi spends degrees of freedom to bend the track: more robust to model error and long-baseline curvature, but a longer effective path picks up more noise (more trials). The multi-voice constraint is precisely what claws that sensitivity back. 
    - The constant $\dot f$ per segment is a good approximation for segments of not too long a length, $5e4$ seconds proved to work relatively well for the single harmonic statistic. Until the last segments of the inspiral, the approximation continues to hold. Longer segments would increase the SNR per segment but also increase the model error and therefore the noise pickup. 
        - > **Reply:** Consistent. With $T_{\mathrm{sft}}\approx5\times10^4$ s over a multi-year $T_{\mathrm{obs}}$, $N\approx\text{few}\times10^3$ segments, and Viterbi cost is *linear* in $N$, so the path length is comfortable. The slow-chirp ceiling $|\ddot f|\,T_{\mathrm{sft}}^3/6\lesssim1$ is exactly what caps the segment length; near plunge $\ddot f$ grows and the final segments break it, which is already handled by the $A_\alpha$ mask. One subtlety the bending buys you: even when constant-$\dot f$ holds *within* a segment, the per-segment $\dot f$ value itself drifts *across* segments, and that cross-segment drift is what the deterministic $f^{(0)}+\dot f^{(0)}\alpha T$ propagation gets wrong over $10^3$ segments — Viterbi tracks it without lengthening the segments.
- *Semi-coherence is required.* The clean factorization relies on the incoherent per-segment max over amplitude and phase. Partial coherence across adjacent segments breaks the Viterbi decomposition and would require a two-level coherent-then-incoherent scheme (coherent within blocks, Viterbi across blocks). 
    - We want to keep the semi-coherent structure: this is what makes the search tractable and robust to waveform modelling error. The Viterbi step is an add-on that adds track flexibility without changing the underlying per-segment statistic or the semi-coherent sum.
        - > **Reply:** Exactly — Viterbi here is purely an *aggregation-layer* swap: it changes how the per-segment $c_\alpha$ are combined *across* segments (deterministic propagation $\to$ smoothness-constrained DP), and leaves the per-segment matched filter and the incoherent $\sum_\alpha$ untouched. So everything that makes the current statistic robust survives unchanged.
- *Discretization.* Viterbi wants a discrete state grid, whereas the statistic is differentiable in $\dot f$. Either grid finely (cost) or use soft-Viterbi plus gradient refinement to recover continuous anchors.
    - I don't understand this statement. Elaborate on this caveat. 
        - > **Reply (elaboration).** There are two separate issues bundled here; let me unbundle them.
        - > **(1) Viterbi forces a discrete state lattice, but $\Lambda$ lives on a continuum.** The Bellman recursion $V_\alpha(s)=E_\alpha(s)+\max_{s'\to s}V_{\alpha-1}(s')$ requires a *finite, enumerable* set of states $s$ per segment, because both the $\max$ over predecessors and the integer backpointers index into a fixed array. So you must lay $(f,\dot f)$ — or the joint $(f_\phi,\dot f_\phi,f_r,\dot f_r)$ — on a grid. But the true track sits at continuous values, and the emission $E_\alpha$ is a smooth function that *falls off* as you move the evaluation point away from the truth. If the grid is too coarse, the nearest node is offset from the true $(f,\dot f)$ and you lose match — a pure *discretization SNR loss*, unrelated to noise. So the spacing isn't free to choose: it's set by how fast $\Lambda$ decorrelates off-track, i.e. the template-bank *metric*. The natural scales are $\delta f\sim1/T_{\mathrm{sft}}$ (one SFT bin) and $\delta\dot f\sim1/T_{\mathrm{sft}}^2$ (the $\dot f$ resolution within a segment). Grid at roughly these spacings for, say, $\le3\%$ mismatch; finer is wasted cost, coarser bleeds SNR.
        - > **(2) The hard $\max$ + argmax backtracking destroys differentiability.** Once you snap to a grid and take a *hard* max and discrete argmax, the recovered track and its $\Lambda$ are *piecewise-constant* in any continuous quantity (the data, a hyperparameter, the grid origin): the gradient is zero almost everywhere and undefined at the jumps. So the discrete Viterbi output cannot be fed to a gradient-based sampler like parismc — there is nothing to differentiate. This is the tension with the "end-to-end differentiable" goal stated in the Implementation section.
        - > **Two ways out, and they serve different stages:**
        - >   - **(a) Keep Viterbi discrete, use it only for detection/localization.** Run hard Viterbi to find the best path and its neighbourhood (you want the *path*, not a gradient, at this stage). Then hand the recovered anchors to a *separate* gradient-based refinement that optimizes the continuous $(f_\phi^{(0)},\dot f_\phi^{(0)},f_r^{(0)},\dot f_r^{(0)})$ within that basin using the smooth $\Lambda_{\mathrm{MH}}$ directly. Two-stage: Viterbi finds the basin, gradients polish inside it. Note that once localized you often don't even need Viterbi for the posterior stage — the track is pinned, so the smooth parametric $\Lambda_{\mathrm{MH}}$ over four anchors is enough for parismc.
        - >   - **(b) Soft-Viterbi for end-to-end gradients.** Replace $\max$ with $\frac1\tau\log\sum\exp(\tau\,\cdot)$ (`logsumexp` at temperature $\tau$). Now $V_\alpha$ is smooth and you can backpropagate through the whole forward pass. Caveat: softening makes the *selection* differentiable but the grid is *still discrete* — you still pay for a fine lattice, and you've added a temperature to tune ($\tau\to\infty$ recovers hard Viterbi; small $\tau$ over-smooths and blends neighbouring tracks). "Gradient refinement" then means using the soft-Viterbi gradient to nudge continuous quantities — e.g. shift the grid origin or refine an anchor — *between* discrete passes, so the lattice tracks the signal instead of the signal having to land on the lattice.
        - > **Bottom line:** for the *search/detection* stage, discrete hard Viterbi is the right tool and differentiability is irrelevant — go coarse-but-metric-matched and cheap. For the *parismc posterior* stage you want smoothness, but by then the track is localized and you can use the parametric $\Lambda_{\mathrm{MH}}$ (no Viterbi) or soft-Viterbi if you genuinely want the bending to remain part of the differentiable model. The caveat is really "don't expect one object to be both the discrete track-finder and the differentiable likelihood — split the roles."
            - response: that makes sense. My preferred approach is to implement the hard Viterbi first combined with the multi-harmonic search statistic and then pass the best guess on to a gradient based sampler to refine the parameters. We keep the soft Viterbi as a backup option. The main goal here is to provide an estimate of the tracks that we can pass on to the next stage, which is the inverse problem. For this inverse problem I have parismc and fewtrax ready and so we can map back to good estimates of the EMRI parameters in a Bayesian way. The resulting posteriors from that stage will then be used to seed a full PE run with the full waveform model in the likelihood, which we will write using the WDM basis. This is a longer-term project and I am now trying to assemble the different pieces separately for that. 
