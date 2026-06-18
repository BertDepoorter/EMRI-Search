import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.special as jsp
from scipy.signal.windows import tukey
from few.utils.constants import YRSID_SI
from few.trajectory.inspiral import EMRIInspiral
from emrisearch.da_utils import psd, compute_sfts, generate_noise, get_snr
from emrisearch.emri_utils import create_signal, get_f_fdot_fddot_back
from emrisearch.track_optimizer import get_default_mode_candidates

traj = EMRIInspiral(func='KerrEccEqFlux')
m1,m2,a,p0,e0,x0 = 1e6,10.0,0.99,7.0,0.7,1.0
t,p,e,xI,Pp,Pt,Pr = traj(m1,m2,a,p0,e0,x0,T=2.0)
Tpl=t[-1]/YRSID_SI; ef=e[-1]
true_values=np.array([m1,m2,a,Tpl,ef,1.0])

T_data, T_sft, deltaT, snr_ref = 0.2, 5e4, 5.0, 30.0
tt, hp, hx, pdct = create_signal(true_values, T=T_data, deltaT=deltaT, return_params=True, randomize=True)
snr_sig = get_snr(hp, deltaT)
mask = tt < YRSID_SI*T_data
hp = snr_ref/snr_sig*hp[mask]
samples_per_sft = int(T_sft/deltaT)
wind = tukey(samples_per_sft, 0.2)
signal_sfts = compute_sfts(hp, deltaT, wind, samples_per_sft)
noise_sfts = compute_sfts(generate_noise(hp.shape[0], deltaT, psd), deltaT, wind, samples_per_sft)
data_sfts = jnp.asarray(signal_sfts + noise_sfts)
N_freq, N_seg = data_sfts.shape
freq = np.fft.rfftfreq(samples_per_sft, deltaT)
deltaf = 1.0/T_sft
t_alpha = np.arange(N_seg)*T_sft
print(f"N_freq={N_freq} N_seg={N_seg} deltaf={deltaf:.2e}")

_phi,_f,_dotf,_ddf = get_f_fdot_fddot_back(true_values, t_alpha)
f_phi_true=np.asarray(_f[0]); f_r_true=np.asarray(_f[1])
df_phi_true=np.asarray(_dotf[0]); df_r_true=np.asarray(_dotf[1])

modes = np.array(get_default_mode_candidates(max_m=4,max_n=2),dtype=float)
m_arr,n_arr = modes[:,0],modes[:,1]
H=len(modes)

def fresnel_kernel(f0,f1,T):
    f0=jnp.asarray(f0); f1=jnp.asarray(f1)
    r=jnp.abs(f1)+1e-30
    def pos(o):
        u1=jnp.sqrt(2.0/r)*o; u2=jnp.sqrt(2.0*r)*T+u1
        S1,C1=jsp.fresnel(u1); S2,C2=jsp.fresnel(u2)
        pref=jnp.exp(-1j*jnp.pi*o**2/r)/jnp.sqrt(2.0*r)
        return pref*((C2-C1)+1j*(S2-S1))
    val=jnp.where(f1>=0,pos(f0),jnp.conj(pos(-f0)))
    dirich=T*jnp.sinc(f0*T)*jnp.exp(1j*jnp.pi*f0*T)
    return jnp.where(jnp.abs(f1)<1e-12,dirich,val)

psd_f=jnp.asarray(np.asarray(psd(freq)))
d_white=data_sfts/jnp.sqrt(psd_f)[:,None]
Sigma=jnp.median(jnp.abs(d_white)**2,axis=0)

# ---- emission along the TRUE track (per-segment true state) ----
def emission_along(fphi,dfphi,fr,dfr):
    """Return per-segment Fresnel statistic for a track given as length-N_seg arrays."""
    out=np.zeros(N_seg)
    for al in range(N_seg):
        if fphi[al]<=0:  # out of integration range -> skip
            continue
        f_mn = fphi[al]*m_arr + fr[al]*n_arr            # (H,)
        df_mn= dfphi[al]*m_arr + dfr[al]*n_arr
        Kwin=min(int(np.ceil(np.abs(df_mn).max()*T_sft**2))+6,32)
        dk=np.arange(-Kwin,Kwin+1); W=dk.size
        k0=np.round(f_mn/deltaf).astype(int)
        bins=k0[:,None]+dk[None,:]                       # (H,W)
        f0=f_mn[:,None]-bins*deltaf
        hvalid=(f_mn>5e-4)
        wvalid=(bins>=0)&(bins<N_freq)
        bins_clip=np.clip(bins,0,N_freq-1)
        Fre=fresnel_kernel(jnp.asarray(f0),jnp.asarray(df_mn)[:,None],T_sft)
        Fre=Fre*jnp.asarray(wvalid.astype(float))
        sumFre2=jnp.sum(jnp.abs(Fre)**2,axis=-1)
        dWg=d_white[:,al][jnp.asarray(bins_clip)]
        C=jnp.sum(jnp.conj(dWg)*Fre,axis=-1)             # NO extra deltaf
        rho=jnp.abs(C)**2/(Sigma[al]*sumFre2+1e-30)
        out[al]=float(jnp.sum(rho*jnp.asarray(hvalid.astype(float))))
    return out

true_emi = emission_along(f_phi_true,df_phi_true,f_r_true,df_r_true)
# off-track: shift f_phi up by 30 bins (away from any harmonic)
off_emi  = emission_along(f_phi_true+30*deltaf, df_phi_true, f_r_true+30*deltaf, df_r_true)
inb=f_phi_true>0
print(f"TRUE track  : per-seg emission mean={np.mean(true_emi[inb]):.3f}  sum={true_emi.sum():.1f}")
print(f"OFF  track  : per-seg emission mean={np.mean(off_emi[inb]):.3f}  sum={off_emi.sum():.1f}")
print(f"contrast (true/off) = {true_emi.sum()/max(off_emi.sum(),1e-9):.2f}")
np.savez("diag_truetrack.npz", true_emi=true_emi, off_emi=off_emi,
         f_phi_true=f_phi_true, f_r_true=f_r_true, t_alpha=t_alpha)
print("saved diag_truetrack.npz")

