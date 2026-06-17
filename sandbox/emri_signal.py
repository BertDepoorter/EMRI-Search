"""code to generate a dictionary of EMRI signal + SFTs for testing the multi-voice Viterbi search."""

from few.utils.constants import YRSID_SI

# use functions from emrisearch package
from emrisearch.da_utils import psd, sft_inner_product, compute_sfts, generate_noise, get_snr
from emrisearch.emri_utils import create_signal, get_f_fdot_fddot_back   
from scipy.signal.windows import tukey

def generate_emri_signal_and_sfts(true_values, T_data, T_sft, deltaT, snr_ref, T_snr):
    """
    Generate EMRI signal, compute SFTs, and prepare data for analysis.
    
    Parameters:
    -----------
    true_values : np.ndarray
        EMRI parameters [m1, m2, a, Tpl, ef, x0]
    T_data : float
        Total data duration in years
    T_sft : float
        SFT duration in seconds
    deltaT : float
        Time step in seconds
    snr_ref : float
        SNR of the signal with duration T_snr
    T_snr : float
        Duration of the signal in years it must be always bigger or equal to T_data

    Returns:
    --------
    dict : Dictionary containing:
        - 't': trimmed time array
        - 'signal_sfts': SFTs of the signal
        - 'noise_sfts': SFTs of noise realization
        - 'new_noise_sfts': SFTs of second noise realization
        - 'data_sfts': SFTs of signal + noise
        - 't_alpha': time samples for SFTs
        - 'samples_per_sft': number of samples per SFT
        - 'wind': window function
        - 'snr_final': final SNR of scaled signal
        - 'true_phi_f_fdot_fddot': frequency evolution parameters
    """
    # Generate EMRI signal
    t, hp, hx, param_dict = create_signal(true_values, T=T_snr, deltaT=deltaT, return_params=True, randomize=True)
    snr_sig = get_snr(hp, deltaT)  # Check SNR of the signal
    
    # Trim signal to desired duration
    mask = t < YRSID_SI * T_data
    hp, hx = snr_ref/snr_sig * hp[mask], snr_ref/snr_sig * hx[mask]  # Rescale signal to desired SNR
    t = t[mask]
    snr_final = get_snr(hp, deltaT)  
    print(f"Final SNR: {snr_final}")
    
    # update distance
    param_dict['dist'] = param_dict['dist'] * snr_sig/snr_ref
    
    # SFT parameters
    samples_per_sft = int(T_sft/deltaT) 
    wind = tukey(samples_per_sft, 0.2)  # Window function 
    
    # Compute SFTs
    signal_sfts = compute_sfts(hp, deltaT, wind, samples_per_sft)
    num_sfts = signal_sfts.shape[1]
    t_alpha = np.arange(num_sfts) * T_sft
    
    # Generate noise SFTs
    noise_td = generate_noise(hp.shape[0], deltaT, psd)
    data_td = hp + noise_td
    noise_sfts = compute_sfts(noise_td, deltaT, wind, samples_per_sft)
    new_noise_sfts = compute_sfts(generate_noise(hp.shape[0], deltaT, psd), deltaT, wind, samples_per_sft)
    data_sfts = signal_sfts + noise_sfts
    
    # compute matched filtering
    freq = np.fft.rfftfreq(samples_per_sft, deltaT)
    m_hd = sft_inner_product(signal_sfts, data_sfts, freq)
    m_hh = sft_inner_product(signal_sfts, signal_sfts, freq)
    m_hn = sft_inner_product(signal_sfts, noise_sfts, freq)

    # Get frequency evolution for true values
    true_phi_f_fdot_fddot = get_f_fdot_fddot_back(true_values, t_alpha)
    
    return {
        'data_td': data_td,
        'param_dict': param_dict,
        't': t,
        'data_sfts': data_sfts,
        'signal_sfts': signal_sfts,
        'noise_sfts': noise_sfts,
        'new_noise_sfts': new_noise_sfts,
        't_alpha': t_alpha,
        'samples_per_sft': samples_per_sft,
        'num_sfts': num_sfts,
        'wind': wind,
        'snr_final': snr_final,
        'true_phi_f_fdot_fddot': true_phi_f_fdot_fddot,
        'm_hd': m_hd,
        'm_hh': m_hh,
        'm_hn': m_hn
    }


true_values = [1e6, 10.0, 0.9, 1.5, 0.1, 1.0]
T_data = 2.0
T_sft = 5e4
deltaT = 5.0
snr_ref = 40
T_snr = 2.0
injection = generate_emri_signal_and_sfts(
    true_values, T_data, T_sft, deltaT, snr_ref, T_snr
)
# check content of injection dictionary
breakpoint()


# Plot the SFts

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.title("Signal SFTs (magnitude)")
plt.imshow(10 * np.log10(np.abs(injection['signal_sfts'])), aspect='auto', origin='lower')
plt.colorbar(label='Magnitude (dB)')            
plt.subplot(2, 1, 2)
plt.title("Data SFTs (magnitude)")
plt.imshow(10 * np.log10(np.abs(injection['data_sfts'])), aspect='auto', origin='lower')
plt.colorbar(label='Magnitude (dB)')
plt.tight_layout()
plt.show()