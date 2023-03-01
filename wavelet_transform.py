"""
A module for generic wavelet transforms.
"""

import pywt
import matplotlib.pyplot as plt
from demodulation import get_envelope_spectrum
from dataset import get_ims_data

def compute_energy(spectrum):
    """
    Compute the energy of the envelope spectrum.
    """
    energy = sum(spectrum**2)
    return energy

def get_discrete_wavelet_coefficients(data, wavelet='db1'):
    """
    Compute the wavelet transform of the data.
    The approximate coefficients represent the low
    frequency components of the signal. The detail
    coefficients represent the high frequency components.

    Parameters
    ----------
    data : The data to transform.
    wavelet : The wavelet to use.

    Returns
    -------
    cA : The approximate coefficients.
    Cd : The detail coefficients.
    """
    cA, Cd = pywt.dwt(data, wavelet)
    return cA, Cd


if __name__=='__main__':
    bearing_number = 0
    experiment_number = 2

    bearing_data = get_ims_data(experiment_number, bearing_number)
    sample_number = 704
    sample_data = bearing_data[sample_number, ...]
    ca, cd = get_discrete_wavelet_coefficients(sample_data)
    # freq, amp = get_envelope_spectrum(sample_data)
    # freq_ca, amp_ca = get_envelope_spectrum(ca, sampling_freq=10000, low_cut=1000, high_cut=4950)
    # freq_cd, amp_cd = get_envelope_spectrum(cd, sampling_freq=10000, low_cut=1000, high_cut=4950)

    # #frequencies, amplitude = get_envelope_spectrum(sample_data)
    # # two subplots
    # # figure size
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(freq_ca, amp_ca, label='Approximate coefficients')
    # plt.plot(freq, amp, label='Original signal')
    # plt.xlim(0, 500)
    # plt.ylim(0, 0.12)
    # plt.legend()
    # plt.title('Approximation coefficients spectrum')
    # plt.subplot(1, 2, 2)
    # plt.plot(freq_cd, amp_cd, label='Detail coefficients')
    # plt.plot(freq, amp, label='Original signal')
    # plt.xlim(0, 500)
    # plt.ylim(0, 0.12)
    # plt.legend()
    # plt.title('Detail coefficients spectrum')
    # plt.show()