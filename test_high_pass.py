
import numpy as np
import scipy
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import heapq
#from demodulation import get_filtered_signal
#from butter_filters import get_butter_high_pass_filtered_signal
from cheby_filters import get_cheby_high_pass_filtered_signal
from demodulation import get_fft

def y1(sr=2000, freq=10):
    t1 = np.linspace(0,1,sr,endpoint=False)
    x1 = 3*np.sin(2*np.pi*freq*t1)
    return t1, x1

def y2(sr=2000, freq=40):
    t2 = np.linspace(0,1,sr, endpoint=False)
    x2 = np.sin(2*np.pi*freq*t2)
    return t2, x2

def y3(sr=2000, freq=70):
    t3 = np.linspace(0,1,sr, endpoint=False)
    x3 = 0.5*np.sin(2*np.pi*freq*t3)
    return t3, x3

t1, x1 = y1()
t2, x2 = y2()
t3, x3 = y3()

sr = 2000
high_cut = 50
x = x1 + x2 + x3
filtere_signal = get_cheby_high_pass_filtered_signal(x, high_cut, sr, order=10)
#
_, freqs_filtered, amps_filtered = get_fft(filtere_signal)
_, freqs, amps = get_fft(x)
plt.figure(figsize = (20, 6))
plt.subplot(131)
plt.plot(t1, x, 'b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original signal')

plt.subplot(132)
plt.stem(freqs_filtered, amps_filtered, 'b', \
            markerfmt=" ", basefmt="-b")
# plt.stem(freqs_filtered, amps_filtered, 'r', \
#             markerfmt=" ", basefmt="-r")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |signal_fft(freq)|')
plt.title(f'FFT of  filtered signal. Freq > {high_cut} Hz')
plt.xlim(0, 100)
plt.ylim(0, 1.5)

plt.subplot(133)
plt.plot(t1, filtere_signal, 'r')
plt.ylim(-4, 4)
plt.legend(['filtered signal'])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(f'filtered signal. Freq > {high_cut} Hz')
plt.tight_layout()
plt.show()
