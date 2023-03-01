
import numpy as np
import scipy
import matplotlib.pyplot as plt
from demodulation import get_fft


sr = 2000
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

freq = 1.
x1 = 3*np.sin(2*np.pi*freq*t)

freq = 4
x2 = np.sin(2*np.pi*freq*t)

freq = 7
x3 = 0.5*np.sin(2*np.pi*freq*t)
x = x1 + x2 + x3
X, freq, amplitude = get_fft(x)
plt.figure(figsize = (12, 6))
plt.subplot(131)
plt.plot(t, x, 'b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original signal')

plt.subplot(132)
plt.stem(freq, amplitude, 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |signal_fft(freq)|')
plt.title('FFT of signal')
plt.xlim(0, 10)

plt.subplot(133)
plt.plot(t, scipy.fftpack.ifft(X), 'r')
#plt.plot(t, x, 'g')
plt.legend(['inverse FFT'])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Inverse FFT')
plt.tight_layout()
plt.show()