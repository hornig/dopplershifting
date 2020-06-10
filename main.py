import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fftpack import fft, fftshift, ifft

adc_offset = -127.5
offset = 44
file1 = "../station1_yagi_SDRSharp_20170312_060959Z_137650kHz_IQ.wav"
file2 = "../station2_turnstile_SDRSharp_20170312_061008Z_137650000Hz_IQ.wav"
file3 = "../SDRSharp_20190521_191501Z_137500000Hz_IQ.wav" #big file
data = np.memmap(file1, offset=offset)

#Convert data to complex in chunks
time_a = time.time()
window = 2048000
iqdata = []
for slice in range(0, int(len(data) // (window * 2)) * window * 2, window*2):  # the -1 is because the slice could be less than the region we need to cut out for the out. xxx
    data_slice = adc_offset + (data[slice: slice + window * 2: 2]) + 1j * (adc_offset + data[slice + 1: slice + window * 2: 2])
    iqdata.extend(data_slice)
    print(slice, len(data)/2, slice / (len(data)), len(data_slice), len(iqdata))
    
iq = np.asarray(iqdata)
time_b = time.time()
print('Time:',time_b - time_a)
print('Len',len(iq), ' Type:', type(iq))

fs = window
#NFFT = int(fs*0.005)
noverlap = int(fs*0.0025) 

#Spectrogram Plot
plt.figure(figsize=(30,15))
time_a = time.time()
plt.specgram(iq,NFFT=1024, Fs=fs)
plt.xlabel('Time(s)')
plt.ylabel('Frequency(KHz)')
# plt.ylim([2.5e5, 4e5])
plt.title('Spectrogram')
plt.show()
time_b = time.time()
print('Time to Plot:',time_b - time_a)

#SciPy's Spectrogram
from scipy import signal
iqlen = int(len(iq)/10000)
f, t, Sxx = signal.spectrogram(iq[:1000000], fs=window, nfft= 1024)

print(len(f), len(t), len(Sxx))

time_a = time.time()
plt.figure(figsize=(30,15))
plt.pcolormesh(t, f, Sxx)
# plt.ylim([0, 1e5])
plt.ylabel('Frequency [KHz]')
plt.xlabel('Time [sec]')
plt.show()
time_b = time.time()
print('Time to Plot:',time_b - time_a)

#PSD
plt.figure(figsize=(30,15))
plt.psd(iq, 512, window)
plt.xlabel('Time(s)')
plt.ylabel('Frequency(MHz)')
plt.title('PSD')
plt.show()