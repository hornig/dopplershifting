import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
from scipy import signal
from scipy.fftpack import fft, fftshift, ifft

offset = 44
file1 = "../station1_yagi_SDRSharp_20170312_060959Z_137650kHz_IQ.wav"
file2 = "../station2_turnstile_SDRSharp_20170312_061008Z_137650000Hz_IQ.wav"
file3 = "../SDRSharp_20190521_191501Z_137500000Hz_IQ.wav" #big file
data = np.memmap(file1, offset=offset)

adc_offset = -127.5
window = 2048000
fs = 2048000
T = 1/fs
total_duration = T*len(data)
chunk = 2048000
num_chunks = int(len(data)/(chunk*2))
iterate = 0
specx = []

def calFFT(sig):
    norm_fft = fftshift(fft(sig))
    abs_fft = np.abs(norm_fft)
    return abs_fft

def calFFTPower(afft, fs):
    transform = 10 * np.log10(afft)
    fc = 0
    freq = np.arange((-1 * fs) / 2 + fc, fs / 2 + fc)
    return transform, freq

time_a = time.time()
for slice in range(0, int(len(data) // (window * 2)) * window * 2, window*2): 
    data_slice = adc_offset + (data[slice: slice + window * 2: 2]) + 1j * (adc_offset + data[slice + 1: slice + window * 2: 2])

    fft_iq = calFFT(data_slice)
    
    transform, freq = calFFTPower(fft_iq, fs)
    
    if slice==0:
        #pre storing array in mem. 
        specx = np.zeros([num_chunks, 2048000])

    # specx[:,iterate] = transform 
    specx[num_chunks-iterate-1] = transform

    iterate +=1
    print(len(fft_iq), len(transform), len(specx), iterate)
    del data_slice,transform,fft_iq   
   
del data
time_b = time.time()
print('Time:',time_b - time_a)
print('Len',len(specx), ' Type:', type(specx))
print('Frequency Resolution: {} Hz'.format(fs/(2*fs)))
#np.save('iq_save.npy', iq)

#Plotting Waterfall
plt.figure(figsize=(20,8))
plt.imshow(specx, extent=[(-1 * fs) / 2 + 0, (fs / 2) + 0, 0, total_duration/2], origin='lower', aspect='auto')
plt.colorbar()
plt.xlabel('Frequency (kHz)')
plt.ylabel('Time (s)')
plt.title('Waterfall - Fs=2048kHz : NOAA Sat Pass')
plt.show()

#Select Chanenl:
fc = 1.05e6 
BW = 16e3
start_band = fc - (BW/2)
stop_band = fc + (BW/2)

plt.figure(figsize=(20,8))
plt.imshow(specx, origin='lower', aspect='auto')
plt.colorbar()
plt.xlabel('Frequency (kHz)')
plt.ylabel('Time (s)')
plt.xlim([start_band, stop_band])
plt.title('Waterfall - Fs=2048kHz Fc = {}Hz : NOAA Sat Pass'.format(fc))
plt.show()