import numpy as np
import time

adc_offset = -127.5
offset = 44
filename = "SDRSharp_20190521_191501Z_137500000Hz_IQ.wav"

data = np.memmap(filename, offset=offset)

time_a = time.time()
iq = []
for i in range(0, len(data), 2):
    iq.append(data[i+0] + 1j*data[i+1])
time_b = time.time()
print(time_b - time_a)
del iq



time_a = time.time()
window = 2048000

for slice in range(0, int(len(data) // (window * 2)) * window * 2, window * 2):
    data_slice = adc_offset + (data[slice: slice + window * 2: 2]) + 1j * (adc_offset + data[slice + 1: slice + window * 2: 2])
    print(slice, len(data), slice / (len(data)), len(data_slice))

time_b = time.time()
print(time_b - time_a)