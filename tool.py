import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
from scipy import signal
from scipy.fftpack import fft, fftshift, ifft

class Waterfall():
    """Waterfall Tool Main Class"""

    def __init__(self):
        self.specx = []

    def calFFT(self, sig):
        norm_fft = fftshift(fft(sig))
        abs_fft = np.abs(norm_fft)
        return abs_fft

    def calFFTPower(self, afft, fs):
        transform = 10 * np.log10(afft)
        fc = 0
        freq = np.arange((-1 * fs) / 2 + fc, fs / 2 + fc)
        return transform, freq

    def run(self, filename, save_flag, fs=2048000):
        """Loads the data and performs fft in chunks."""

        self.filename = filename
        self.fs = fs
        self.save_flag = save_flag
        offset = 44
        data = np.memmap(filename, offset=offset)

        T = 1/fs
        iterate = 0 
        adc_offset = -127.5
        window = 2048000
        chunk = window
        self.total_duration = T*len(data)
        num_chunks = int(len(data)/(chunk*2))

        try:
            self.specx = np.load("iq_spec.npy")
            skip=True
        except:
            skip=False

        if(skip == False):
            data_slice = []

            time_a = time.time()
            for slice in range(0, int(len(data) // (window * 2)) * window * 2, window*2): 
                data_slice = adc_offset + (data[slice: slice + window * 2: 2]) + 1j * (adc_offset + data[slice + 1: slice + window * 2: 2])

                fft_iq = self.calFFT(data_slice)
                
                transform, freq = self.calFFTPower(fft_iq, self.fs)
                
                if slice==0:
                    #pre storing array in mem. 
                    self.specx = np.zeros([num_chunks, 2048000])

                self.specx[num_chunks-iterate-1] = transform

                iterate +=1
                print(len(fft_iq), len(transform), len(self.specx), iterate)
                del data_slice,transform,fft_iq   
            
            del data
            time_b = time.time()

            if save_flag==True:
                np.save('iq_spec.npy', self.specx)

            print('Time:',time_b - time_a)

        print('Len',len(self.specx), ' Type:', type(self.specx))
        print('Frequency Resolution: {} Hz'.format(fs/(2*fs)))
        print('Time Resolution: {} s'.format((2*fs)/fs))

    def plot(self):
        """Plots the full waterfall"""

        plt.figure(figsize=(20,8))
        plt.imshow(self.specx, extent=[(-1 * self.fs) / 2 + 0, (self.fs / 2) + 0, 0, self.total_duration/2], origin='lower', aspect='auto')
        plt.colorbar()
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Time (s)')
        plt.title('Waterfall - Fs=2048kHz : NOAA Sat Pass')
        plt.show()

    def select_channels(self, fc, BW=16e3):
        """Plots multiple channels in on figure"""

        self.BW = BW
        self.fc = fc
        fig, ax = plt.subplots(ncols=len(fc)) 
        fig.suptitle('Waterfall Channel View', fontsize=15)
        # fig.set_size_inches(20,8)
        fig.tight_layout(pad=3)
        start_band = []
        stop_band = []

        for n in range(len(fc)):
            start_band.append(self.fc[n] - (self.BW/2))
            stop_band.append(self.fc[n] + (self.BW/2))
            # left_limit = (self.fs/2) + self.fc[n] - (self.BW/2)
            # right_limit = (self.fs/2) - self.fc[n] + (self.BW/2)
            ax[n].imshow(self.specx, origin='lower', aspect='auto')
            ax[n].axvline(self.fc[n])
            ax[n].set_xlabel('Frequency (kHz)')
            ax[n].set_ylabel('Time (s)')
            ax[n].set_xlim([start_band[n], stop_band[n]])
            ax[n].set_title('Fs=2048kHz Fc={}Hz, BW = {}kHz'.format(self.fc[n], self.BW/1e3))
            ax[n].annotate('Fc',xy=(self.fc[n], 0))
        plt.show()

    def find_signal(self):
        #Find periodicity of the signal using zero crossing.
        zero_crosses = np.nonzero(np.diff(self.specx > self.specx.shape[0]/2))
        plt.hist(zero_crosses)
        plt.xlabel('Samples')
        plt.ylabel('Zero Crossings')
        plt.show()

def args():
    parser = argparse.ArgumentParser(description='Reading File')
    parser.add_argument('--f', help="Specify path to IQ .wav file", type=str)
    parser.add_argument('--save', help="Set this flag to let program store FFT_iq file", action="store_true")

    return parser.parse_args()

if __name__ == '__main__':
    print("Waterfall Tool")

    args_input = args()

    w = Waterfall()
    w.run(args_input.f,args_input.save)
    # w.find_signal()
    w.plot()
    w.select_channels([1.05e6, 1.32e6], BW=16e3)    


