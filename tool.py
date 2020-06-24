import numpy as np
import argparse, sys
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
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

    def run(self, filename, save_flag=False, fs=2048000):
        """Loads the data and performs fft in chunks."""

        self.filename = filename
        self.fs = fs
        self.save_flag = save_flag
        self.overlap = 0.5
        offset = 44
        data = np.memmap(filename, offset=offset)
        T = 1/fs
        iterate = 0 
        adc_offset = -127.5
        window = 2048000
        chunk = window
        self.total_duration = T*len(data)
        num_chunks = int(len(data)/(chunk*2))
        
        #Load file if flag set to true
        skip = False
        if (self.save_flag == True):
            try:
                print('Loading Data')
                self.specx = np.load("iq_spec.npy")
                skip=True
            except:
                pass

        if(skip == False):
            data_slice = []

            time_a = time.time()
            for slice in range(0, int(len(data) // (window * 2)) * window * 2, window*2): 
                data_slice = adc_offset + (data[slice: slice + window * 2: 2]) + 1j * (adc_offset + data[slice + 1: slice + window * 2: 2])

                data_slice = data_slice*np.hamming(len(data_slice))
                fft_iq = self.calFFT(data_slice)

                transform, freq = self.calFFTPower(fft_iq, self.fs)

                if slice==0:
                    # first = transform[:int(self.overlap*window)]
                    # M_avg = first
                    #pre storing array in mem. 
                    self.specx = np.zeros([num_chunks, 2048000])
                # else:
                    #Overlapping:
                    # M_avg = M_avg + transform[int(self.overlap*window):]

                self.specx[num_chunks-iterate-1] = transform

                iterate +=1

                sys.stdout.write("\r")
                sys.stdout.write("{} %".format(round(((iterate/num_chunks)*100),2)))
                sys.stdout.flush()
                
                del data_slice,transform,fft_iq   
            
            del data
            time_b = time.time()

            if self.save_flag==True:
                print('Saving data to disk')
                np.save('iq_spec.npy', self.specx)
                print('iq_spec saved.', end=' ')
            print('Time:',time_b - time_a)

        if num_chunks>100:
            self.jump = 3
        else:
            self.jump = 1
        self.specx = self.specx[::self.jump]

    def plot(self):
        """Plots the full waterfall and marks signal fc"""

        plt.figure(figsize=(20,8))
        plt.imshow(self.specx, extent=[(-1 * self.fs) / 2 + 0, (self.fs / 2) + 0, 0, self.total_duration/2], origin='lower', aspect='auto')
        plt.axvline(self.fc - (self.fs/2))
        plt.annotate('Fc',xy=(self.fc - (self.fs/2), 0))
        plt.colorbar()
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Time (s)')
        plt.title('Waterfall - Fs:{}kHz Fc:{}KHz : NOAA Sat Pass'.format(self.fs/1e3, self.fc/1e3))
        plt.suptitle("file:"+ self.filename)
        plt.show()

    def select_channels(self, chanel_center, BW=16e3):
        """Plots multiple channels in one figure while marking signal fc"""

        self.BW = BW
        self.chanel_center = chanel_center
        fig, ax = plt.subplots(ncols=len(self.chanel_center )) 
        fig.suptitle('Waterfall Channel View', fontsize=15)
        # fig.set_size_inches(20,8)
        fig.tight_layout(pad=3)
        start_band = []
        stop_band = []

        for n in range(len(self.chanel_center)):
            start_band.append(self.chanel_center[n] - (self.BW/2))
            stop_band.append(self.chanel_center[n] + (self.BW/2))
            # left_limit = (self.fs/2) + self.chanel_center[n] - (self.BW/2)
            # right_limit = (self.fs/2) - self.chanel_center[n] + (self.BW/2)
            ax[n].imshow(self.specx, origin='lower', aspect='auto')
            # ax[n].axvline(self.chanel_center[n])
            ax[n].axvline(self.fc)
            ax[n].set_xlabel('Frequency (kHz)')
            ax[n].set_ylabel('Time (s)')
            ax[n].set_xlim([start_band[n], stop_band[n]])
            ax[n].set_title('Fs=2048kHz Fc={}KHz, BW = {}kHz'.format(self.fc/1e3, self.BW/1e3))
            ax[n].annotate('Fc',xy=(self.fc, 0))
        plt.show()

    def find_signal(self, threshold, distance, draw=False):
        """Finds the sig peaks with parameters and selects the ones with distance less than 2500(preceding peak-peak)
            Then selects the peak with max magnitude. - center frequency. """
            
        self.threshold = threshold
        self.distance = distance
        self.draw = draw
        fft_vals = self.specx[1]

        peaks, _ = find_peaks(fft_vals, height=self.threshold*np.mean(fft_vals), distance=100)
        peaks = peaks[peaks > int(len(fft_vals)/2)]

        sig_peaks = []
        for n in range(0, len(peaks)):
            if(np.abs(peaks[n] - peaks[n-1]) < self.distance):
                sig_peaks.append(peaks[n-1])
        
        #Select peak with max mag - Center frequency
        for j in range(len(fft_vals[sig_peaks])):
            if (np.max(fft_vals[sig_peaks]) == fft_vals[sig_peaks[j]]):
                self.fc = sig_peaks[j]
        
        # self.fc = np.median(sig_peaks)

        if self.draw==True:
            fig, ax = plt.subplots(2, 1)
            fig.suptitle('Signal Detection')
            ax[0].plot(fft_vals)
            ax[0].plot(sig_peaks, fft_vals[sig_peaks], "x")
            ax[0].set_xlabel('Frequency(Hz)')
            ax[0].set_ylabel('Magnitude (dB)')
            ax[0].set_title('Single Window FFT')
            ax[0].axvline(self.fc)

            ax[1].plot(fft_vals)
            ax[1].plot(sig_peaks, fft_vals[sig_peaks], "X")
            ax[1].set_xlim([sig_peaks[0],sig_peaks[len(sig_peaks)-1]])
            ax[1].set_ylim(bottom=30)
            ax[1].axvline((self.fc))
            ax[1].set_xlabel('Frequency(Hz)')
            ax[1].set_ylabel('Magnitude (dB)')
            ax[1].set_title('Single Window FFT- Cropped')
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
    w.find_signal(1.5, 2500)
    w.plot()
    w.select_channels([w.fc, w.fc+5e3], BW=50e3)    


