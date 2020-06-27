import numpy as np
import argparse, sys, time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt
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
        return transform

    def run(self,filename,  fs=2048000, save_flag=False):
        """Loads the data and performs fft in chunks.
        
        Params:
            filename: str
                name of the iq file in .wav
            
            fs: int
                sampling rate
                defautlt: 2048000

            save_flag: bool
                Enable to the save the fft of iq in .npy
                which will autoload next time.
        """

        self.filename = filename
        self.fs = fs
        self.save_flag = save_flag
        self.overlap = 0.5
        offset = 44
        T = 1/fs
        iterate = 0 
        adc_offset = -127.5
        window = self.fs
        chunk = self.fs
        data = np.memmap(filename, offset=offset)
        self.total_duration = T*(len(data)/2)
        num_chunks = int(len(data)/(chunk*2))
        file_name = 'Spec_'+self.filename.split('.wav')[0]+'.npy'

        #Load file if flag set to true
        skip = False
        if (self.save_flag == True):
            try:
                print('Loading Data')
                self.specx = np.load(file_name)
                skip=True
            except:
                pass

        if(skip == False):
            data_slice = []
            self.specx = np.zeros([num_chunks, 2048000])    

            time_a = time.time()
            for slice in range(0, int(len(data) // (window * 2)) * window * 2, window*2): 
                data_slice = adc_offset + (data[slice: slice + window * 2: 2]) + 1j * (adc_offset + data[slice + 1: slice + window * 2: 2])

                # data_slice = data_slice*np.hamming(len(data_slice))
                fft_iq = self.calFFT(data_slice)

                transform = self.calFFTPower(fft_iq, self.fs)

                self.specx[num_chunks-iterate-1] = transform

                iterate +=1

                sys.stdout.write("\r")
                sys.stdout.write("{} %".format(round(((iterate/num_chunks)*100),2)))
                sys.stdout.flush()
                
                del data_slice,transform,fft_iq   
            
            del data
            time_b = time.time()

            if self.save_flag==True:
                print('\nSaving data to disk')
                np.save(file_name, self.specx)
                print('iq_spec saved.', end=' ')
            print('Time:',round(time_b - time_a),2)

        if num_chunks>100:
            self.jump = 5
        else:
            self.jump = 1
        
        self.time_bins = np.linspace(0, self.total_duration, self.specx.shape[0])
        self.leftlim = (0, self.fs)
        self.rightlim = (0, self.total_duration)

    def plot(self, show_signal=False):
        """Plots the full waterfall and marks signal fc
        
        Params: 
            show_signal: bool
                Enable to show the signal centre track in the plot
                default: False
        """
        
        self.show_signal = show_signal
        plt.figure(figsize=(20,8)) 
        plt.imshow(self.specx[::self.jump], extent=self.leftlim + self.rightlim, origin='lower', aspect='auto')

        if self.show_signal==True:
            for i in range(len(self.fc_middle)):
                plt.plot(self.fc_middle[i], self.time_bins[i], marker='.', color='r')
                plt.plot(self.fc_track[i], self.time_bins[i], marker='.', color='k')

        plt.xlabel('Frequency (kHz)\n'+'\nFile:'+ self.filename)
        plt.ylabel('Time (s)')
        plt.title('Waterfall - Fs:{}kHz Fc:{} Hz'.format(self.fs/1e3, self.fc_middle[0]))
        plt.colorbar()
        plt.show()

    def select_channels(self, channel_center, BW=16e3, show_signal=False):
        """Plots multiple channels in one figure while marking signal fc
        Params:
            channel_center: list
                List of channel centers to plot

            BW: float
                specifies the bandwidth of the channel slice 

            show_signal: bool
                Enable to show the signal centre track in the plot
                default: False
        """

        self.BW = BW
        self.channel_center = channel_center
        self.show_signal = show_signal

        fig, ax = plt.subplots(ncols=len(self.channel_center)) 
        fig.suptitle('Waterfall Channel View\n'+'File:'+self.filename, fontsize=10)
        # fig.set_size_inches(20,8)
        fig.tight_layout(pad=3)

        for n in range(len(self.channel_center)):
            start_band = self.channel_center[n] - (self.BW/2)
            stop_band = self.channel_center[n] + (self.BW/2)

            ax[n].imshow(self.specx[::self.jump], origin='lower', aspect='auto', extent=self.leftlim+self.rightlim)
            
            if show_signal==True:
                #plot signal middle
                for i in range(len(self.fc_middle)):
                    ax[n].plot(self.fc_middle[i], self.time_bins[i], marker='.', color='r')
                    ax[n].plot(self.fc_track[i], self.time_bins[i], marker='.', color='k')

            ax[n].set_xlabel('Frequency (kHz)\n Start band: {}Hz Stop band: {}Hz'.format(start_band, stop_band))
            ax[n].set_ylabel('Time (s)')
            ax[n].set_xlim([start_band, stop_band])
            ax[n].set_title('Fs={} Hz Fc={}KHz, BW = {}kHz'.format(self.fs, self.fc_middle[0]/1e3, self.BW/1e3))
        plt.show()
         
    def find_signal(self, threshold, distance, draw=False):
        """Finds the sig peaks with parameters and selects the ones with distance less than 2500(preceding peak-peak)
            Selects max magnitude or median peaks and checks if the fc_track follows in signal in waterfall 
            
        Params:
            threshold: float
                specifies the minmum signal height

            distance: float
                specifies the minmum peak-peak range
            
            draw: bool
                Enable to view: Signal Peak Detection, Max vs Median peak plots
        """
     
        self.threshold = threshold
        self.distance = distance
        self.draw = draw
        self.fc = 0
        self.fc_track = []
        self.fc_middle = []

        for step in range(self.specx.shape[0]):
            fft_vals = self.specx[step]
            sig_peaks = []
            
            #Detect peaks
            peaks, _ = find_peaks(fft_vals, 
                        height=self.threshold*np.mean(fft_vals),   
                        distance=100, prominence=25)
            peaks = peaks[peaks > int(len(fft_vals)/2)]

            #Selects meadian peaks
            for n in range(0, len(peaks)):
                if(np.abs(peaks[n] - peaks[n-1]) < self.distance):
                    sig_peaks.append(peaks[n-1])
            self.fc_middle.append(np.median(sig_peaks))

            #Selects peaks with max mag 
            for j in range(len(fft_vals[sig_peaks])):
                if (np.max(fft_vals[sig_peaks]) == fft_vals[sig_peaks[j]]):
                    self.fc_track.append(sig_peaks[j])
        
        #Interpolate and/or curve fitting:
        # from scipy import optimize

        # def model(x, a, b):
        #     return int(a)*(x) + b

        # def sigmoid(x, a, b):
        #     return 1/(1+np.exp(-b*(x-1)))
        
        # params, param_cov = optimize.curve_fit(model, np.asarray(self.fc_middle), self.time_bins)
        # plt.plot(self.time_bins, model(self.fc_middle, params[0], params[1]))
        # plt.plot(self.time_bins, self.fc_middle, 'r')
        # plt.show()

        #Checking if fc_track follows signal in waterfall
        for i in range(0, len(self.fc_middle)):
            if (self.specx[i, int(self.fc_middle[i])]) < 45:
                self.fc_middle[i] = 0

            if (self.specx[i, int(self.fc_track[i])]) < 50:
                self.fc_track[i] = 0

        #Doppler track
        delta_freq = []
        for i in range(len(self.fc_track)):
            delta_freq.append(self.fc_track[i-1] - self.fc_track[i])
        if self.draw==True:
            fig, ax = plt.subplots(2,1)
            fig.suptitle('Freq and Doppler track')

            ax[0].plot(self.time_bins, self.fc_track)
            ax[0].plot(self.time_bins, self.fc_middle)
            ax[0].set_xlabel('Time (s)')
            ax[0].set_ylabel('Frequency (khz)')
            ax[0].set_title('Fc from Max vs Median peaks')
            ax[0].legend(['Max Peak', 'Median Peaks'])

            ax[1].plot(self.time_bins, delta_freq)
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Delta Frequency (khz)')    
            ax[1].set_title('Delta Track')
            
            plt.show()

        if self.draw==True:
            index = 0
            fft_vals = self.specx[index]
            fig, ax = plt.subplots(2, 1)
            fig.suptitle('Signal Peak Detection')

            ax[0].plot(fft_vals)
            ax[0].plot(sig_peaks, fft_vals[sig_peaks], "x")
            ax[0].axvline(self.fc_middle[index], color='r')
            ax[0].set_xlabel('Frequency(Hz)')
            ax[0].set_ylabel('Magnitude (dB)')
            ax[0].set_title('Single Window FFT')

            ax[1].plot(fft_vals)
            ax[1].plot(sig_peaks, fft_vals[sig_peaks], "X")
            ax[1].set_xlim([self.fc_track[index]-2e4,self.fc_track[index]+2e4])
            ax[1].set_ylim(bottom=30)
            ax[1].axvline(self.fc_track[index], color='k')
            ax[1].axvline(self.fc_middle[index], color='r')
            ax[1].set_xlabel('Frequency(Hz)')
            ax[1].set_ylabel('Magnitude (dB)')
            ax[1].set_title('Single Window FFT- Cropped')
            plt.show()
        
        del fft_vals, sig_peaks

def args():
    parser = argparse.ArgumentParser(description='Reading File')
    parser.add_argument('--f', help="Specify path to IQ .wav file", type=str)
    parser.add_argument('--save', help="Set this flag to let the program store FFT_iq file", action="store_true")

    return parser.parse_args()

if __name__ == '__main__':
    print("Waterfall Tool")

    args_input = args()

    w = Waterfall()

    w.run(args_input.f, save_flag=args_input.save)
    w.find_signal(1.5, 2500, draw=False)
    w.plot(show_signal=True)
    w.select_channels([1.05e6, 1.045e6], BW=25e3)    

