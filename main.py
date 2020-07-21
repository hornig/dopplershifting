import numpy as np
import argparse, sys, time, math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft, fftshift, ifft

class Waterfall():
    """Waterfall Tool Main Class"""

    def __init__(self):
        self.specx = []

    def calFFT(self, sig):
        # win = np.hanning(len(sig))
        # sig = sig*win    
        norm_fft = fftshift(fft(sig))
        abs_fft = np.abs(norm_fft)/len(sig)
        return abs_fft

    def calFFTPower(self, afft, fs): 
        transform = 10 * np.log10(afft/127)
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
        self.num_chunks = int(len(data)/(chunk*2))
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
            self.specx = np.zeros([self.num_chunks, window])    

            time_a = time.time()
            for slice in range(0, int(len(data) // (window * 2)) * window * 2, window*2): 
                data_slice = adc_offset + (data[slice: slice + window * 2: 2]) + 1j * (adc_offset + data[slice + 1: slice + window * 2: 2])
                
                fft_iq = self.calFFT(data_slice)

                transform = self.calFFTPower(fft_iq, self.fs)

                self.specx[self.num_chunks-iterate-1] = transform

                iterate +=1
                progress(iterate, self.num_chunks)
                del data_slice,transform,fft_iq   
            
            del data
            time_b = time.time()

            if self.save_flag==True:
                print('\nSaving data to disk')
                np.save(file_name, self.specx)
                print('iq_spec saved.', end=' ')
            print('Time:',round(time_b - time_a),2)

        if self.num_chunks>100:
            self.jump = 15
        else:
            self.jump = 10
        
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
            for i in range(len(self.fc_track)):
                plt.plot(self.fc_middle[i], self.time_bins[i], marker='.', color='r')
                plt.plot(self.fc_track[i], self.time_bins[i], marker='.', color='k')

        plt.xlabel('Frequency (kHz)\n'+'\nFile:'+ self.filename)
        plt.ylabel('Time (s)')
        plt.title('Waterfall - Fs:{}kHz Fc:{} Hz'.format(self.fs/1e3, self.fc_middle[0]))
        plt.colorbar()
        plt.show()

    def full_plot(self, show_signal=False):
        """Plots the full waterfall with it's fft spectrum
        
        Params: 
            show_signal: bool
                Enable to show the signal track in the plot
                default: False
        """
    
        self.show_signal = show_signal
        # lim = [1.03e6, 1.07e6]
        lim = [0, 2048000]
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4,1]})
                        
        fig.suptitle('Waterfall')
        fig.set_size_inches(15,8)
        fig.tight_layout(pad=3)

        axs[0].imshow(self.specx[::self.jump], extent=self.leftlim + self.rightlim, origin='lower', aspect='auto')      
        axs[0].set_xlim(lim)
        axs[0].set_xlabel('Frequency(Hz)')
        axs[0].set_ylabel('Magnitude (dBFS)')
        axs[0].set_title('Spectrogram')
        
        axs[1].plot(self.specx[0])
        axs[1].set_xlim(lim)
        axs[1].set_ylim([-80,0])
        axs[1].set_xlabel('Frequency(Hz) \n \n' + 'File: ' + self.filename)
        axs[1].set_title('FFT Spectrum')
        axs[1].set_ylabel('Magnitude (dBFS)')
        axs[1].legend(['fft'])
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
        sig_present = []

        for step in range(self.specx.shape[0]):
            fft_vals = self.specx[step]
            sig_peaks = []
                
            #Detect peaks in spectrum
            peaks, _ = find_peaks(fft_vals, 
                        height=self.threshold*np.mean(fft_vals),   
                        distance=100, prominence=25)

            peaks = peaks[peaks > int(len(fft_vals)/2)]

            #find min distance of sig peaks
            min_dist = []
            for n in range(len(peaks)):
                min_dist.append(np.abs(peaks[n] - peaks[n-1]))

            for x in range(len(min_dist)):
                win = 5
                win_avg = np.mean(min_dist[x-win:x])
                local_avg = min_dist[x]/win_avg
                if (math.isclose(local_avg, 1, rel_tol=1e-3)):
                    self.distance = min_dist[x]
                    break

            #Find signal peaks 
            sig_presence = False
            peak_counter = 0
            self.distance = 2500
            for n in range(0, len(peaks)):
                try:
                    if((np.abs(peaks[n] - peaks[n-1])) >= self.distance-500 and 
                        (np.abs(peaks[n] - peaks[n-1])) <= self.distance+500):
                        if((np.abs(peaks[n+1] - peaks[n])) >= self.distance-500 and 
                            (np.abs(peaks[n+1] - peaks[n])) <= self.distance+500):
                            sig_peaks.append(peaks[n])

                            if peak_counter>=9:
                                sig_presence = True
                                peak_counter = 0
                                
                            peak_counter +=1
                        else:
                            peak_counter = 0
                except:
                    pass
            sig_present.append(sig_presence)

            if step==0:
                peak_zero = peaks
                sig_zero = sig_peaks
                
            if step==int(self.num_chunks/2):
                peak_mid = peaks
                sig_mid = sig_peaks

            #TWM - Error PM
            f0_cal, f0_cal_noise = TWM(sig_peaks, fft_vals[sig_peaks], 20, f_cands=np.arange(1.02e6,1.1e6,1.0)) 

            #New SNR
            s = np.max(fft_vals[f0_cal]-1)
            noise_bins = np.arange(1, self.specx.shape[1]+1)
            noise_bins = np.delete(noise_bins, noise_bins[f0_cal-1]-1)
            n = np.mean(fft_vals[noise_bins-1])
            SNR = (s - n)
            print(step, s, n, SNR)

            #Selects max and median peaks
            for i in range(len(f0_cal)):
                if (fft_vals[f0_cal[i]] == np.max(fft_vals[f0_cal])):
                    self.fc_track.append(f0_cal[i])           

            self.fc_middle.append(np.median(fft_vals[f0_cal]))

        #print
        for i in range(len(self.fc_track)):
            plt.plot(self.fc_middle[i], self.time_bins[i], marker='.', color='r')
            plt.plot(self.fc_track[i], self.time_bins[i], marker='.', color='k')
        plt.show()
        

        # Doppler track
        # delta_freq = []
        # for i in range(len(self.fc_track)):
        #     delta_freq.append(self.fc_track[i-1] - self.fc_track[i])
        # if self.draw==True:
        #     fig, ax = plt.subplots(2,1)
        #     fig.suptitle('Freq and Doppler track')

        #     ax[0].plot(self.time_bins, self.fc_track)
        #     ax[0].plot(self.time_bins, self.fc_middle)
        #     ax[0].set_xlabel('Time (s)')
        #     ax[0].set_ylabel('Frequency (khz)')
        #     ax[0].set_title('Fc from Max vs Median peaks')
        #     ax[0].legend(['Max Peak', 'Median Peaks'])

        #     ax[1].plot(self.time_bins, delta_freq)
        #     ax[1].set_xlabel('Time (s)')
        #     ax[1].set_ylabel('Delta Frequency (khz)')    
        #     ax[1].set_title('Delta Track')
        #     plt.show()
        
        if self.draw==True:

            fig, ax = plt.subplots(3, 1)
            fig.tight_layout()

            index = 0
            fft_vals = self.specx[index]
            ax[0].plot(fft_vals)
            ax[0].plot(peak_zero, fft_vals[peak_zero], "x", color='k')
            ax[0].plot(sig_zero, fft_vals[sig_zero], "x")
            ax[0].plot(f0_cal, fft_vals[f0_cal],  marker='.', color='g')
            ax[0].plot(f0_cal_noise, fft_vals[f0_cal_noise],  marker='.', color='k')
            ax[0].axhline(np.max(fft_vals[f0_cal]), color='r')
            ax[0].axhline(np.mean(fft_vals), color='k')
            ax[0].set_title('Signal Level ts=' + str(index) + ' '+'Sig_Presence: ' + str(sig_present[index]))
            ax[0].set_xlabel('Frequency(Hz)')
            ax[0].set_ylabel('Magnitude (dBFS)')
            ax[0].legend(['fft', 'peaks', 'sig_peaks', 'Error_PM < 0 - f0', 'Error_PM >0 - noise'])
            ax[0].set_xlim([0, self.fs])

            # index = int(self.num_chunks/2)
            # fft_vals = self.specx[index]
            # ax[1].plot(fft_vals)
            # ax[1].plot(peak_mid, fft_vals[peak_mid], "x", color='k')
            # ax[1].plot(sig_mid, fft_vals[sig_mid], "x")
            # ax[1].axhline(fft_vals[self.fc_track[index]], color='r')
            # ax[1].axhline(np.mean(fft_vals), color='k')
            # ax[1].set_title('Signal Level ts=' + str(index) + ' '+'Sig_Presence: ' + str(sig_present[index]))
            # ax[1].set_xlabel('Frequency(Hz)')
            # ax[1].set_ylabel('Magnitude (dBFS)')
            # ax[1].legend(['fft', 'peaks', 'sig_peaks', 'Signal Max'])
            # ax[1].set_xlim([0, self.fs])

            # index = self.num_chunks - 1
            # fft_vals = self.specx[index]
      
            # ax[2].plot(fft_vals)
            # ax[2].plot(peaks, fft_vals[peaks], "x", color='k')
            # ax[2].plot(sig_peaks, fft_vals[sig_peaks], "x")
            # # ax[2].axhline(fft_vals[self.fc_track[index]], color='r')
            # ax[2].axhline(s, linestyle='-')
            # ax[2].axhline(n, linestyle='-.')
            # ax[2].axhline(n*0.8, linestyle='--', color='k')
            # ax[2].set_title('Signal Level ts=' + str(index) + ' '+'Sig_Presence: ' + str(sig_present[index]))
            # ax[2].set_xlabel('Frequency(Hz)\n' +'File: ' + self.filename)
            # ax[2].set_ylabel('Magnitude (dBFS)')
            # ax[2].set_xlim([0, self.fs])
            plt.show()
        
        del fft_vals, sig_peaks

def TWM(peaks, pmag, N_peaks, f_cands):
    """Two-Way Mistmatch Algorithm to find the best possible case of `f0`  
    Params:
        peaks: list
            fft spectral peaks

        pmag: list
            fft mags of respective peaks

        N_peaks: int
            Number of spectral peaks

    Returns:
        f0_cal: array
            frequency index bin bellow zero

        f0_cal_noise: array
            frequency index bin above zero
    """
    #initiate params
    p = 0.5
    q = 1.4
    r = 0.4
    a_max = max(pmag)
    fn = np.matrix(f_cands)

    #Error - predicted to Measured
    Error_PM = np.zeros(fn.size) 
    maxn = min(N_peaks, peaks.size)
    for i in range(0, maxn):
        delta_fn = fn.T * np.ones(peaks.size)
        delta_fn = abs(delta_fn - np.ones((fn.size, 1))*peaks)
        delta_fn_final = np.amin(delta_fn, axis=1)

        loc_peak = np.argmin(delta_fn_final, axis=1)
        peak_mag = pmag[loc_peak]
        product = np.array(delta_fn_final)*(np.array(fn.T)**(-p))
        factor = peak_mag/a_max
        Error_PM = Error_PM + (product + factor*(q*product-r)).T

    f0_cal = f_cands[Error_PM[0] < 0].astype(int)
    f0_cal_noise = f_cands[Error_PM[0] > 0].astype(int)

    # plt.plot(f_cands, Error_PM[0])
    # plt.ylabel('Error - Predicted->Measured')
    # plt.xlabel('Freq Candidates')
    # plt.title('TWM Error_PM')
    # plt.show()

    return f0_cal, f0_cal_noise

def progress(now, total):
    sys.stdout.write("\r")
    sys.stdout.write("{}%".format(round(((now/total)*100),2)))
    sys.stdout.flush()

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
    w.find_signal(1.5, 2500, draw=True)
    w.plot(show_signal=True)
    # w.full_plot()
    # w.select_channels([1.05e6, 1.045e6], BW=25e3)    
