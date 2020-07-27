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
            print('Time:',round(time_b - time_a, 2))

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
        plt.figure(figsize=(15,8)) 
        plt.imshow(self.specx[::self.jump], extent=self.leftlim + self.rightlim, origin='lower', aspect='auto')

        plt.plot(self.fc_track, self.time_bins, 'k')
        plt.plot(self.center_twm, self.time_bins, 'r')
        plt.plot(self.center_sp, self.time_bins, 'lightgray')
        plt.xlabel('Frequency (kHz)\n'+'\nFile:'+ self.filename + '\nRuntime:' + str(self.t_fs))
        plt.ylabel('Time (s)')
        plt.title('Waterfall - Fs:{} kHz Fc:{} kHz'.format(self.fs//1e3, self.fc_track[0]//1e3))
        plt.xlim([self.fc_track[0] - 50e3, self.fc_track[0] + 50e3])
        plt.colorbar()
        plt.savefig('waterfal_plot.png', dpi=400, transparent=False)
        plt.show()

    def full_plot(self, show_signal=False):
        """Plots the full waterfall with it's fft spectrum
        
        Params: 
            show_signal: bool
                Enable to show the signal track in the plot
                default: False
        """
    
        self.show_signal = show_signal
        lim = [self.fc_track[0] - 50e3, self.fc_track[0] + 50e3]
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4,1]})
                        
        fig.set_size_inches(15,8)
        fig.tight_layout(pad=3)

        im = axs[0].imshow(self.specx[::self.jump], extent=self.leftlim + self.rightlim, origin='lower', aspect='auto')      
        axs[0].plot(self.fc_track, self.time_bins, 'k')
        axs[0].plot(self.center_sp, self.time_bins, 'white')
        axs[0].set_xlim(lim)
        axs[0].set_xlabel('Frequency(Hz) - white(center), black(max)')
        axs[0].set_ylabel('Magnitude (dBFS)')
        axs[0].set_title('Waterfall')
        fig.colorbar(im, ax=axs[0])

        frame = int(self.num_chunks/2)
        fft_vals = self.specx[frame]
        axs[1].plot(fft_vals, 'lightgray')
        axs[1].plot(self.sp, fft_vals[self.sp]+5, 'k')
        axs[1].plot(self.sp2, fft_vals[self.sp2]+10, 'c')
        axs[1].axvline(self.center_sp[frame], color='r')
        axs[1].set_xlim(lim)
        axs[1].set_ylim([-55,0])
        axs[1].set_xlabel('Frequency(Hz) \n \n' + 'File: ' + self.filename)
        axs[1].set_ylabel('Magnitude (dBFS)')
        axs[1].set_title('FFT Spectrum')
        axs[1].legend(['fft', 'sig-peaks', 'final-peaks', 'centroid'])
        plt.savefig('waterfal_full_plot.png', dpi=400, transparent=False)
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
        self.center_twm = []
        self.center_sp = []
        SNR = []
        time_a = time.time()

        for step in range(self.specx.shape[0]):
            fft_vals = self.specx[step]
            sig_peaks = []
                
            #Detect peaks in spectrum
            peaks, _ = find_peaks(fft_vals, 
                        height=self.threshold*np.mean(fft_vals),   
                        distance=100, prominence=25)

            peaks = peaks[peaks > int(len(fft_vals)/2)]
            t_level = np.mean(fft_vals) * 0.8
            peaks = peaks[fft_vals[peaks] > t_level]

            # find min distance of sig peaks
            min_dist = []
            for n in range(len(peaks)):
                min_dist.append(np.abs(peaks[n] - peaks[n-1]))

            for x in range(len(min_dist)):
                win =11
                win_avg = np.mean(min_dist[x-win:x])
                local_avg = min_dist[x]/win_avg
                if (math.isclose(local_avg, 1, rel_tol=1e-3)):
                    self.distance = min_dist[x]
                    break
            
            #Find signal peaks 
            sig_presence = False
            peak_counter = 0
            self.distance = 2400
            for n in range(0, len(peaks)):
                try:
                    if((np.abs(peaks[n] - peaks[n-1])) >= self.distance-500 and 
                        (np.abs(peaks[n] - peaks[n-1])) <= self.distance+500):
                        sig_peaks.append(peaks[n])

                        if peak_counter>=10:
                            sig_presence = True
                            peak_counter = 0
                            
                        peak_counter +=1
                    else:
                        peak_counter = 0
                except:
                    pass
            sig_present.append(sig_presence)

            #TWM - Error PM
            if (len(sig_peaks)>0):
                f_cands = np.arange(np.median(sig_peaks) - 16e3, np.median(sig_peaks) + 16e3, 10)
                f0_cal, f0_cal_noise, E_PM, E_MP, E_TWM = TWM(sig_peaks, fft_vals[sig_peaks], 20, f_cands)
            else:
                f0_cal =[]
            
            #SNR
            s = np.max(fft_vals[sig_peaks]-1)
            noise_bins = np.arange(1, self.specx.shape[1]+1)
            noise_bins = np.delete(noise_bins, noise_bins[sig_peaks]-1)
            n = np.mean(fft_vals[noise_bins-1])
            SNR.append(s - n)

            #Selects max and median peaks
            if (len(f0_cal) == 0):
                self.fc_track.append(np.nan)
                self.fc_middle.append(np.nan)
            else: 
                for i in range(len(f0_cal)):
                    if (fft_vals[f0_cal[i]] == np.max(fft_vals[f0_cal])):
                        self.fc_track.append(f0_cal[i])      
                        temp = f0_cal[i] 
                self.fc_middle.append(np.median(f0_cal))
            
            intensity = (np.mean(fft_vals[sig_peaks])/SNR[step]) + 0.1
            f_center = []
            sig_peaks = sig_peaks[:-1]

            for i in range(len(sig_peaks)):
                if (fft_vals[sig_peaks[i]]/SNR[step] >= intensity):
                    f_center.append(sig_peaks[i]) 
                    
            print(step, SNR[step], len(f_center))

            self.center_twm.append(find_center(f0_cal, fft_vals[f0_cal]))
            if (len(f_center)>0):    
                self.center_sp.append(find_center(f_center, fft_vals[f_center]))
            else:
                self.center_sp.append(self.center_sp[step-1])
                
            if step==int(self.num_chunks/2):
                self.sp = sig_peaks
                self.sp2 = f_center                

        time_b = time.time()
        self.t_fs = round(time_b - time_a, 2)
        print('Time(find_signal):', self.t_fs)

        from scipy import signal
        win = int(self.num_chunks*0.2)
        win = win if win%2>0 else win+1
        y1 = signal.medfilt(self.fc_track, win)
        y2 = signal.medfilt(self.center_twm, win)
        y3 = signal.medfilt(self.center_sp, win)

        self.fc_track = y1
        self.center_twm = y2
        self.center_sp = y3

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
            plt.show()
        
        del fft_vals, sig_peaks

def find_center(x, x_mag):
    """ Find spectral centroid """

    product_sum = np.sum(x * x_mag)
    mag_sum = np.sum(x_mag)

    return product_sum/mag_sum

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
    rho = 0.33
    a_max = max(pmag)
    fn = np.matrix(f_cands)

    #Error - predicted to Measured
    Error_PM = np.zeros(fn.size) 
    maxn_pm = min(N_peaks, len(peaks))
    for i in range(0, maxn_pm):
        delta_fn = fn.T * np.ones(len(peaks))
        delta_fn = abs(delta_fn - np.ones((fn.size, 1))*peaks)
        delta_fn_final = np.amin(delta_fn, axis=1)

        loc_peak = np.argmin(delta_fn_final, axis=1)
        peak_mag = pmag[loc_peak]
        product = np.array(delta_fn_final)*(np.array(fn.T)**(-p))
        factor = peak_mag/a_max
        Error_PM = Error_PM + (product + factor*(q*product-r)).T

    f0_cal = f_cands[Error_PM[0] < 0].astype(int)
    f0_cal_noise = f_cands[Error_PM[0] > 0].astype(int)

    #Error - measured to predicted
    Error_MP = np.zeros(fn.size)
    peaks = np.array(peaks)
    maxn_mp = min(10, peaks.size)
    new_fcands = np.array(f_cands[Error_PM[0] < 0].astype(int))

    for i in range(0, len(new_fcands)):
        fn = np.round(peaks[:maxn_mp]/new_fcands[i])
        fn = (fn >= 1)*fn + (fn < 1)
        delta_fn_mp = abs(peaks[:maxn_mp] - fn*new_fcands[i])

        product = np.array(delta_fn_mp) * (peaks[:maxn_mp]**(-p))
        peakmag = pmag[:maxn_mp]

        factor = peak_mag/a_max
        Error_MP[i] = np.sum(factor * (product + factor*(q*product - r))) 

    #Total Error
    TWM_Error = (Error_PM[0]/maxn_pm) + (rho * Error_MP/maxn_mp)
    if len(f0_cal)==0:
        f0_cal =[]
    return f0_cal, f0_cal_noise, Error_PM[0], Error_MP, TWM_Error

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
    w.find_signal(1.5, 2500, draw=False)
    # w.plot(show_signal=True)
    w.full_plot()
    # w.select_channels([1.05e6, 1.045e6], BW=25e3)    
