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
        # win = np.hanning(len(sig))
        # sig = sig*win    
        norm_fft = fftshift(fft(sig))
        abs_fft = np.abs(norm_fft)/len(sig)
        return abs_fft

    def calFFTPower(self, afft, fs):
        transform = 10 * np.log10(afft/np.max(afft))
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

    def full_plot(self, show_signal=False):
        """Plots the full waterfall and marks signal fc
        
        Params: 
            show_signal: bool
                Enable to show the signal centre track in the plot
                default: False
        """
    
        self.show_signal = show_signal
        lim = [1.03e6, 1.07e6]
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4,1]}) # gridspec_kw={'height_ratios': [1]}
                        
        fig.suptitle('Waterfall')

        # axs[0].imshow(self.specx[::self.jump], extent=self.leftlim + self.rightlim, origin='lower', aspect='auto')      
        # axs[0].set_xlim(lim)
        print(np.std(self.specx[0]))
        axs[0].plot(self.specx[0])
        axs[0].axhline(y=40, color='k')
        axs[0].axhline(y=25, color='r')
        axs[0].axhline(y=np.std(self.specx[0]), color='g')
        axs[0].axhline(y=np.mean(self.specx[0]), color='g', linestyle='--')
        axs[0].axhline(y=np.min(self.specx[0]), color='b', linestyle='-')
        axs[0].axhline(y=np.max(self.specx[0]), color='yellow', linestyle='-')
        axs[0].set_xlabel('Frequency(Hz)')
        axs[0].set_ylabel('Magnitude (dB)')
        axs[0].legend(['fft','y=40', 'y=25', 'SD','Mean','Min'])
        # axs[0].set_ylim(bottom=3)
        # axs[0].set_xlim(lim)
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
        self.fc_new_middle = np.zeros([83,100])
        self.fc_new_max = []
        frame_prev = []
        frame_now = []

        for step in range(self.specx.shape[0]):
            fft_vals = self.specx[step]
            sig_peaks = []
            
            #Detect peaks
            peaks, _ = find_peaks(fft_vals, 
                        height=self.threshold*np.mean(fft_vals),   
                        distance=100, prominence=25)

            for n in range(0, len(peaks)):
                if(np.abs(peaks[n] - peaks[n-1]) < self.distance):
                    sig_peaks.append(peaks[n-1])

            # #PMA-2
            if step==0:
                frame_prev = sig_peaks
            else:
                frame_now = sig_peaks
                for i in range(len(frame_now)):
                    try:
                        # print(step, i, frame_now[i], frame_prev[i], fft_vals[frame_now[i]], np.abs(frame_now[i] - frame_prev[i]))
                        if  np.abs(frame_now[i] - frame_prev[i]) >= 100:
                            sig_peaks[i] = 0
                            pass 
                    except:
                        pass
                frame_prev = frame_now

            # Selects peaks with max mag 
            for j in range(len(fft_vals[sig_peaks])):
                if (fft_vals[sig_peaks[j]] == np.max(fft_vals[sig_peaks])):
                    self.fc_track.append(sig_peaks[j])
                    sig_pos = sig_peaks[j]
            #Selects meadian peaks
            self.fc_middle.append(np.median(sig_peaks))
            
            #Find SNR:
            sig_pos = np.arange(sig_pos - 4*2500, sig_pos + 4*2500, 2500)
            sig_pow = np.mean(fft_vals[sig_pos])
            fft_vals[sig_pos] = 0
            noise_pow = np.mean(fft_vals)
            SNR = sig_pow - noise_pow
            print(SNR)


        print(len(self.fc_track), self.fc_track)

        #Convolution
        from math import pi

        def gauss(n=11,sigma=1):
            r = range(-int(n/2),int(n/2)+1)
            return [1 / (sigma * np.sqrt(2*pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r]
      
        def sigmoid_kernel(n=4):
            x = np.arange(-int(n/2), int(n/2)+1, 1)
            y =  (-1/(1 + np.exp(-x)))
            return  (y - np.mean(y)) / np.std(y)    

        # sigma = np.std(self.fc_middle)
        # print(sigmoid_kernel())
        # fc_middle_conv = np.convolve(sigmoid_kernel(), self.fc_middle, mode='valid')

        # print(sigma)
        # print(len(self.fc_middle))
        # print(len(fc_middle_conv))
        

        # plt.plot(self.fc_middle)
        # plt.plot(fc_middle_conv)
        # plt.title('1d Kernel - Convolution\n' + 'File: ' + self.filename)
        # plt.ylabel('Frequency (Hz)')
        # plt.xlabel('Time (s)')
        # plt.legend(['Input Signal' ,'gauss Conv'])
        # plt.show()

        # plt.plot(self.fc_middle)
        # plt.show()

        # self.fc_middle = fc_middle_conv[:len(self.fc_middle)]

        # #Checking if fc_track follows signal in waterfall
        # for i in range(0, len(self.fc_middle)):
        #     print(i, np.abs(self.fc_middle[i] - self.fc_middle[i-1]))

        #     if np.abs(self.fc_middle[i-1] - self.fc_middle[i]) > 1e3:
        #         self.fc_middle[i-1] = 0 

        # for i in range(0, len(self.fc_middle)):
        #     if (self.specx[i, int(self.fc_middle[i])]) < 45:
        #         self.fc_middle[i] = 0

        #     if (self.specx[i, int(self.fc_track[i])]) < 50:
        #         self.fc_track[i] = 0

        # print(self.fc_middle)

        #Doppler track
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
            ax[0].plot(self.specx[index])
            ax[0].axhline(fft_vals[self.fc_track[index]], color='r')
            ax[0].axhline(np.mean(fft_vals), color='k')
            ax[0].axhline(1.2*np.mean(fft_vals), color='k')
            ax[0].set_title('Signal Level ts=' + str(index))
            ax[0].set_xlabel('Frequency(Hz)')
            ax[0].set_ylabel('Magnitude (dBFS)')
            ax[0].legend(['fft', 'Signal Max', 'Mean', 'Signal Floor'])
            ax[0].set_xlim([0, 2048000])

            index = 40
            fft_vals = self.specx[index]
            ax[1].plot(self.specx[index])
            ax[1].axhline(fft_vals[self.fc_track[index]], color='r')
            ax[1].axhline(np.mean(fft_vals), color='k')
            ax[1].set_title('Signal Level ts=' + str(index))
            ax[1].set_xlabel('Frequency(Hz)')
            ax[1].set_ylabel('Magnitude (dBFS)')
            ax[1].legend(['fft', 'Signal Max', 'Mean'])
            ax[1].set_xlim([0, 2048000])

            index = 80
            fft_vals = self.specx[index]
            ax[2].plot(self.specx[index])
            ax[2].axhline(fft_vals[self.fc_track[index]], color='r')
            ax[2].axhline(np.mean(fft_vals), color='k')
            ax[2].set_title('Signal Level ts=' + str(index))
            ax[2].set_xlabel('Frequency(Hz)\n' +'File: ' + self.filename)
            ax[2].set_ylabel('Magnitude (dBFS)')
            ax[2].legend(['fft', 'Signal Max', 'Mean'])
            ax[2].set_xlim([0, 2048000])

            # ax[0].plot(fft_vals)
            # ax[0].plot(sig_peaks, fft_vals[sig_peaks], "x")
            # ax[0].axvline(self.fc_middle[index], color='r')
            # ax[0].set_xlabel('Frequency(Hz)')
            # ax[0].set_ylabel('Magnitude (dB)')
            # ax[0].set_title('Single Window FFT')
            # ax[0].legend(['fft', 'Sig Max'])

            # ax[1].plot(fft_vals)
            # # ax[1].plot(sig_peaks, fft_vals[sig_peaks], "X")
            # # ax[1].set_xlim([self.fc_track[index]-2e4,self.fc_track[index]+2e4])
            # # ax[1].set_ylim(bottom=30)
            # # ax[1].axvline(self.fc_track[index], color='k')
            # ax[1].axhline(fft_vals[self.fc_track[index]], color='r')
            # # ax[1].axvline(self.fc_middle[index], color='k')
            # ax[1].set_xlabel('Frequency(Hz)')
            # ax[1].set_ylabel('Magnitude (dB)')
            # ax[1].set_title('Single Window FFT ts = 0')
            # ax[1].legend(['fft', 'Sig Max Lvl'])

            # index = 1
            # ax[2].plot(self.specx[index])
            # ax[2].set_ylim(bottom=30)
            # ax[2].axhline(fft_vals[self.fc_track[index]], color='r')
            # ax[2].set_title('Single Window FFT ts = ' + str(index))
            # ax[2].legend(['fft', 'Sig Max Lvl'])
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
    # w.plot(show_signal=True)
    # w.select_channels([1.05e6, 1.045e6], BW=25e3)    

    # w.full_plot()