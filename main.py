import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import argparse, sys, time, math, json
from scipy.fftpack import fft, fftshift, ifft

class Waterfall():
    """Waterfall Tool Main Class"""

    def __init__(self, fs, fc, f_chan, BW):

        self.fs = fs
        self.BW = BW
        self.fc = fc
        self.f_chan = f_chan

    def calFFT(self, sig):   
        norm_fft = (1/self.fs)*fftshift(fft(sig))
        abs_fft = np.abs(norm_fft)
        return abs_fft

    def calFFTPower(self, afft, fs): 
        transform = 10 * np.log10(afft/127)
        return transform

    def run(self,filename, save_flag=False):
        """Loads the data and performs fft in chunks.
        
        Params:
            filename: str
                name of the iq file in .wav
            
            save_flag: bool
                Enable to the save the fft of iq in .npy
                which will autoload next time.
        """

        self.overlap = 0.5
        offset = 44
        T = 1/fs
        iterate = 0 
        adc_offset = -127.5
        window = self.fs
        self.filename = filename
        self.save_flag = save_flag
        data = np.memmap(filename, offset=offset)
        self.total_duration = T*(len(data)/2)
        self.num_chunks = int(len(data)/(window*2))
        file_name = 'Spec_'+self.filename.split('.wav')[0]+'.npy'
        # self.filename = self.filename.split('/')[-1].split('.wav')[0]

        #ErrorHandling:
        if len(self.BW) > 1:
            if not len(self.BW) == len(self.f_chan):
                print('Error: Number of bw need to be equal to number of f_chan given')
                sys.exit()
        elif len(self.f_chan) > 1:
            self.BW = self.BW * len(self.f_chan)

        for j in range(len(self.f_chan)):
            if np.abs(self.f_chan[j] - self.fc) > self.fs/2:
                print('Error: Frequency offset is out of range')
                sys.exit()

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
                data_slice = adc_offset + (data[slice: slice + window * 2: 2]) + 1j * (adc_offset + data[slice+1: slice + window * 2: 2])
                
                fft_iq = self.calFFT(data_slice)
                
                transform = self.calFFTPower(fft_iq, self.fs)
                
                self.specx[self.num_chunks-iterate-1] = transform

                iterate +=1
                progress(iterate, self.num_chunks)
                del data_slice,transform,fft_iq   
                
            del data
            self.specx = np.flip(self.specx, 0)
            time_b = time.time()

            if self.save_flag==True:
                print('\nSaving data to disk')
                np.save(file_name, self.specx)
                print('iq_spec saved.', end=' ')
            print('Time:',round(time_b - time_a, 2))

        if self.num_chunks>100:
            self.jump = 5
        else:
            self.jump = 2
        
        self.time_bins = np.linspace(0, self.total_duration, self.specx.shape[0])
        self.leftlim = (0, self.fs)
        self.rightlim = [0, self.total_duration]

    def plot_default(self):
        """Plots the full waterfall only.
        
        Params: 
            show_signal: bool
                Enable to show the signal centre track in the plot
                default: False
        """
        plt.figure(figsize=(12,8)) 
        plt.imshow(self.specx[::2], extent=self.leftlim + self.rightlim, origin='lower', aspect='auto')       
        plt.xlabel('Frequency Bins \n'+'\nFile:'+ self.filename + '\nRuntime:' + str(self.t_fs))
        plt.ylabel('Time (s)')
        plt.title('Waterfall')
        plt.colorbar()
        plt.savefig('waterfal_just_plot.png', dpi=400, transparent=False)
        plt.show()

    def plot(self):
        """Plots the full waterfall and the signal track.
        
        Params: 
            show_signal: bool
                Enable to show the signal centre track in the plot
                default: False
        """
        timebin = np.linspace(0, self.total_duration, self.specx.shape[0])
        freq_vector = [self.fc - (self.fs/2), (self.fs/2) + self.fc]
        
        plt.figure(figsize=(12,8))  
        plt.imshow(self.specx[::self.jump], extent=freq_vector + self.rightlim, origin='lower', aspect='auto')       
        plt.plot(self.track_center[0], self.time_bins[0], color = 'k')
        plt.plot(self.raw_center[:, 0], timebin, color = 'white', marker='.', alpha=0.5)
        plt.xlabel('Frequency (Hz) \n'+'\nFile:'+ self.filename + '\nRuntime:' + str(self.t_fs))
        plt.ylabel('Time (s)')
        plt.title('Waterfall')
        plt.xlim([self.f_chan[0] - self.BW[0]/2, self.f_chan[0] + self.BW[0]/2])
        plt.colorbar()
        plt.savefig('waterfal_plot.png', dpi=400, transparent=False)
        plt.show()

    def multi_plot(self):
        """Plots multiple channels in one figure

        Params:
            show_signal: bool
                Enable to show the signal centre track in the plot
                default: False
        """

        n_plots = len(self.track_center) 
        freq_vector = [self.fc - (self.fs/2), (self.fs/2) + self.fc]
        
        fig, ax = plt.subplots(nrows=1,ncols=n_plots) 
        fig.suptitle('Waterfall Multi Channel View\n'+'File:'+self.filename, fontsize=10)
        fig.set_size_inches(15,8)
        fig.tight_layout(pad=3)
        self.jump = 5
        for n in range(0, n_plots):
            ax[n].imshow(self.specx[::self.jump], extent=freq_vector + self.rightlim, origin='lower', aspect='auto')
            ax[n].plot(self.track_center[n], self.time_bins[n], color = 'k', marker=".")
            ax[n].set_xlabel('Frequency (Hz) \n  F_chan: {}Hz F_c: {}Hz'.format(self.f_chan[n], self.fc))
            ax[n].set_ylabel('Time (s)')
            chan_start = self.f_chan[n] - self.BW[n]/2
            chan_end = self.f_chan[n] + self.BW[n]/2
            ax[n].set_xlim([chan_start, chan_end]) 
            ax[n].set_title('Channel: {} BW: {}'.format(n, self.BW[n]))
        # fig.colorbar(im)
        plt.savefig('waterfall_multi_plot.png', dpi=200, transparent=False)
        plt.show()
        
    def find_signal(self, draw=False):
        """Finds the signal by taking decision of neighbouring frequency bins when above a calculated threshold.
            Plots the spectra and fits the final track.
            
        Params:
            draw: bool
                Enable to view: Four Frames of Spectra in one figure. 
        """

        self.draw = draw
        self.fc_track = []
        self.fc_middle = []
        self.sig_present = False
        self.track_center = []
        pc = 0
        time_a = time.time()
        
        #mean
        sum_fft = np.zeros(int(self.fs))
        for i in range(self.specx.shape[0]):
            sum_fft += self.specx[i]
        fft = sum_fft/self.num_chunks 
        
        channel_start, channel_end = find_channel(self.fs, self.fc, self.f_chan, self.BW)
        frame = np.linspace(0, self.num_chunks-1, 4).astype(int)
        sig_center = np.zeros([self.specx.shape[0], len(self.f_chan)])
        sig_freqs = np.zeros([self.specx.shape[0], len(self.f_chan)])
        # print(self.BW, self.f_chan, channel_start, channel_end)

        for step in range(self.specx.shape[0]):
            # progress(step, self.num_chunks)

            #Spectral Average
            fft_vals = self.specx[step]
            fft_vals = self.specx[step] - fft

            #Threshold
            mean = np.mean(fft_vals[fft_vals > 0])
            sd = np.std(fft_vals[fft_vals > 0])
            safety = 0.5
            threshold = mean + sd + safety                  
            
            #Decision Type 2:
            c = 0 
            fft_threshold_idx = []
            full_spectrum = False

            for f_c in self.f_chan:
                if full_spectrum == False:
                    for i in range(int(channel_start[c]), int(channel_end[c]), 1):
                        if(fft_vals[i] > threshold and fft_vals[i-1] > threshold and fft_vals[i-2] > threshold):
                            fft_threshold_idx.append(i)
                else:
                    for i in range(self.specx.shape[0]):
                        if(fft_vals[i] > threshold and fft_vals[i-1] > threshold and fft_vals[i-2] > threshold):
                            fft_threshold_idx.append(i)

                centroid = find_center(fft_vals[fft_threshold_idx], fft_threshold_idx)

                if len(fft_threshold_idx) > 200:
                    sig_center[step, c] = centroid 
                    sig_freqs[step, c] = self.f_chan[c] + centroid - (channel_start[c] + self.BW[c]/2) 
                else:
                    sig_center[step, c] = np.nan
                    sig_freqs[step, c] = np.nan
                print(step, c, centroid, len(fft_threshold_idx))
                c +=1

            if self.draw==True:
                """Plots a multiple frames of channel spectra and one full spectra with channel markings"""

                if step == 8:
                    if step == 8:
                        fig, ax = plt.subplots(2, 1)
                        fig.tight_layout()
                        fig.set_size_inches(15,10)

                    ax[0].plot(fft_vals)
                    ax[0].axvline(sig_center[step, 0], color='r', label="centroid")
                    ax[0].plot(fft_threshold_idx, fft_vals[fft_threshold_idx], color='k', label="fft_threshold")
                    ax[0].set_title('Chan Spectra ts=' + str(step))
                    ax[0].set_xlabel('Frequency Bins M:{} SD:{} T:{}'.format(round(mean), round(sd), round(threshold)))
                    ax[0].set_ylabel('Magnitude (dBFS)')
                    ax[0].axhline(mean, color='k', label="mean(+)")
                    ax[0].axhline(sd, color='orange', label="sd(+)")
                    ax[0].axhline(threshold, color='yellow', label="threhshold")
                    ax[0].set_xlim([channel_start[0], channel_end[0]])
                    ax[0].grid()
                    ax[0].legend(loc="best")

                    ax[1].plot(fft_vals)
                    ax[1].plot(fft_threshold_idx, fft_vals[fft_threshold_idx], color='k', label="fft_threshold")
                    ax[1].set_title('Chan Spectra ts=' + str(step))
                    ax[1].set_xlabel('Frequency Bins M:{} SD:{} T:{}'.format(round(mean), round(sd), round(threshold)))
                    ax[1].set_ylabel('Magnitude (dBFS)')
                    ax[1].axhline(mean, color='k', label="mean(+)")
                    ax[1].axhline(sd, color='orange', label="sd(+)")
                    ax[1].axhline(threshold, color='yellow', label="threhshold")
                    ax[1].axvspan(channel_start[0], channel_end[0],  facecolor='green', alpha=0.4, label="channel") 
                    ax[1].set_xlim([0, self.fs])
                    ax[1].grid()
                    ax[1].legend(loc="best")
                    plt.savefig('spectra-plt.png', dpi=200, transparent=False)
                    plt.show() 
                    if step == frame[-1]:
                        for i in range(len(self.f_chan)):
                            ax[pc].axvline(sig_center[step, i], color='r', label="centroid")
                            ax[pc].axvspan(channel_start[i], channel_end[i],  facecolor='green', alpha=0.4, label="chan "+str(i)) 
                        ax[pc].set_title('Full Spectra ts=' + str(step))
                        ax[pc].set_xlim([0, self.fs])
                        plt.savefig('spectra-plt.png', dpi=200, transparent=False)
                        plt.show() 
                    pc+=1            

        time_b = time.time()
        self.t_fs = round(time_b - time_a, 2)
        print('\nTime(find_signal):', self.t_fs)
        
        #Polyfit
        time_bin = []
        new_freqs = []
        self.raw_center = []

        win = int(self.num_chunks*0.3)
        win = win if win%2>0 else win+1

        for i in range(0, sig_freqs.shape[1]):
            freqs = sig_freqs[:,i]
            valid = ~np.isnan(freqs)
            freqs = freqs[valid]
            self.raw_center.append(freqs.tolist())
            time_bin.append(self.time_bins[valid].tolist())

            if len(time_bin[i]) > 0:   
                # freqs = signal.medfilt(freqs, win)             
                p = np.poly1d(np.polyfit(time_bin[i], freqs, 10))
                
                result = p(time_bin[i])
                new_freqs.append(result.tolist())
                self.sig_present = True
            else:
                new_freqs = []
                time_bin = []
                print('No signal found')
        
        #dump to json 
        data = {'filename': self.filename, 
                'sampling rate': self.fs,
                'centre frequency':self.fc,
                'channel frequency': self.f_chan,
                'bandwidth': self.BW,
                "frequency": new_freqs,
                "raw-frequency": self.raw_center,
                'time': time_bin}    
        
        with open("data.json", "w") as outfile: 
            json.dump(data, outfile, indent=2)
        
        # self.plot_default()
        self.track_center = new_freqs
        self.time_bins = np.array(time_bin)
        self.raw_center = sig_freqs

        if (self.sig_present):
            if sig_freqs.shape[1] > 1:
                self.multi_plot(channel_start, channel_end)
            else:
                self.plot()

        del fft_vals, new_freqs, time_bin, valid

def find_channel(fs, fc, f_chan, bw):
    """ Finds multiple channel centers with offset """
    start = []
    end = []

    for i in range(len(f_chan)):
        center = fs/2 + (f_chan[i] - fc)
        start.append(center - bw[i]/2)
        end.append(center + bw[i]/2)

    return start, end

def find_center(x_mag, x_idx):
    """ Find spectral centroid """

    if len(x_mag)>0:
        product_sum = np.sum(x_idx * x_mag)
        mag_sum = np.sum(x_mag)
        result = product_sum/mag_sum
    else:
        result = 0

    return result

def progress(now, total):
    sys.stdout.write("\r")
    sys.stdout.write("{}%".format(round(((now/total)*100),2)))
    sys.stdout.flush()

def args():
    parser = argparse.ArgumentParser(description='Reading File')
    parser.add_argument('-f', help="Specify path to IQ .wav file", type=str)
    parser.add_argument('-save', help="Set this flag to save FFT_iq file", action="store_true")

    return parser.parse_args()

if __name__ == '__main__': 
    print("Waterfall Tool")

    args_input = args()
    fs = 2048000
    bw = [10e3] 
    fc = 145.825e6
    f_chan = [145.825e6] 

    w = Waterfall(fs, fc, f_chan, bw)
    w.run(args_input.f, args_input.save)
    w.find_signal(draw=False)
    # w.plot()
    # w.multi_plot()
