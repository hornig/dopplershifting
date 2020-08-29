# [GSOC 2020| Orbit Determinator] - Tracking Continuous and Sporadic Signals of Satelites

# Introduction 
With increasing popularity in CubeSat technologies, it has gotten ever so important to have low-cost systems that complement the economical and self-reliant nature of today’s cubesats providers. One of the most important parts of an end to end small satellite business is ground-based tracking. Satellite tracking provides valuable information on the whereabouts. Satellite tracking industry is booming with the use of large antennas and high power transmitters at cost-prohibitive nature but at the cost of expense and lead time. 

It is thus important to use an alternative tracking method, for example, Doppler Tracking. Doppler based orbit determination uses a doppler frequency shift to convert to a distance problem. To do doppler tracking, one has to first track the frequency of the signal. This way the cost of the tracking system is kept low because equipment needs beyond the essential receiver are small, at a minimum consisting of an amplifier and a variable oscillator. This project aims to provide a universal tracking solution for burst and continuous type signals of satellites.

## Overview
![overview](https://i.imgur.com/h2c3nV8.png)
This project aims to have a universal tracker for sporadic and continuous type signals. This requires the above workflow. Overall there are three main stages of processing before we arrive at our final track. Every stage has their own function and uses a particular algorithm. 
 - Stage 1: Pre-Processing
 - Stage 2: Decision Making
 - Stage 3: Tracking 

# Installation
 **Requirements:**
 ```
Python-3.x
Numpy >= 1.12.0
Matplotlib >= 2.0.0
Scipy >= 0.18.1

```

# Usage
Simply run the program by locating the iq wav file after "-f"
```
python main.py -f SDRSharp_20190521_184218Z_137500000Hz_IQ.wav
```
Add "-save" arg to save the fft iq and load the same next time you run it. 
 ```
python main.py -f SDRSharp_20190521_184218Z_137500000Hz_IQ.wav -save
```
For multi input:
Input channel frequecies and bandwidth in lists.

If one bandwidth is given, then it's common for all channel frequencies.
```
bw = [32e3] 
f_chan = [137.62e6, 137.62e6] 
```
Enable "draw" to plot 4 spectras in one figure
```
  find_signal(draw=True)
```

# Output
Json Format:
```
filename	:	SDRSharp_20190521_184218Z_137500000Hz_IQ.wav
sampling rate	:	2048000
centre frequency	:	137500000
channel frequency [n]
bandwidth [n]
frequency	[n]
raw-frequency [n]
time [n]
```

| Signal  | ChannelFrequency  | BW  | Waterfall  |  Data |
|---|---|---|---|---|
| NOAA -1 | 137.62 Mhz | 32 kHz | [Waterfall](https://i.imgur.com/7YUSvKv.jpg) | [Data](/results/noaa-2019521-data.json) |
| APRS -1 | 145.825 Mhz | 10 kHz | [Waterfall](https://i.imgur.com/CTGxOq3.png) | [Data](/results/aprs-2019527-data.json) |
| APRS -2 | 145.825 Mhz | 10 kHz | [Waterfall](https://i.imgur.com/n1DcUQQ.png) | [Data](/results/aprs-2019526-data.json) |

# Links
 - [Blog Post](https://aerospaceresearch.net/?p=1942)
 - [Google Docs](https://docs.google.com/document/d/1F9XKZiT5WFehbTom8RvvZalw7KCDICRS1sFiqU8vgps/edit?usp=sharing)
