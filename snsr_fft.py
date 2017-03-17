import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import os


# Define path to import SNSR file in CSV form
path = 'Z:\Operations\Mamaroneck SENSR Files\CSV Files\cx1 1745_20160601_0755.snsr.csv'
filename = path[len(path)-31:len(path)-9]

# Extract string of only filename

#snsr_path = input('Enter path for SNSR file already converted to CSV: ')

#def snsr_fft(path):

# Read in data and dump to separate arrays
alldata = pd.read_csv(path,delimiter = ',',header = 0,skipfooter = 664, engine = 'python', usecols = [3,4,5])


# Separate Accel signal into components
AccelX = np.array(alldata.loc[:,'Accel X(g)'])
AccelY = np.array(alldata.loc[:,'Accel Y(g)'])
AccelZ = np.array(alldata.loc[:,'Accel Z(g)'])

"""
# Read in data and dump to separate arrays
alldata = pd.read_csv(snsr_path,delimiter=',',header=0,usecols=[1,2,3])


# Separate Accel signal into components
AccelX = np.array(alldata.loc[:,'AccelX'])
AccelY = np.array(alldata.loc[:,'AccelY'])
AccelZ = np.array(alldata.loc[:,'AccelZ'])
"""

# Define sampling characteristics
timestep = 60 / len(AccelX)
n = AccelX.size
f = np.fft.fftfreq(n, timestep)
f = f[0:int(len(f)/2)]


# Perform FFT
AccelX_fft = sp.fft(AccelX)
AccelY_fft = sp.fft(AccelY)
AccelZ_fft = sp.fft(AccelZ)


# Take only positive frequency components
AccelX_fft = AccelX_fft[0:int(len(AccelX_fft)/2)]
AccelY_fft = AccelY_fft[0:int(len(AccelY_fft)/2)]
AccelZ_fft = AccelZ_fft[0:int(len(AccelZ_fft)/2)]


# Calculate amplitude spectrum
AccelX_fft_amp = 2*(AccelX_fft.real**2+AccelX_fft.imag**2)**(1/2)/n
AccelY_fft_amp = 2*(AccelY_fft.real**2+AccelY_fft.imag**2)**(1/2)/n
AccelZ_fft_amp = 2*(AccelZ_fft.real**2+AccelZ_fft.imag**2)**(1/2)/n


#  Plot
plt.figure(1)
plt.plot(f,AccelX_fft_amp, linewidth=0.75)
plt.grid(True)
plt.title('X Accel Spectral Power',weight='bold',fontsize='x-large')
plt.xlabel('Frequency (hz)',weight='bold')
plt.xlim((0,40))
plt.ylim((0,max(AccelX_fft_amp)*1.1))
plt.savefig(filename + ' - X Accel Spectral Power.png',dpi = 500)

plt.figure(2)
plt.plot(f,AccelY_fft_amp, linewidth=0.75)
plt.grid(True)
plt.title('Y Accel Spectral Power',weight='bold',fontsize='x-large')
plt.xlabel('Frequency (hz)',weight='bold')
plt.xlim((0,40))
plt.ylim((0,max(AccelY_fft_amp)*1.1))

plt.show()
plt.savefig(filename + ' - Y Accel Spectral Power.png',dpi = 500)
	
#snsr_fft(snsr_path)