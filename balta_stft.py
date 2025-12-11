# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 19:47:31 2025

@author: male1
"""

# try:
#     with open('PZT_dis_1.7_2c.txt', 'r') as file:
#         for line in file:
#             print(line.strip())  # .strip() removes leading/trailing whitespace, including newline characters
# except FileNotFoundError:
#     print("Error: The file 'signals.txt' was not found.")
# except Exception as e:
#     print(f"An error occurred: {e}")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

#%% Synthetic signal

fs = 200
T = 2
time = np.linspace(0, T, int(T*fs), endpoint=False)
t1 = 1

x = np.zeros_like(time)
x[time <= t1] = np.sin(2 * np.pi * 50 * time[time <= t1])
x[time > t1] = np.sin(2 * np.pi * 10 * time[time > t1])

#%% STFT Synthetic 
f, t_seg, Zxx = signal.stft(x, fs, nperseg=1000)

plt.figure(2)
plt.plot(time, x)
plt.title("Synthetic signal")
plt.xlabel("Time [s]")
plt.show()

plt.figure(3)
plt.pcolormesh(t_seg, f, np.abs(Zxx), shading = 'gouraud')
plt.title("STFT Synthetic Magnitude")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.show()

#%% Synthetic CWT (Morlet)

import pywt

scales = np.arange(1, 256)

coeffs, freqs = pywt.cwt(x, scales, 'morl', sampling_period = 1/fs)

plt.figure(1)
plt.pcolormesh(time, freqs, np.abs(coeffs), shading='gouraud')
plt.title("CWT Morlet")
plt.ylabel("Frecuencia [Hz]")
plt.xlabel("Time [s]")
plt.ylim([0, 100])
plt.show()

#%% Synthetic CWT (Ricket)

widths = np.arange(1,31)

cwt_result = pywt.cwt(x, widths, 'mexh')[0]

plt.imshow(np.abs(cwt_result), extent=[0, T, widths[-1], widths[0]],
           cmap='PRGn', aspect='auto',
           vmax=abs(cwt_result).max(), vmin=-abs(cwt_result).max())
plt.title("CWT with Mexican Hat Wavelet")
plt.xlabel("Time [s]")
plt.ylabel("Width (Scale)")
plt.show()

#%% Experimental signal

# Cargar el archivo .txt en un DataFrame, especificando el separador
# Si las columnas están separadas por tabulaciones, usa sep='\t'
df = pd.read_csv('PZT_dis_1.7_2c.txt', sep='  ') 

# Acceder a la primera columna
t = df.iloc[:, 0]         
y = df.iloc[:, 1]

# También puedes acceder a la columna usando su nombre (si lo conoces)
# primera_columna = df['nombre_de_la_columna'] 

plt.figure(4)
plt.plot(t,y)
plt.title("Experimental Signal")
plt.grid()
plt.show()

fs = 1 / (t.iloc[-1] - t.iloc[-2])

f, t_seg, Zxx = signal.stft(t, fs, nperseg=999)

plt.figure()
plt.pcolormesh(t_seg, f, np.abs(Zxx), shading = 'gouraud')
plt.title("STFT Experimental Magnitude")
plt.ylim([0, 2500])
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.show()

#%% Experimental CWT (Morlet)

scales = np.arange(1, 1500)

coeffs, freqs = pywt.cwt(t, scales, 'morl', sampling_period = 1/fs)

plt.figure(5)
plt.pcolormesh(t, freqs, np.abs(coeffs), shading='gouraud')
plt.title("CWT Morlet")
plt.ylabel("Frecuencia [Hz]")
plt.xlabel("Time [s]")
plt.ylim([0, 15000])
plt.show()

#%% Experimental (Ricker)

widths = np.arange(1,7500)

cwt_result = pywt.cwt(t, widths, 'mexh')[0]

plt.imshow(np.abs(cwt_result), extent=[0, t.iloc[-1], widths[-1], widths[0]],
           cmap='PRGn', aspect='auto',
           vmax=abs(cwt_result).max(), vmin=-abs(cwt_result).max())
plt.title("CWT with Mexican Hat Wavelet")
plt.xlabel("Time [s]")
plt.ylabel("Width (Scale)")
plt.show()

