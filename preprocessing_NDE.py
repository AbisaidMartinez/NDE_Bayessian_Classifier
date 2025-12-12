# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 00:35:05 2025

@author: qbo28
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pywt

# Cargar el archivo .csv en un DataFrame
df = pd.read_csv('Class_2_5MHz.csv')

# Acceder a la primera columna
t = df.iloc[:, 0]

# Tomar el resto de señales almacenadas en un DataFrame, manteniendo un formato matricial
y = df.iloc[:, 1:51]

n_signals = y.shape[1]

#%% Visualización

plt.figure(1)

# Observar las señales obtenidas
for i in range(n_signals):    
    plt.plot(t,y.iloc[:,i] + i/2)
    
plt.title("Experimental Signal")
plt.xlabel("time [s]")
plt.ylabel("y(t)")
plt.grid()
plt.show()

# También puedes acceder a la columna usando su nombre (si lo conoces)
# primera_columna = df['signal_30']

#%% Obtención de STFT

fs = 1 / (t.iloc[-1] - t.iloc[-2])
stfts = []

for i in range(n_signals):
    f, t_seg, Zxx = signal.stft(y.iloc[:, i], fs, nperseg=999)
    stfts.append((f, t_seg, Zxx))

for f, t_seg, Zxx in stfts: 
    plt.figure()
    plt.pcolormesh(t_seg, f, np.abs(Zxx), cmap='gray', shading = 'gouraud')
    plt.title("STFT Experimental Magnitude")
    plt.ylim([0, 0.1e8])
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.show()

#%% Obtención de CWT

widths = np.arange(1,31)
CWT_all = np.zeros((n_signals, len(widths), y.shape[0]))

for i in range(n_signals):
    cwt_result = pywt.cwt(y.iloc[:,i], widths, 'mexh')[0]
    CWT_all[i, :, :] = np.abs(cwt_result)
    
    #plt.imshow(np.abs(cwt_result), extent=[0, t.iloc[-1], widths[-1], widths[0]],
    #           cmap='gray', aspect='auto',
    #           vmax=abs(cwt_result).max(), vmin=-abs(cwt_result).max())
    #plt.title(["CWT of signal ",i ,"with Mexican Hat Wavelet"])
    #plt.xlabel("Time [s]")
    #plt.ylabel("Width (Scale)")
    #plt.show()

#%% Aqui implementamos GLCM 

from skimage.feature import graycomatrix, graycoprops

dist= [1]#[1,2]
ang=[0]#, np.pi/4, np.pi/2, 3*np.pi/4]

contrast = np.zeros(n_signals)
homogeneity = np.zeros(n_signals)
energy = np.zeros(n_signals)
correlation = np.zeros(n_signals)
dissimilarity = np.zeros(n_signals)

for i in range(50):
    image = CWT_all[i, :, :]
    image = (image / image.max() * 255).astype(np.uint8)
    
    GLCM = graycomatrix(image, 
                        distances=dist,
                        angles=ang, 
                        levels=256, 
                        symmetric=True, 
                        normed=True)

    contrast[i] = graycoprops(GLCM, 'contrast')
    homogeneity[i] = graycoprops(GLCM, 'homogeneity')
    energy[i] = graycoprops(GLCM, 'energy')
    correlation[i] = graycoprops(GLCM, 'correlation')
    dissimilarity[i] = graycoprops(GLCM, 'dissimilarity')

#%% Almacenar caracteristicas en matriz

Features = np.zeros((50, 5))
#Features = np.column_stack([contrast, homogeneity, energy, correlation, dissimilarity])

Features = pd.DataFrame({
    'contrast': contrast,
    'homogeneity': homogeneity,
    'energy': energy,
    'correlation': correlation,
    'dissimilarity': dissimilarity
}, index= [f"signal {i+1}" for i in range(n_signals)])

#%% Visualizar esta cosa bien locochona

plt.figure()
plt.imshow(Features)


