# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 00:33:17 2025

@author: qbo28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.feature import graycomatrix, graycoprops

def Extract_Features_STFT(filename,
                          n_signals=50,
                          nperseg=256,  # ventana para STFT (ajustable)
                          noverlap=128, # overlap para mejor resolución
                          distances=[1],
                          angles=[0],
                          fs=None  # si no se da, se calcula del tiempo
    ):
    """
    Extrae características GLCM a partir de STFT de múltiples señales.
    
    Parámetros:
        filename: ruta del archivo .csv
        n_signals: número de señales (columnas) a procesar
        nperseg: longitud de la ventana para STFT
        noverlap: overlap entre ventanas
        distances: distancias para el GLCM
        angles: ángulos para el GLCM en radianes
        fs: frecuencia de muestreo (si None, se calcula del tiempo)
        
    Retorna:
        DataFrame con contrast, homogeneity, energy, correlation y dissimilarity
    """
    
    # --- Cargar archivo ---
    df = pd.read_csv(filename)
    y = df.iloc[:, 1:n_signals+1]  # señales en columnas 1..n
    
    # Calcular fs si no se proporciona
    if fs is None:
        dt = df.iloc[1, 0] - df.iloc[0, 0]  # asumiendo que tiempo está en columna 0
        fs = 1 / dt
    
    # --- Inicializar características ---
    contrast = np.zeros(n_signals)
    homogeneity = np.zeros(n_signals)
    energy = np.zeros(n_signals)
    correlation = np.zeros(n_signals)
    dissimilarity = np.zeros(n_signals)
    
    print(f"Procesando {n_signals} señales con STFT (fs={fs:.2f} Hz, nperseg={nperseg})")
    
    # --- STFT + GLCM para cada señal ---
    for i in range(n_signals):
        # Calcular STFT
        f, t_stft, Zxx = signal.stft(y.iloc[:, i], 
                                   fs=fs, 
                                   nperseg=nperseg, 
                                   noverlap=noverlap,
                                   window='hann')  # ventana de Hann para mejor resolución
        
        # --- TRANSFORMACIÓN LOGARÍTMICA + NORMALIZACIÓN (CLAVE) ---
        # 1. Convertir a dB (escala logarítmica)
        Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)  # +1e-10 para evitar log(0)
        
        # 2. Recortar valores extremos para reducir el padding
        Zxx_db = np.clip(Zxx_db, -60, 0)  # recortar entre -60dB y 0dB
        
        # 3. Normalizar a [0, 255]
        Zxx_norm = (Zxx_db - Zxx_db.min()) / (Zxx_db.max() - Zxx_db.min()) * 255
        image = Zxx_norm.astype(np.uint8)
        
        # --- Visualización opcional (comenta si no la quieres) ---
        # plt.figure(figsize=(8, 6))
        # plt.pcolormesh(t_stft, f, np.abs(Zxx), cmap='viridis', shading='gouraud')
        # plt.title(f"STFT Original - Señal {i+1}")
        # plt.ylim([0, fs/2 * 0.1])  # 10% de la frecuencia de Nyquist
        # plt.ylabel("Frecuencia [Hz]")
        # plt.xlabel("Tiempo [s]")
        # plt.colorbar(label='Magnitud')
        # plt.show()
        
        # plt.figure(figsize=(8, 6))
        # plt.pcolormesh(t_stft, f, image, cmap='gray', shading='gouraud')
        # plt.title(f"STFT Normalizada para GLCM - Señal {i+1}")
        # plt.ylim([0, fs/2 * 0.1])
        # plt.ylabel("Frecuencia [Hz]")
        # plt.xlabel("Tiempo [s]")
        # plt.colorbar(label='Intensidad [0-255]')
        # plt.show()
        
        # --- Calcular GLCM ---
        glcm = graycomatrix(image,
                            distances=distances,
                            angles=angles,
                            levels=256,
                            symmetric=True,
                            normed=True)
        
        # Extraer propiedades
        contrast[i] = graycoprops(glcm, 'contrast')#.mean()
        homogeneity[i] = graycoprops(glcm, 'homogeneity')#.mean()
        energy[i] = graycoprops(glcm, 'energy')#.mean()
        correlation[i] = graycoprops(glcm, 'correlation')#.mean()
        dissimilarity[i] = graycoprops(glcm, 'dissimilarity')#.mean()
        
        #print(f"Señal {i+1}: Contrast={contrast[i]:.2f}, Energy={energy[i]:.2f}")
    
    # --- DataFrame final ---
    Features = pd.DataFrame({
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'dissimilarity': dissimilarity
    }, index=[f"signal {i+1}" for i in range(n_signals)])
    
    return Features

Feature01 = Extract_Features_STFT("Class_1_5MHz.csv")
Feature02 = Extract_Features_STFT("Class_1_15MHz.csv")

Features01 = pd.concat([Feature01, Feature02])
Features01['Class'] = 'Class_1'

Feature03 = Extract_Features_STFT("Class_2_5MHz.csv")
Feature04 = Extract_Features_STFT("Class_2_15MHz.csv")

Features02 = pd.concat([Feature03, Feature04])
Features02['Class'] = 'Class_2'

Feature05 = Extract_Features_STFT("Class_3_5MHz.csv")
Feature06 = Extract_Features_STFT("Class_3_15MHz.csv")

Features03 = pd.concat([Feature05, Feature06])
Features03['Class'] = 'Class_3'

Features = pd.concat([Features01, Features02, Features03])

#%% Cambiar indexación

# Supongamos que Features tiene 300 filas
num_filas = Features.shape[0]

# Crear la lista de nuevos índices
nuevo_index = [f"signal_{i}" for i in range(1, num_filas + 1)]

# Asignar el nuevo índice
Features.index = nuevo_index

#%% Preprocesar para PCA 

from sklearn.preprocessing import StandardScaler

features = Features.columns.tolist()[:-1]

#%%  Analisis a partir de la matriz de correlación

import seaborn as sns

R = Features[features].corr(min_periods=3)

plt.figure()
sns.heatmap(R, annot = True, fmt='.2f', cmap='coolwarm',
            linewidths=.5, square=True)         
plt.title("Correlation Matrix")
plt.show()

#%% Separating out the features
x = Features.loc[:, features].values

# Separating out the target
y = Features.loc[:,['Class']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

#%% Criterio de selección de número de componentes

from sklearn.decomposition import PCA

# PCA con todos los componentes posibles (5 en este caso)
pca_full = PCA()
pca_full.fit(x)  # x son los datos ya estandarizados

# Varianza explicada por cada componente
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Imprimir valores
print("Varianza explicada por cada componente:")
for i, var in enumerate(explained_variance_ratio):
    print(f"Componente {i+1}: {var:.4f} ({var*100:.2f}%)")

print("\nVarianza acumulada:")
for i, cum in enumerate(cumulative_variance):
    print(f"Hasta componente {i+1}: {cum:.4f} ({cum*100:.2f}%)")

# Grafica 1: Scree plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, 'o-')
plt.title('Scree Plot (Varianza explicada por componente)')
plt.xlabel('Número de componente')
plt.ylabel('Varianza explicada')
plt.grid(True)

# Grafica 2: Varianza acumulada
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'o-')
plt.axhline(y=0.90, color='r', linestyle='--', label='90%')
plt.axhline(y=0.85, color='orange', linestyle='--', label='85%')
plt.title('Varianza acumulada')
plt.xlabel('Número de componentes')
plt.ylabel('Proporción acumulada de varianza')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%% Implementar PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

#%%
principalDf['Class'] = Features['Class'].values

finalDf = principalDf
#pd.concat([principalDf, Features[['Class']]], axis = 1)

#%% Visualizacion

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = finalDf['Class'].unique()

colors = ['r', 'g', 'b'][:len(targets)]

for target, color in zip(targets,colors):#features
    indicesToKeep = finalDf['Class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)#features)
ax.grid()
plt.show()

#%% Datos Normalizados

scaler = StandardScaler()
X_scaled = scaler.fit_transform(principalComponents)   # escalar antes de PCA

XDf = pd.DataFrame(data = X_scaled
             , columns = ['principal component 1', 'principal component 2'])

XDf['Class'] = Features['Class'].values

#%% Visualizacion

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

Xtargets = XDf['Class'].unique()

colors = ['r', 'g', 'b'][:len(Xtargets)]

for target, color in zip(Xtargets,colors):#features
    indicesToKeep = XDf['Class'] == target
    ax.scatter(XDf.loc[indicesToKeep, 'principal component 1']
               , XDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)#features)
ax.grid()
plt.show()