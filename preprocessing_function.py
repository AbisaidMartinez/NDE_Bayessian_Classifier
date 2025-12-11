# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 10:49:16 2025

@author: qbo28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.feature import graycomatrix, graycoprops

def Extract_Features(filename,
        n_signals=50,
        widths=np.arange(1, 31),
        wavelet='mexh',
        distances=[1],
        angles=[0]
    ):
    """
    Extrae características GLCM a partir de la transformada wavelet continua (CWT) 
    de múltiples señales almacenadas en un archivo CSV.

    Parámetros:
        filename: ruta del archivo .csv
        n_signals: número de señales (columnas) a procesar
        widths: anchos escalares para la CWT
        wavelet: tipo de wavelet de pywt
        distances: distancias para el GLCM
        angles: ángulos para el GLCM en radianes
    
    Retorna:
        DataFrame con contrast, homogeneity, energy, correlation y dissimilarity
    """
    
        # --- Cargar archivo ---
    df = pd.read_csv(filename)
    y = df.iloc[:, 1:n_signals+1]   # señales en columnas 1..n
    
    # --- CWT ---
    CWT_all = np.zeros((n_signals, len(widths), y.shape[0]))
    
    for i in range(n_signals):
        cwt_result = pywt.cwt(y.iloc[:, i], widths, wavelet)[0]
        CWT_all[i, :, :] = np.abs(cwt_result)
    
    # --- Inicializar características ---
    contrast = np.zeros(n_signals)
    homogeneity = np.zeros(n_signals)
    energy = np.zeros(n_signals)
    correlation = np.zeros(n_signals)
    dissimilarity = np.zeros(n_signals)
    
    # --- GLCM ---
    for i in range(n_signals):
        image = CWT_all[i, :, :]
        image = (image / image.max() * 255).astype(np.uint8)
    
        glcm = graycomatrix(image,
                            distances=distances,
                            angles=angles,
                            levels=256,
                            symmetric=True,
                            normed=True)
    
        contrast[i]      = graycoprops(glcm, 'contrast')
        homogeneity[i]   = graycoprops(glcm, 'homogeneity')
        energy[i]        = graycoprops(glcm, 'energy')
        correlation[i]   = graycoprops(glcm, 'correlation')
        dissimilarity[i] = graycoprops(glcm, 'dissimilarity')
    
    # --- Matriz final ---
    Features = pd.DataFrame({
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'dissimilarity': dissimilarity
    }, index=[f"signal {i+1}" for i in range(n_signals)])
    
    return Features

Feature01 = Extract_Features("Class_1_5MHz.csv")
Feature02 = Extract_Features("Class_1_15MHz.csv")

Features01 = pd.concat([Feature01, Feature02])
Features01['Class'] = 'Class_1'

Feature03 = Extract_Features("Class_2_5MHz.csv")
Feature04 = Extract_Features("Class_2_15MHz.csv")

Features02 = pd.concat([Feature03, Feature04])
Features02['Class'] = 'Class_2'

Feature05 = Extract_Features("Class_3_5MHz.csv")
Feature06 = Extract_Features("Class_3_15MHz.csv")

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

# Separating out the features
x = Features.loc[:, features].values

# Separating out the target
y = Features.loc[:,['Class']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

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
