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

#%%  Analisis a partir de la matriz de correlación

import seaborn as sns

R = Features[features].corr(min_periods=3)

plt.figure(figsize=(8,8))
sns.heatmap(R, annot = True, fmt='g', cmap='coolwarm')
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

#%% Visualizacion con curvas de nivel

import seaborn as sns

palette_rgb = {
    "Class_1": (255/255, 0/255, 0/255),   # rojo
    "Class_2": (0/255, 255/255, 0/255),   # verde
    "Class_3": (0/255, 0/255, 255/255)    # azul
}

sns.jointplot(
    data=principalDf,
    x="principal component 1", y="principal component 2",
    hue="Class", kind="kde", palette=palette_rgb)

#%%
sns.pairplot(principalDf)


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