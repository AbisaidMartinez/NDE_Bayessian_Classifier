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
        f_max = 1e7  # ejemplo: hasta 10 MHz
        mask = f <= f_max

        # Recortar tanto f como Zxx_norm
        f_cut = f[mask]
        
        # 1. Convertir a dB (escala logarítmica)
        Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)  # +1e-10 para evitar log(0)
        
        # 2. Recortar valores extremos para reducir el padding
        Zxx_db = np.clip(Zxx_db, -60, 0)  # recortar entre -60dB y 0dB
        
        # 3. Normalizar a [0, 255]
        Zxx_norm = (Zxx_db - Zxx_db.min()) / (Zxx_db.max() - Zxx_db.min()) * 255
        Zxx_cut = Zxx_norm[mask, :]

        image = Zxx_cut.astype(np.uint8)
        
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
        # plt.pcolormesh(t_stft, f_cut, image, cmap='gray', shading='gouraud')
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

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

#%%
principalDf['Class'] = Features['Class'].values

finalDf = principalDf
#pd.concat([principalDf, Features[['Class']]], axis = 1)

#%%  Analisis a partir de la matriz de correlación

import seaborn as sns

r = principalDf[['principal component 1', 'principal component 2', 'principal component 3']].corr(min_periods=3)

plt.figure()
sns.heatmap(r, annot = True, fmt='.2f', cmap='coolwarm',
            linewidths=.5, square=True)         
plt.title("Correlation Matrix")
plt.show()


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

#%% Grafica 3D

from mpl_toolkits.mplot3d import Axes3D

# 2. Create the figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # or ax = plt.axes(projection='3d')

# 3. Plot the scatter points
targets = principalDf['Class'].unique()

colors = ['red', 'blue', 'green']

for target, color in zip(targets, colors):
    subset = principalDf[principalDf['Class'] == target]
    ax.scatter(subset['principal component 1'],
               subset['principal component 2'],
               subset['principal component 3'],
               c=color, label=target, s=50)

# Optional: Add labels
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend(targets)
ax.grid()
# 4. Display the plot
plt.show()

#%% Visualizacion con curvas de nivel
import seaborn as sns

sns.jointplot(
    data=principalDf,
    x="principal component 1", y="principal component 2",
    hue="Class", kind="kde"
)

#%%
sns.pairplot(principalDf)

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

#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

# Supongamos que principalDf tiene columnas: 'principal component 1', 'principal component 2', 'principal component 3', 'Class'

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Colores para las clases
colors = ['red', 'blue', 'green', 'orange', 'purple']
targets = principalDf['Class'].unique()

for target, color in zip(targets, colors):
    subset = principalDf[principalDf['Class'] == target]
    
    # Graficar los puntos de cada clase
    ax.scatter(subset['principal component 1'],
               subset['principal component 2'],
               subset['principal component 3'],
               c=color, label=target, s=40, alpha=0.6)
    
    # Calcular media y covarianza
    data = subset[['principal component 1',
                   'principal component 2',
                   'principal component 3']].values
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    
    # Autovalores y autovectores
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Generar puntos de esfera
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack((x, y, z), axis=-1)
    
    # Escalar por autovalores (radio ~ sqrt(eigval))
    radii = np.sqrt(eigvals)
    ellipsoid = sphere @ np.diag(radii) @ eigvecs.T + mean
    
    # Dibujar superficie del elipsoide
    ax.plot_wireframe(ellipsoid[:,:,0],
                      ellipsoid[:,:,1],
                      ellipsoid[:,:,2],
                      color=color, alpha=0.2)

# Etiquetas y leyenda
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
plt.title("PCA 3D con elipsoides gaussianos por clase")
plt.show()
