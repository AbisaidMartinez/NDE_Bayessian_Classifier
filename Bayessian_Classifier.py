# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:35:06 2025

@author: qbo28
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB #MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 2. Load and Preprocess Data: Load your dataset and preprocess it. For this example, we’ll assume you have a CSV file with two columns: text and label.

#%% View first few rows
print(finalDf.head())

#%% Separate features and labels

# Transform text data into feature vectors
X = finalDf[['principal component 1', 'principal component 2', 'principal component 3']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # escalar antes de PCA

le = LabelEncoder()
y_encoded = le.fit_transform(Features['Class'])#y)  # → 0, 1, 2

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#%% Initialize the model
Gauss_classifier = GaussianNB()
#MultinomialNB()

# Train the model
Gauss_classifier.fit(X_train, y_train)

y_pred = Gauss_classifier.predict(X_test)

#%%

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

classes = le.classes_  # ← orden que tú quieras

plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='RdBu',
            xticklabels=classes, yticklabels=classes, linewidths=.5, square=True)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predict")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

#%% Visualizacion de los datos de prueba 2D

from matplotlib.colors import ListedColormap

# Datos de prueba (ya transformados con PCA a 2 componentes)
X_set, y_set = X_test.values, y_test#, y_test.ravel()  # Asegúrate de que X_test sea el conjunto en espacio PCA (2D)

# Crear la malla para visualizar la frontera de decisión
X1, X2, X3 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, 
                               stop=X_set[:, 0].max() + 1, step=0.1),
                     np.arange(start=X_set[:, 1].min() - 1, 
                               stop=X_set[:, 1].max() + 1, step=0.1),
                     np.arange(start=X_set[:, 2].min() - 1, 
                               stop=X_set[:, 2].max() + 1, step=0.1))

#%% Predecir sobre toda la malla

Z = Gauss_classifier.predict(np.array([X1.ravel(), X2.ravel(), X3.ravel()]).T)
Z = Z.reshape(X1.shape)

from mpl_toolkits.mplot3d import Axes3D

# Colores para las 3 clases (fondo y puntos)
colors = ['#FF9999', '#99FF99', '#9999FF']  # rojo claro, verde claro, azul claro
colors_dark = ['red', 'green', 'blue']       # para los puntos
cmap_background = ListedColormap(colors)
cmap_points = ListedColormap(colors_dark)

# Crear figura 3D
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Graficar la malla predicha
ax.scatter(X1.ravel(), X2.ravel(), X3.ravel(),
           c=Z.ravel(), cmap=cmap_background, alpha=0.1, s=5)

# Graficar los puntos reales de prueba
for i, j in enumerate(np.unique(y_set)):
    ax.scatter(X_set[y_set == j, 0],
               X_set[y_set == j, 1],
               X_set[y_set == j, 2],
               c=cmap_points(i), label=f'Class_{j+1}',
               edgecolors='black', s=50)

# Etiquetas y detalles
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend(title='Clases')
plt.title('Clasificador GaussNB en espacio PCA (3D)')
plt.show()