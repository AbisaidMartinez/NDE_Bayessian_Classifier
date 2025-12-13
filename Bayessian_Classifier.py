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
X = finalDf[['principal component 1', 'principal component 2']]

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

#%% Visualizacion de los datos de prueba

from matplotlib.colors import ListedColormap

# Datos de prueba (ya transformados con PCA a 2 componentes)
X_set, y_set = X_test.values, y_test#, y_test.ravel()  # Asegúrate de que X_test sea el conjunto en espacio PCA (2D)

# Crear la malla para visualizar la frontera de decisión
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, 
                               stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, 
                               stop=X_set[:, 1].max() + 1, step=0.01))

# Predecir sobre toda la malla
Z = Gauss_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
Z = Z.reshape(X1.shape)

# Colores para las 3 clases (fondo y puntos)
colors = ['#FF9999', '#99FF99', '#9999FF']  # rojo claro, verde claro, azul claro
colors_dark = ['red', 'green', 'blue']       # para los puntos
cmap_background = ListedColormap(colors)
cmap_points = ListedColormap(colors_dark)

# Graficar la frontera de decisión
plt.figure(figsize=(10, 8))
plt.contourf(X1, X2, Z, alpha=0.75, cmap=cmap_background)

# Graficar los puntos de prueba
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=cmap_points(i), label=f'Class_{j+1}', edgecolors='black', s=50)

# Detalles del gráfico
plt.title('Clasificador Naive Bayes Gaussiano (Conjunto de Prueba - PCA)', fontsize=14)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Clases')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()