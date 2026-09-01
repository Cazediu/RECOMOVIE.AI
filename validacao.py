"""Valida a qualidade da clusterização usada pelo RecomendadorFilmes.

Roda duas verificações:
1. Calcula métricas internas de clusterização (Silhouette, Davies-Bouldin
   e Calinski-Harabasz) para o modelo já treinado (k=25).
2. Repete o K-Means com diferentes valores de k para comparar o quanto o
   número de clusters escolhido influencia o Silhouette Score, plotando
   o resultado em um gráfico.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.cluster import MiniBatchKMeans

from recomendador import RecomendadorFilmes


# -----------------------------
# 1. Métricas de qualidade
# -----------------------------
# Reaproveita o modelo já treinado pelo RecomendadorFilmes (k=25) para
# avaliar a qualidade dos clusters formados.
recomendador = RecomendadorFilmes()
X = recomendador.X
clusters = recomendador.dados["cluster"]

# `sample_size` limita quantos pontos são usados no cálculo do Silhouette
# `random_state` garante que a amostra seja sempre a mesma entre execuções.
silhueta = silhouette_score(X, clusters, sample_size=10000, random_state=42)
# Davies-Bouldin: quanto menor, melhor (clusters mais separados/compactos).
davies = davies_bouldin_score(X, clusters)
# Calinski-Harabasz: quanto maior, melhor (mais separação entre clusters).
calinski = calinski_harabasz_score(X, clusters)

print("VALIDAÇÃO DO MODELO")
print(f"Silhouette Score: {silhueta:.3f}")
print(f"Davies-Bouldin: {davies:.3f}")
print(f"Calinski-Harabasz: {calinski:.2f}")


# -----------------------------
# 2. Teste com vários números de clusters
# -----------------------------
# Treina um K-Means separado para cada valor de k só para comparar o
# Silhouette Score — não altera o modelo usado pela aplicação.
valores_k = [5, 10, 15, 20, 25, 30]
resultados = []

for k in valores_k:
    modelo = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=3)
    grupos = modelo.fit_predict(X)
    score = silhouette_score(X, grupos, sample_size=10000, random_state=42)
    resultados.append(score)
    print(f"k={k}: Silhouette={score:.3f}")

# Gráfico de sensibilidade: ajuda a visualizar qual k tende a formar os
# clusters mais coesos e bem separados.
plt.plot(valores_k, resultados, marker="o")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Sensibilidade ao número de clusters")
plt.grid(True)
plt.show()
