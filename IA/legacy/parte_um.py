import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram


# Carregar o dataset (corrigindo o nome do arquivo)
df = pd.read_csv("auxiliar.csv")

# Verificar se existem dados nulos em cada coluna
print("Valores nulos por coluna:")
print(df.isnull().sum())

# Verificar se existe algum valor nulo em todo o dataset
print("\nExiste algum valor nulo no dataset?")
print(df.isnull().values.any())

# Mostrar os tipos de dados de cada coluna
print("\nTipos de dados de cada coluna:")
print(df.dtypes)

# Seleciona apenas colunas numéricas
dados = df.select_dtypes(include=[np.number])
print(dados.head())

dados1 = dados.drop(columns=['movieId', 'quant_avaliadores','variancia_nota'], errors='ignore')

print(dados1.head())

#normarlização dos dados - media_nota_r
scaler = StandardScaler()
dados1['media_normalizada'] = scaler.fit_transform(dados1[['media_nota_r']])

dados1= dados1.drop(columns=['media_nota_r'], errors='ignore')
print(dados1[['media_normalizada']].head())


#Calcular a similaridade do cosseno
dados_amostra = dados1.sample(n=100, random_state=42)

matriz_similaridade = cosine_similarity(dados_amostra)
print(matriz_similaridade)
input()


#similaridade em distância
matriz_distancia = 1 - matriz_similaridade
print(matriz_similaridade)
Z = linkage(matriz_distancia, method='average')
plt.figure(figsize=(12,6))
dendrogram(Z)
plt.title("Dendrograma - Similaridade do Cosseno")
plt.xlabel("Objetos")
plt.ylabel("Distância")
plt.show()