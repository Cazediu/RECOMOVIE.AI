# Recomovie AI - Etapa 3

Projeto simples de recomendação de filmes usando clusterização.

## Tecnologias
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib


## Como o projeto funciona

1. Os filmes são separados em 25 grupos usando MiniBatch K-Means.
2. Os gêneros e a média das avaliações são usados como características.
3. Quando o usuário escolhe um filme, o sistema encontra o grupo dele.
4. Dentro desse grupo, os filmes são comparados usando similaridade de cosseno.
5. Os filmes mais parecidos são mostrados como recomendação.

## Validação

A validação usa duas estratégias pedidas para clusterização:

- métricas internas: Silhouette, Davies-Bouldin e Calinski-Harabasz;
- teste de diferentes valores de k com gráfico de Silhouette.

## Limitações

O sistema depende das características presentes no dataset. Filmes com poucas avaliações podem ter resultados menos confiáveis. Além disso, o agrupamento representa apenas os padrões encontrados nos dados usados no treinamento.
