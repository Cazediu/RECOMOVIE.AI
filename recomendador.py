"""Núcleo do sistema de recomendação de filmes.

A abordagem usada tem duas etapas:
1. Agrupar (clusterizar) todos os filmes em 25 grupos com K-Means, usando
   gêneros e nota média como características.
2. Dentro do grupo do filme escolhido, comparar os filmes por similaridade
   de cosseno para achar os mais parecidos.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


class RecomendadorFilmes:
    """Recomenda filmes usando K-Means + similaridade de cosseno."""

    def __init__(self, arquivo_dados="auxiliar.csv", arquivo_filmes="movies.csv"):
        # `auxiliar.csv` já vem pré-processado (uma linha por filme, com
        # gêneros em one-hot e estatísticas de avaliação agregadas).
        pasta = Path(__file__).parent
        self.dados = pd.read_csv(pasta / arquivo_dados)
        filmes = pd.read_csv(pasta / arquivo_filmes)

        # Traz o título do filme (que não está no auxiliar.csv) via merge
        # pelo movieId, mantendo todas as linhas de `dados` (left join).
        self.dados = self.dados.merge(filmes[["movieId", "title"]], on="movieId", how="left")

        # Todas as colunas que não são metadados fixos representam os
        # gêneros do filme (uma coluna binária por gênero).
        colunas_fixas = {"movieId", "quant_avaliadores", "media_nota_r", "variancia_nota", "title"}
        self.generos = [c for c in self.dados.columns if c not in colunas_fixas]

        self.scaler = StandardScaler()
        # MiniBatchKMeans é usado em vez do KMeans padrão por ser mais
        # rápido em bases grandes (processa os dados em lotes/batches).
        self.modelo = MiniBatchKMeans(n_clusters=25, random_state=42, batch_size=2048, n_init=3)

        self._preparar_dados()
        self._treinar()

    def _preparar_dados(self):
        """Monta a matriz de características (`self.X`) usada pelo K-Means."""
        # Padroniza a nota média (média 0, desvio padrão 1) para que ela
        # fique na mesma escala numérica dos gêneros (que são 0 ou 1).
        media = self.scaler.fit_transform(self.dados[["media_nota_r"]])
        generos = self.dados[self.generos].to_numpy(dtype=float)

        # Multiplicar os gêneros por 2 faz o algoritmo priorizar a
        # semelhança de gênero acima da nota média ao formar os clusters.
        self.X = np.hstack((generos * 2, media))

    def _treinar(self):
        """Roda o K-Means e guarda o grupo (cluster) de cada filme."""
        self.dados["cluster"] = self.modelo.fit_predict(self.X)

    def buscar_filmes(self, texto):
        """Retorna até 20 filmes cujo título contém o texto pesquisado.

        Usado pela barra de busca com autocomplete: recebe o que o usuário
        já digitou e devolve as sugestões correspondentes.
        """
        texto = texto.strip().lower()
        if not texto:
            return pd.DataFrame()

        return self.dados[self.dados["title"].str.lower().str.contains(texto, na=False)] \
            [["movieId", "title"]].head(20)

    def recomendar(self, movie_id, quantidade=10, minimo_avaliacoes=50):
        """Retorna filmes parecidos com o filme escolhido.

        Passos: localiza o cluster do filme, filtra candidatos do mesmo
        cluster com avaliações suficientes e ordena por similaridade de
        cosseno (e, em caso de empate, pela nota média).
        """
        filme = self.dados[self.dados["movieId"] == movie_id]

        if filme.empty:
            return pd.DataFrame()

        indice = filme.index[0]
        cluster = self.dados.loc[indice, "cluster"]

        # Só considera filmes do mesmo grupo, diferentes do escolhido e
        # com avaliações suficientes para a nota média ser confiável.
        candidatos = self.dados[
            (self.dados["cluster"] == cluster)
            & (self.dados["movieId"] != movie_id)
            & (self.dados["quant_avaliadores"] >= minimo_avaliacoes)
        ].copy()

        if candidatos.empty:
            return candidatos

        indices = candidatos.index
        vetor_filme = self.X[indice].reshape(1, -1)
        vetores = self.X[indices]

        # Distância cosseno = 1 - similaridade de cosseno; portanto
        # similaridade = 1 - distância. Quanto mais próximo de 1, mais
        # parecido o filme é do escolhido.
        distancias = pairwise_distances(vetor_filme, vetores, metric="cosine")[0]
        candidatos["similaridade"] = 1 - distancias

        candidatos = candidatos.sort_values(
            ["similaridade", "media_nota_r"], ascending=False
        ).head(quantidade)

        # Converte para porcentagem só para facilitar a exibição na tela.
        candidatos["similaridade"] *= 100
        return candidatos[["title", "media_nota_r", "quant_avaliadores", "similaridade"]]

    def salvar(self, arquivo="modelo_recomovie.joblib"):
        """Salva apenas o que é necessário para o deploy.

        Não salva `self.dados`/`self.X` porque eles são recriados a partir
        dos CSVs sempre que a aplicação sobe; salvar só o modelo treinado
        evita duplicar dados grandes no arquivo .joblib.
        """
        joblib.dump(
            {
                "modelo": self.modelo,
                "scaler": self.scaler,
                "generos": self.generos,
            },
            Path(__file__).parent / arquivo,
        )


if __name__ == "__main__":
    # Permite treinar e salvar o modelo rodando `python recomendador.py`.
    recomendador = RecomendadorFilmes()
    recomendador.salvar()
    print("Modelo treinado e salvo com sucesso!")
