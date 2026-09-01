"""Núcleo do sistema de recomendação de filmes (Recomovie AI).

A abordagem utilizada possui duas etapas principais:
1. Agrupar (clusterizar) os filmes em 25 grupos usando MiniBatch K-Means,
   tendo como features as colunas binárias de gênero e a nota média padronizada.
2. Dentro do mesmo grupo (cluster) do filme selecionado, calcular a similaridade
   de cosseno para ranquear e recomendar os filmes mais próximos.
"""

# Importa Path para manipulação de caminhos de arquivos de forma multiplataforma
from pathlib import Path

# Importa joblib para persistência e carregamento de modelos treinados
import joblib

# Importa numpy para operações matriciais e manipulação de arrays numéricos
import numpy as np

# Importa pandas para manipulação de dados estruturados em DataFrames
import pandas as pd

# Importa o algoritmo de clusterização MiniBatchKMeans do Scikit-Learn
from sklearn.cluster import MiniBatchKMeans

# Importa função para cálculo de matriz de distâncias de cosseno
from sklearn.metrics import pairwise_distances

# Importa StandardScaler para padronização de variáveis numéricas (média 0, desvio 1)
from sklearn.preprocessing import StandardScaler


class RecomendadorFilmes:
    """Classe responsável pelo treinamento, indexação e recomendação de filmes."""

    def __init__(self, arquivo_dados="auxiliar.csv", arquivo_filmes="movies.csv", arquivo_links="links.csv"):
        # Obtém o diretório onde este arquivo de script está localizado
        pasta = Path(__file__).parent

        # Carrega o CSV auxiliar pré-processado contendo estatísticas de notas e gêneros em one-hot
        self.dados = pd.read_csv(pasta / arquivo_dados)

        # Carrega o CSV de filmes contendo os títulos e IDs
        filmes = pd.read_csv(pasta / arquivo_filmes)

        # Realiza o merge trazendo o título do filme para a base principal através do movieId
        self.dados = self.dados.merge(filmes[["movieId", "title"]], on="movieId", how="left")

        # Define o caminho para o arquivo links.csv contendo os IDs do IMDb e TMDb
        caminho_links = pasta / arquivo_links

        # Verifica se o arquivo links.csv existe no diretório
        if caminho_links.exists():
            # Lê o arquivo links.csv
            links = pd.read_csv(caminho_links)
            # Filtra apenas as colunas relevantes presentes no arquivo
            colunas_links = [c for c in ["movieId", "imdbId", "tmdbId"] if c in links.columns]
            # Realiza o merge dos identificadores imdbId e tmdbId com base no movieId
            self.dados = self.dados.merge(links[colunas_links], on="movieId", how="left")
        else:
            # Caso o arquivo não exista, inicializa as colunas com valor nulo
            self.dados["imdbId"] = None
            self.dados["tmdbId"] = None

        # Define o conjunto de colunas de metadados fixos que não representam features de gênero
        colunas_fixas = {"movieId", "quant_avaliadores", "media_nota_r", "variancia_nota", "title", "imdbId", "tmdbId"}

        # Identifica todas as colunas de gêneros (todas exceto as colunas fixas)
        self.generos = [c for c in self.dados.columns if c not in colunas_fixas]

        # Instancia o escalador para normalizar as notas médias
        self.scaler = StandardScaler()

        # Instancia o MiniBatchKMeans com 25 clusters, semente fixa (42) e tamanho de lote de 2048
        self.modelo = MiniBatchKMeans(n_clusters=25, random_state=42, batch_size=2048, n_init=3)

        # Prepara a matriz de características numéricas X
        self._preparar_dados()

        # Executa o treinamento do modelo de clusterização
        self._treinar()

    def _preparar_dados(self):
        """Monta a matriz de características numéricas (self.X) usada pelo K-Means."""
        # Padroniza a nota média para ficar na mesma escala das features binárias
        media = self.scaler.fit_transform(self.dados[["media_nota_r"]])

        # Converte as colunas binárias de gênero em um array numpy de floats
        generos = self.dados[self.generos].to_numpy(dtype=float)

        # Multiplica o peso dos gêneros por 2 para priorizar afinidade de gênero sobre a nota
        # Concatena horizontalmente os gêneros ponderados e a nota média padronizada
        self.X = np.hstack((generos * 2, media))

    def _treinar(self):
        """Treina o modelo K-Means e armazena o cluster correspondente a cada filme."""
        # Executa o fit e atribui o número do cluster na coluna 'cluster' do DataFrame
        self.dados["cluster"] = self.modelo.fit_predict(self.X)

    def buscar_filmes(self, texto):
        """Retorna até 20 filmes cujo título contenha o texto pesquisado pelo usuário.

        Usado pelo autocomplete da barra de pesquisa no Streamlit.
        """
        # Remove espaços em branco nas extremidades e converte o termo de busca para minúsculo
        texto = texto.strip().lower()

        # Se o texto de busca estiver vazio, retorna um DataFrame vazio
        if not texto:
            return pd.DataFrame()

        # Filtra os filmes cujo título contém o texto pesquisado (sem diferenciar maiúsculas/minúsculas)
        # Retorna os primeiros 20 resultados contendo movieId, title, imdbId e tmdbId
        return self.dados[self.dados["title"].str.lower().str.contains(texto, na=False)] \
            [["movieId", "title", "imdbId", "tmdbId"]].head(20)

    def obter_filme(self, movie_id):
        """Retorna os dados completos de um filme específico a partir do seu movieId."""
        # Filtra a linha correspondente ao movieId informado
        filmes = self.dados[self.dados["movieId"] == movie_id]

        # Se não encontrar o filme, retorna None
        if filmes.empty:
            return None

        # Retorna a primeira linha encontrada como uma Series do pandas
        return filmes.iloc[0]

    def recomendar(self, movie_id, quantidade=10, minimo_avaliacoes=50):
        """Gera recomendações de filmes parecidos com base em similaridade de cosseno dentro do mesmo cluster.

        Parâmetros:
        - movie_id: Identificador do filme base selecionado pelo usuário.
        - quantidade: Número de recomendações a retornar (padrão 10).
        - minimo_avaliacoes: Quantidade mínima de avaliadores para garantir confiabilidade estatística (padrão 50).
        """
        # Localiza o filme de referência na base de dados
        filme = self.dados[self.dados["movieId"] == movie_id]

        # Se o filme não existir no dataset, retorna um DataFrame vazio
        if filme.empty:
            return pd.DataFrame()

        # Obtém o índice da linha do filme no DataFrame
        indice = filme.index[0]

        # Obtém o número do cluster atribuído a esse filme
        cluster = self.dados.loc[indice, "cluster"]

        # Filtra candidatos que pertencem ao mesmo cluster, diferentes do filme atual
        # e que possuam pelo menos o número mínimo estipulado de avaliações
        candidatos = self.dados[
            (self.dados["cluster"] == cluster)
            & (self.dados["movieId"] != movie_id)
            & (self.dados["quant_avaliadores"] >= minimo_avaliacoes)
        ].copy()

        # Se não houver candidatos suficientes no mesmo grupo, retorna DataFrame vazio
        if candidatos.empty:
            return candidatos

        # Obtém os índices dos candidatos filtrados
        indices = candidatos.index

        # Extrai o vetor de características do filme selecionado no formato 2D (1, n_features)
        vetor_filme = self.X[indice].reshape(1, -1)

        # Extrai a matriz com os vetores de características de todos os candidatos
        vetores = self.X[indices]

        # Calcula a distância de cosseno entre o filme selecionado e todos os candidatos
        distancias = pairwise_distances(vetor_filme, vetores, metric="cosine")[0]

        # Converte a distância em similaridade percentual (similaridade = (1 - distância) * 100)
        candidatos.loc[:, "similaridade"] = (1 - distancias) * 100

        # Ordena os candidatos de forma decrescente pela similaridade e pela nota média em caso de empate
        # Seleciona os top N filmes conforme a quantidade solicitada
        candidatos = candidatos.sort_values(
            ["similaridade", "media_nota_r"], ascending=False
        ).head(quantidade)

        # Retorna apenas as colunas essenciais para exibição visual
        return candidatos[["movieId", "title", "media_nota_r", "quant_avaliadores", "similaridade", "imdbId", "tmdbId"]]

    def salvar(self, arquivo="modelo_recomovie.joblib"):
        """Exporta e salva os artefatos treinados do modelo para arquivo .joblib."""
        # Salva o modelo, o scaler e a lista de gêneros em um dicionário serializado
        joblib.dump(
            {
                "modelo": self.modelo,
                "scaler": self.scaler,
                "generos": self.generos,
            },
            Path(__file__).parent / arquivo,
        )


# Bloco de execução direta via linha de comando (python recomendador.py)
if __name__ == "__main__":
    # Instancia a classe do recomendador (o que realiza a preparação e o treinamento)
    recomendador = RecomendadorFilmes()
    # Salva o modelo treinado em disco
    recomendador.salvar()
    # Exibe mensagem de confirmação
    print("Modelo treinado e salvo com sucesso!")
