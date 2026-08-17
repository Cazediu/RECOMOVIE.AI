"""
Recomendador simples de filmes com MovieLens.ssss

Recursos:
- gêneros + média normalizada como características;
- MiniBatch K-Means treinado em lotes;
- barras de progresso com tqdm;
- persistência do modelo e do StandardScaler com joblib;
- reutilização automática do modelo salvo;
- similaridade de cosseno dentro do cluster;
- movies.csv opcional para exibir títulos.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


COLUNAS_FIXAS = {
    "movieId",
    "quant_avaliadores",
    "media_nota_r",
    "variancia_nota",
}


class RecomendadorFilmes:
    def __init__(
        self,
        caminho_auxiliar: str | Path,
        caminho_movies: str | Path | None = None,
        caminho_modelo: str | Path = "modelo_recomendador.joblib",
        numero_clusters: int = 25,
        peso_generos: float = 2.0,
        random_state: int = 42,
        batch_size: int = 2048,
        epocas: int = 5,
        retreinar: bool = False,
        verbose: bool = True,
    ) -> None:
        self.caminho_auxiliar = Path(caminho_auxiliar)
        self.caminho_movies = Path(caminho_movies) if caminho_movies else None
        self.caminho_modelo = Path(caminho_modelo)
        self.numero_clusters = numero_clusters
        self.peso_generos = peso_generos
        self.random_state = random_state
        self.batch_size = batch_size
        self.epocas = epocas
        self.retreinar = retreinar
        self.verbose = verbose

        self.df_original: pd.DataFrame
        self.df_modelo: pd.DataFrame
        self.X: pd.DataFrame
        self.colunas_generos: list[str]
        self.scaler: StandardScaler
        self.kmeans: MiniBatchKMeans

        self._mensagem("Lendo e validando o dataset...")
        self._carregar_dados()

        if not self.retreinar and self._carregar_modelo_salvo():
            self._mensagem(f"Modelo carregado de: {self.caminho_modelo}")
        else:
            self._preparar_e_treinar()
            self._salvar_modelo()
            self._mensagem(f"Modelo treinado e salvo em: {self.caminho_modelo}")

    def _mensagem(self, texto: str) -> None:
        if self.verbose:
            tqdm.write(texto)

    def _carregar_dados(self) -> None:
        if not self.caminho_auxiliar.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.caminho_auxiliar}")

        df = pd.read_csv(self.caminho_auxiliar)

        obrigatorias = {"movieId", "quant_avaliadores", "media_nota_r"}
        faltantes = obrigatorias - set(df.columns)
        if faltantes:
            raise ValueError(
                "Colunas obrigatórias ausentes: " + ", ".join(sorted(faltantes))
            )

        self.colunas_generos = [
            coluna
            for coluna in df.columns
            if coluna not in COLUNAS_FIXAS
            and pd.api.types.is_numeric_dtype(df[coluna])
        ]

        if not self.colunas_generos:
            raise ValueError("Nenhuma coluna de gênero foi encontrada.")

        colunas_usadas = [
            "movieId",
            "quant_avaliadores",
            "media_nota_r",
            *self.colunas_generos,
        ]
        df = df.dropna(subset=colunas_usadas).copy()
        df["movieId"] = df["movieId"].astype(int)

        if self.caminho_movies and self.caminho_movies.exists():
            self._mensagem("Carregando títulos de movies.csv...")
            movies = pd.read_csv(self.caminho_movies)
            if {"movieId", "title"}.issubset(movies.columns):
                movies = movies[["movieId", "title"]].copy()
                movies["movieId"] = movies["movieId"].astype(int)
                df = df.merge(movies, on="movieId", how="left")
            else:
                self._mensagem(
                    "movies.csv ignorado: colunas movieId e title não encontradas."
                )

        self.df_original = df.reset_index(drop=True)
        self.df_modelo = self.df_original.copy()

        self._mensagem(
            f"Dataset carregado: {len(df):,} filmes e "
            f"{len(self.colunas_generos)} gêneros."
        )

    def _assinatura_atual(self) -> dict[str, Any]:
        stat = self.caminho_auxiliar.stat()
        return {
            "arquivo": str(self.caminho_auxiliar.resolve()),
            "tamanho": stat.st_size,
            "modificado_ns": stat.st_mtime_ns,
            "linhas": len(self.df_original),
            "colunas_generos": self.colunas_generos,
            "clusters": self.numero_clusters,
            "peso_generos": self.peso_generos,
            "random_state": self.random_state,
            "batch_size": self.batch_size,
            "epocas": self.epocas,
        }

    def _montar_matriz_caracteristicas(self) -> None:
        self._mensagem("Preparando a matriz de características...")

        self.df_modelo["media_normalizada"] = self.scaler.transform(
            self.df_modelo[["media_nota_r"]]
        )

        generos = (
            self.df_modelo[self.colunas_generos].astype(np.float32)
            * self.peso_generos
        )
        media = self.df_modelo[["media_normalizada"]].astype(np.float32)
        self.X = pd.concat([generos, media], axis=1)

        self._mensagem(
            f"Matriz pronta: {self.X.shape[0]:,} filmes × "
            f"{self.X.shape[1]} características."
        )

    def _preparar_e_treinar(self) -> None:
        if self.numero_clusters < 2:
            raise ValueError("O número de clusters deve ser pelo menos 2.")
        if self.epocas < 1:
            raise ValueError("O número de épocas deve ser pelo menos 1.")
        if self.batch_size < self.numero_clusters:
            raise ValueError(
                "O batch_size deve ser maior ou igual ao número de clusters."
            )

        self._mensagem("Ajustando o StandardScaler...")
        self.scaler = StandardScaler()
        self.scaler.fit(self.df_modelo[["media_nota_r"]])
        self._montar_matriz_caracteristicas()

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.numero_clusters,
            random_state=self.random_state,
            batch_size=self.batch_size,
            n_init=1,
            reassignment_ratio=0.01,
        )

        X_np = self.X.to_numpy(dtype=np.float32, copy=False)
        n = len(X_np)
        lotes_por_epoca = int(np.ceil(n / self.batch_size))
        total_lotes = self.epocas * lotes_por_epoca
        rng = np.random.default_rng(self.random_state)

        self._mensagem(
            f"Treinando MiniBatch K-Means: {self.numero_clusters} clusters, "
            f"{self.epocas} épocas e lotes de {self.batch_size}."
        )

        with tqdm(
            total=total_lotes,
            desc="Treinamento",
            unit="lote",
            disable=not self.verbose,
        ) as barra:
            for epoca in range(self.epocas):
                indices = rng.permutation(n)
                for inicio in range(0, n, self.batch_size):
                    fim = min(inicio + self.batch_size, n)
                    lote = X_np[indices[inicio:fim]]
                    self.kmeans.partial_fit(lote)
                    barra.set_postfix(
                        epoca=f"{epoca + 1}/{self.epocas}",
                        inercia=f"{self.kmeans.inertia_:.2f}",
                    )
                    barra.update(1)

        self._mensagem("Atribuindo clusters a todos os filmes...")
        clusters = self._prever_clusters_em_lotes(X_np)
        self.df_original["cluster"] = clusters
        self.df_modelo["cluster"] = clusters

    def _prever_clusters_em_lotes(
        self,
        X_np: np.ndarray | None = None,
    ) -> np.ndarray:
        if X_np is None:
            X_np = self.X.to_numpy(dtype=np.float32, copy=False)

        partes: list[np.ndarray] = []
        for inicio in tqdm(
            range(0, len(X_np), self.batch_size),
            desc="Inferência dos clusters",
            unit="lote",
            disable=not self.verbose,
        ):
            fim = min(inicio + self.batch_size, len(X_np))
            partes.append(self.kmeans.predict(X_np[inicio:fim]))

        return np.concatenate(partes)

    def _salvar_modelo(self) -> None:
        self._mensagem("Salvando modelo treinado...")
        self.caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "assinatura": self._assinatura_atual(),
                "scaler": self.scaler,
                "kmeans": self.kmeans,
            },
            self.caminho_modelo,
        )

    def _carregar_modelo_salvo(self) -> bool:
        if not self.caminho_modelo.exists():
            self._mensagem("Nenhum modelo salvo foi encontrado.")
            return False

        self._mensagem(f"Tentando carregar: {self.caminho_modelo}")
        try:
            pacote = joblib.load(self.caminho_modelo)
            if pacote.get("assinatura") != self._assinatura_atual():
                self._mensagem(
                    "O modelo salvo não corresponde aos dados ou parâmetros atuais."
                )
                return False

            self.scaler = pacote["scaler"]
            self.kmeans = pacote["kmeans"]
            self._montar_matriz_caracteristicas()

            clusters = self._prever_clusters_em_lotes()
            self.df_original["cluster"] = clusters
            self.df_modelo["cluster"] = clusters
            return True
        except (OSError, KeyError, ValueError, TypeError) as erro:
            self._mensagem(f"Falha ao carregar o modelo salvo: {erro}")
            return False

    def recomendar(
        self,
        movie_id: int,
        quantidade: int = 10,
        minimo_avaliacoes: int = 50,
    ) -> pd.DataFrame:
        encontrados = self.df_original.index[
            self.df_original["movieId"] == int(movie_id)
        ]
        if len(encontrados) == 0:
            raise ValueError(f"movieId {movie_id} não encontrado.")

        indice_filme = int(encontrados[0])
        cluster_filme = int(self.df_original.at[indice_filme, "cluster"])

        mascara = (
            (self.df_original["cluster"] == cluster_filme)
            & (self.df_original["movieId"] != int(movie_id))
            & (self.df_original["quant_avaliadores"] >= minimo_avaliacoes)
        )
        indices = self.df_original.index[mascara].to_numpy()

        self._mensagem(
            f"Cluster do filme: {cluster_filme}. "
            f"Candidatos após o filtro: {len(indices):,}."
        )

        if len(indices) == 0:
            return pd.DataFrame()

        consulta = self.X.loc[[indice_filme]].to_numpy(
            dtype=np.float32, copy=False
        )

        partes: list[np.ndarray] = []
        for inicio in tqdm(
            range(0, len(indices), self.batch_size),
            desc="Similaridade de cosseno",
            unit="lote",
            disable=not self.verbose,
        ):
            fim = min(inicio + self.batch_size, len(indices))
            indices_lote = indices[inicio:fim]
            candidatos = self.X.loc[indices_lote].to_numpy(
                dtype=np.float32, copy=False
            )
            partes.append(cosine_similarity(consulta, candidatos)[0])

        resultado = self.df_original.loc[indices].copy()
        resultado["similaridade"] = np.concatenate(partes)
        resultado = resultado.sort_values(
            ["similaridade", "media_nota_r", "quant_avaliadores"],
            ascending=[False, False, False],
        ).head(quantidade)

        colunas = [
            "movieId",
            "media_nota_r",
            "quant_avaliadores",
            "cluster",
            "similaridade",
        ]
        if "title" in resultado.columns:
            colunas.insert(1, "title")
        return resultado[colunas].reset_index(drop=True)

    def descrever_filme(self, movie_id: int) -> str:
        encontrados = self.df_original[
            self.df_original["movieId"] == int(movie_id)
        ]
        if encontrados.empty:
            return f"movieId {movie_id}"

        filme = encontrados.iloc[0]
        if "title" in filme.index and pd.notna(filme["title"]):
            return f"{filme['title']} (movieId={movie_id})"

        generos = [g for g in self.colunas_generos if filme[g] == 1]
        return (
            f"movieId={movie_id} | média={filme['media_nota_r']:.3f} | "
            f"avaliações={int(filme['quant_avaliadores'])} | "
            f"gêneros={', '.join(generos) if generos else 'não informados'}"
        )


def executar() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recomendador com MiniBatch K-Means, similaridade de cosseno, "
            "persistência e barras de progresso."
        )
    )
    parser.add_argument("--dados", default="auxiliar.csv")
    parser.add_argument("--movies", default=None)
    parser.add_argument("--modelo", default="modelo_recomendador.joblib")
    parser.add_argument("--movie-id", type=int, default=1)
    parser.add_argument("--quantidade", type=int, default=10)
    parser.add_argument("--min-avaliacoes", type=int, default=50)
    parser.add_argument("--clusters", type=int, default=25)
    parser.add_argument("--peso-generos", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epocas", type=int, default=5)
    parser.add_argument(
        "--retreinar",
        action="store_true",
        help="Ignora o modelo salvo e treina novamente.",
    )
    parser.add_argument(
        "--silencioso",
        action="store_true",
        help="Oculta mensagens e barras de progresso.",
    )
    args = parser.parse_args()

    recomendador = RecomendadorFilmes(
        caminho_auxiliar=args.dados,
        caminho_movies=args.movies,
        caminho_modelo=args.modelo,
        numero_clusters=args.clusters,
        peso_generos=args.peso_generos,
        batch_size=args.batch_size,
        epocas=args.epocas,
        retreinar=args.retreinar,
        verbose=not args.silencioso,
    )

    print("\nFilme consultado:")
    print(recomendador.descrever_filme(args.movie_id))

    recomendacoes = recomendador.recomendar(
        movie_id=args.movie_id,
        quantidade=args.quantidade,
        minimo_avaliacoes=args.min_avaliacoes,
    )

    print("\nRecomendações:")
    if recomendacoes.empty:
        print("Nenhum candidato encontrado com os filtros informados.")
    else:
        print(recomendacoes.to_string(index=False))


if __name__ == "__main__":
    executar()
