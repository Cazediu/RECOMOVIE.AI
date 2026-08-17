from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FIXED_COLUMNS = {'movieId','quant_avaliadores','media_nota_r','variancia_nota','title'}

class MovieRepository:
    def __init__(self, data_path: Path, movies_path: Path | None = None):
        self.data_path = Path(data_path)
        self.movies_path = Path(movies_path) if movies_path else None
        self.df = self._load()
        self.genre_columns = [
            c for c in self.df.columns
            if c not in FIXED_COLUMNS and pd.api.types.is_numeric_dtype(self.df[c])
        ]
        if not self.genre_columns:
            raise ValueError('Nenhuma coluna de gênero numérica foi encontrada.')
        self.df['movieId'] = self.df['movieId'].astype(int)
        self.df = self.df.reset_index(drop=True)
        self.by_id = pd.Series(self.df.index, index=self.df['movieId']).to_dict()

    def _load(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f'Dataset não encontrado: {self.data_path}')
        df = pd.read_csv(self.data_path)
        required = {'movieId','quant_avaliadores','media_nota_r'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'Colunas obrigatórias ausentes: {sorted(missing)}')
        if self.movies_path and self.movies_path.exists():
            movies = pd.read_csv(self.movies_path, usecols=lambda c: c in {'movieId','title','genres'})
            if {'movieId','title'}.issubset(movies.columns):
                movies['movieId'] = movies['movieId'].astype(int)
                df['movieId'] = df['movieId'].astype(int)
                df = df.merge(movies[['movieId','title','genres']], on='movieId', how='left', suffixes=('','_movies'))
        columns = ['movieId','quant_avaliadores','media_nota_r'] + [
            c for c in df.columns if c not in {'movieId','quant_avaliadores','media_nota_r','variancia_nota','title','genres'}
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        df = df.dropna(subset=columns).copy()
        if 'title' not in df.columns:
            df['title'] = df['movieId'].map(lambda x: f'Filme #{int(x)}')
        if 'genres' not in df.columns:
            df['genres'] = ''
        return df

    def index_from_movie_id(self, movie_id: int) -> int:
        try:
            return int(self.by_id[int(movie_id)])
        except KeyError:
            raise ValueError(f'Filme não encontrado: {movie_id}') from None

    def find_titles(self, text: str, limit: int = 50) -> pd.DataFrame:
        query = str(text).strip().lower()
        if not query:
            return self.df.head(limit)[['movieId','title','genres','media_nota_r','quant_avaliadores']]
        mask = self.df['title'].fillna('').str.lower().str.contains(query, regex=False)
        return self.df.loc[mask, ['movieId','title','genres','media_nota_r','quant_avaliadores']].head(limit)
