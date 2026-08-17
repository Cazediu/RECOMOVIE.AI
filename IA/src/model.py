from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib, json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

class ClusterModel:
    def __init__(self, n_clusters=25, genre_weight=2.0, batch_size=2048, epochs=5, random_state=42):
        self.n_clusters=n_clusters; self.genre_weight=genre_weight; self.batch_size=batch_size
        self.epochs=epochs; self.random_state=random_state
        self.scaler=StandardScaler(); self.kmeans=None; self.genre_columns=[]

    def make_features(self, df: pd.DataFrame, genre_columns: list[str], fit=False) -> np.ndarray:
        self.genre_columns = list(genre_columns)
        ratings = df[['media_nota_r']].astype(float)
        if fit: self.scaler.fit(ratings)
        normalized = self.scaler.transform(ratings).astype(np.float32)
        genres = df[self.genre_columns].astype(np.float32).to_numpy(copy=False) * self.genre_weight
        return np.hstack([genres, normalized]).astype(np.float32, copy=False)

    def fit(self, X: np.ndarray) -> 'ClusterModel':
        if len(X) < self.n_clusters: raise ValueError('Poucos dados para o número de clusters escolhido.')
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, random_state=self.random_state,
            batch_size=self.batch_size, n_init=3, reassignment_ratio=0.01
        )
        rng=np.random.default_rng(self.random_state)
        n=len(X)
        for _ in range(self.epochs):
            idx=rng.permutation(n)
            for start in range(0,n,self.batch_size):
                self.kmeans.partial_fit(X[idx[start:start+self.batch_size]])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kmeans is None: raise RuntimeError('Modelo não treinado.')
        return self.kmeans.predict(X)

    def save(self, path: Path, dataset_path: Path, genre_columns: list[str], n_rows: int | None = None) -> None:
        stat=dataset_path.stat()
        payload={
            'schema_version': 2,
            'metadata': {
                'dataset_name': dataset_path.name,
                'dataset_size': stat.st_size,
                'dataset_mtime_ns': stat.st_mtime_ns,
                'n_rows': int(n_rows) if n_rows is not None else None,
                'genre_columns': list(genre_columns),
                'n_clusters': self.n_clusters,
                'genre_weight': self.genre_weight,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'random_state': self.random_state,
                'sklearn_version': __import__('sklearn').__version__,
            },
            'scaler': self.scaler,
            'kmeans': self.kmeans,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> tuple['ClusterModel', dict[str,Any]]:
        payload=joblib.load(path)
        if payload.get('schema_version') != 2:
            raise ValueError('Formato de modelo incompatível. Gere o modelo novamente.')
        m=payload['metadata']; obj=cls(
            n_clusters=m['n_clusters'], genre_weight=m['genre_weight'], batch_size=m['batch_size'],
            epochs=m['epochs'], random_state=m['random_state'])
        obj.scaler=payload['scaler']; obj.kmeans=payload['kmeans']; obj.genre_columns=m['genre_columns']
        return obj,m
