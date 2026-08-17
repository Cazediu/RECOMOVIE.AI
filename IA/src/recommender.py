from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .data import MovieRepository
from .model import ClusterModel

class MovieRecommender:
    def __init__(self, repository: MovieRepository, model: ClusterModel):
        self.repo=repository; self.model=model
        self.X=model.make_features(repository.df, repository.genre_columns, fit=False)
        self.clusters=model.predict(self.X)
        self.df=repository.df.copy()
        self.df['cluster']=self.clusters

    def get_movie(self, movie_id: int) -> pd.Series:
        return self.df.iloc[self.repo.index_from_movie_id(movie_id)]

    def recommendations(self, movie_id: int, quantity: int=10, min_ratings: int=50,
                        genre_filter: str='Todos') -> pd.DataFrame:
        idx=self.repo.index_from_movie_id(movie_id)
        row=self.df.iloc[idx]
        mask=(self.df['cluster']==row['cluster']) & (self.df['movieId']!=movie_id) & (self.df['quant_avaliadores']>=min_ratings)
        if genre_filter and genre_filter != 'Todos' and genre_filter in self.repo.genre_columns:
            mask &= self.df[genre_filter].eq(1)
        candidate_idx=np.flatnonzero(mask.to_numpy())
        if len(candidate_idx)==0: return pd.DataFrame()
        sims=cosine_similarity(self.X[idx:idx+1], self.X[candidate_idx])[0]
        result=self.df.iloc[candidate_idx].copy()
        result['similaridade']=sims
        # Penaliza muito pouco filmes com poucas avaliações, sem apagar filmes de nicho.
        result['score_final']=result['similaridade']*(0.85+0.15*np.minimum(result['quant_avaliadores']/1000,1.0))
        result=result.sort_values(['score_final','media_nota_r'],ascending=False).head(quantity)
        cols=['movieId','title','genres','media_nota_r','quant_avaliadores','cluster','similaridade','score_final']
        return result[cols].reset_index(drop=True)

    def cluster_profile(self, cluster_id: int) -> dict:
        subset=self.df[self.df['cluster']==cluster_id]
        genre_means=subset[self.repo.genre_columns].mean().sort_values(ascending=False)
        genres=[g for g,v in genre_means.head(5).items() if v>0]
        return {'id':int(cluster_id),'size':int(len(subset)),'genres':genres,'mean_rating':float(subset['media_nota_r'].mean())}
