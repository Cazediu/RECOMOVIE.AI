from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from src.config import default_config
from src.data import MovieRepository
from src.model import ClusterModel

BASE=Path(__file__).resolve().parent
OUT=BASE/'reports'; OUT.mkdir(exist_ok=True)


def fit_model(X, k, seed):
    m=MiniBatchKMeans(n_clusters=k,random_state=seed,batch_size=2048,n_init=3,reassignment_ratio=0.01)
    rng=np.random.default_rng(seed)
    for _ in range(3):
        idx=rng.permutation(len(X))
        for start in range(0,len(X),2048): m.partial_fit(X[idx[start:start+2048]])
    return m


def main():
    cfg=default_config(BASE); repo=MovieRepository(cfg.data_path,cfg.movies_path)
    model=ClusterModel(cfg.n_clusters,cfg.genre_weight,cfg.batch_size,cfg.epochs,cfg.random_state)
    X=model.make_features(repo.df,repo.genre_columns,fit=True)
    rng=np.random.default_rng(42)
    sample_size=min(10000,len(X)); sample_idx=rng.choice(len(X),sample_size,replace=False)
    Xs=X[sample_idx]
    # 1) métricas internas do modelo oficial
    labels=model.fit(X).predict(X)
    sample_labels=labels[sample_idx]
    metrics={
        'modelo_oficial': {'k':cfg.n_clusters,'amostra':sample_size,
            'silhouette':float(silhouette_score(Xs,sample_labels)),
            'davies_bouldin':float(davies_bouldin_score(Xs,sample_labels)),
            'calinski_harabasz':float(calinski_harabasz_score(Xs,sample_labels))}
    }
    # 2) sensibilidade a k
    rows=[]
    for k in [10,15,20,25,30,35]:
        mk=fit_model(Xs,k,42)
        lab=mk.predict(Xs)
        rows.append({'k':k,'silhouette':silhouette_score(Xs,lab),'davies_bouldin':davies_bouldin_score(Xs,lab),'calinski_harabasz':calinski_harabasz_score(Xs,lab)})
    kdf=pd.DataFrame(rows); kdf.to_csv(OUT/'sensibilidade_k.csv',index=False)
    fig=plt.figure(figsize=(8,4.5)); plt.plot(kdf.k,kdf.silhouette,marker='o'); plt.xlabel('Número de clusters (k)'); plt.ylabel('Silhouette'); plt.title('Sensibilidade ao número de clusters'); plt.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT/'sensibilidade_k.png',dpi=160); plt.close(fig)
    # 3) estabilidade por seed
    seed_models=[]
    seeds=[0,7,21,42,99]
    base_model=fit_model(Xs,cfg.n_clusters,seeds[0]); seed_labels=[base_model.predict(Xs)]
    for seed in seeds[1:]:
        sm=fit_model(Xs,cfg.n_clusters,seed)
        seed_labels.append(sm.predict(Xs))
    ari=[]
    for i in range(len(seed_labels)):
        for j in range(i+1,len(seed_labels)): ari.append(adjusted_rand_score(seed_labels[i],seed_labels[j]))
    metrics['estabilidade_seeds']={'seeds':seeds,'ari_media':float(np.mean(ari)),'ari_desvio':float(np.std(ari)),'ari_min':float(np.min(ari))}
    # 4) PCA 2D
    pca=PCA(n_components=2,random_state=42); X2=pca.fit_transform(Xs)
    fig=plt.figure(figsize=(8,5)); plt.scatter(X2[:,0],X2[:,1],c=sample_labels,s=5,alpha=.45); plt.xlabel('Componente principal 1'); plt.ylabel('Componente principal 2'); plt.title('Visualização dos clusters via PCA'); fig.tight_layout(); fig.savefig(OUT/'clusters_pca.png',dpi=160); plt.close(fig)
    json.dump(metrics,(OUT/'metricas.json').open('w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    print('Relatórios salvos em',OUT)

if __name__=='__main__': main()
