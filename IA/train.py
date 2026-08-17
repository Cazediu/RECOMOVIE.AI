from pathlib import Path
import json
from src.config import default_config
from src.data import MovieRepository
from src.model import ClusterModel

def main():
    cfg=default_config(Path(__file__).resolve().parent)
    repo=MovieRepository(cfg.data_path,cfg.movies_path)
    model=ClusterModel(cfg.n_clusters,cfg.genre_weight,cfg.batch_size,cfg.epochs,cfg.random_state)
    X=model.make_features(repo.df,repo.genre_columns,fit=True)
    model.fit(X)
    # MiniBatchKMeans não mantém labels_; usamos predict para registrar o total de linhas no manifesto.
    model.save(cfg.model_path,cfg.data_path,repo.genre_columns,n_rows=len(repo.df))
    manifest={
        'modelo':cfg.model_path.name,'filmes':len(repo.df),'clusters':cfg.n_clusters,
        'caracteristicas':len(repo.genre_columns)+1,
        'genero_weight':cfg.genre_weight,
    }
    (Path(__file__).resolve().parent/'model_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Modelo salvo em',cfg.model_path)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
