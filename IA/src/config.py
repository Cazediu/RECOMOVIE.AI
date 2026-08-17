from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    data_path: Path
    movies_path: Path
    model_path: Path
    n_clusters: int = 25
    genre_weight: float = 2.0
    batch_size: int = 2048
    epochs: int = 5
    random_state: int = 42


def default_config(base_dir: Path | None = None) -> AppConfig:
    base = base_dir or Path(__file__).resolve().parent.parent
    return AppConfig(
        data_path=base / 'auxiliar.csv',
        movies_path=base / 'movies.csv',
        model_path=base / 'modelo_recomovie.joblib',
    )
