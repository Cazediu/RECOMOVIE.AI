
from pathlib import Path
import pandas as pd

# =========================
# CONFIGURAÇÃO
# =========================
BASE_DIR = Path(__file__).resolve().parent

# Se o script estiver dentro da pasta "datasets", usar essa pasta como DATASET_DIR.
# Caso contrário, assumir que existe uma subpasta "datasets" no mesmo nível do script.
if BASE_DIR.name == "datasets":
    DATASET_DIR = BASE_DIR
else:
    DATASET_DIR = BASE_DIR / "datasets"

RATINGS_PATH = DATASET_DIR / "ratings.csv"
AUX_PATH = DATASET_DIR / "aux.csv"

# =========================
# FUNÇÕES AUXILIARES
# =========================
def carregar_csv(caminho: Path) -> pd.DataFrame:
    """Carrega CSV em DataFrame com verificação de existência."""
    return pd.read_csv(caminho)


def remover_coluna(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Retorna cópia do DataFrame sem a coluna especificada (se existir)."""
    if coluna in df.columns:
        return df.drop(columns=[coluna]).copy()
    return df.copy()


def agregar_por_filme(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por movieId e calcula:
      - quant_avaliadores: contagem de avaliações
      - media_nota_r: média das notas
      - variancia_nota: variância populacional das notas (ddof=0)
    """

    df2 = df[["movieId", "rating"]].copy()
    df2["rating"] = pd.to_numeric(df2["rating"], errors="coerce")
    df2 = df2.dropna(subset=["rating"])

    agg = (
        df2.groupby("movieId")
        .agg(
            quant_avaliadores=("rating", "count"),
            media_nota_r=("rating", "mean"),
            variancia_nota=("rating", lambda x: x.var(ddof=0)),
        )
        .reset_index()
    )

    agg["media_nota_r"] = agg["media_nota_r"].round(4)
    agg["variancia_nota"] = agg["variancia_nota"].round(4).fillna(0.0)

    return agg


def salvar_csv(df: pd.DataFrame, caminho: Path) -> None:
    """Salva DataFrame em CSV, criando pasta se necessário."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8")


# =========================
# FLUXO PRINCIPAL
# =========================
def main():
    print(f"Procurando ratings em: {RATINGS_PATH}")
    df = carregar_csv(RATINGS_PATH)

    # Remover duplicatas e reatribuir
    df = df.drop_duplicates()

    # Remover colunas e reatribuir o resultado
    df = remover_coluna(df, "timestamp")
    df = remover_coluna(df, "userId")

    # Agregar e salvar
    df_agg = agregar_por_filme(df)
    salvar_csv(df_agg, AUX_PATH)


main()
