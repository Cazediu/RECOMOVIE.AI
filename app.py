"""Interface web (Streamlit) do Recomovie AI.

Este arquivo cuida apenas da camada visual: monta a página, captura a
busca do usuário e exibe as recomendações. Toda a lógica de machine
learning (clusterização e similaridade) fica em `recomendador.py`.
"""

import streamlit as st
from streamlit_searchbox import st_searchbox

from recomendador import RecomendadorFilmes


st.set_page_config(page_title="Recomovie AI", page_icon="R")

st.title(" Recomovie AI")
st.write("Escolha um filme e receba recomendações parecidas.")
st.info("O sistema usa K-Means para separar filmes em grupos e depois usa similaridade de cosseno para encontrar filmes parecidos dentro do mesmo grupo.")


@st.cache_resource
def carregar_recomendador():
    """Cria o RecomendadorFilmes uma única vez por sessão do servidor.

    O Streamlit reexecuta o script inteiro a cada interação do usuário;
    `st.cache_resource` evita retreinar o modelo em cada uma dessas execuções.
    """
    return RecomendadorFilmes()


recomendador = carregar_recomendador()


def buscar_sugestoes(termo: str):
    """Callback de busca usado pela barra de autocomplete.

    É chamado automaticamente pelo componente `st_searchbox` a cada trecho
    digitado pelo usuário (com pequeno atraso/debounce interno). Retorna uma
    lista de tuplas (texto_exibido, valor_retornado); o valor retornado é o
    `movieId`, para evitar ambiguidade quando dois filmes têm o mesmo título.
    """
    filmes = recomendador.buscar_filmes(termo)
    return [(filme.title, filme.movieId) for filme in filmes.itertuples()]


# Barra única de pesquisa: conforme o usuário digita, sugestões de filmes
# aparecem em uma lista para seleção, como em uma busca do Google.
# Substitui o antigo fluxo em duas etapas (texto + selectbox separado).
movie_id = st_searchbox(
    buscar_sugestoes,
    placeholder="Digite o nome de um filme...",
    label="Buscar filme",
    key="busca_filme",
    clear_on_submit=False,
)

if movie_id:
    quantidade = st.slider("Quantidade de recomendações", 5, 20, 10)

    if st.button("Recomendar filmes"):
        resultado = recomendador.recomendar(movie_id, quantidade)

        st.subheader("Filmes recomendados")

        if resultado.empty:
            st.warning("Não encontrei recomendações para este filme.")
        else:
            for _, filme in resultado.iterrows():
                st.markdown(
                    f"**{filme['title']}**  \n"
                    f"Nota média: {filme['media_nota_r']:.2f} | "
                    f"Similaridade: {filme['similaridade']:.1f}%"
                )
                st.divider()
else:
    st.caption("Comece digitando o nome de um filme para ver sugestões.")

st.caption("Limitação: recomendações dependem dos dados disponíveis e podem não fazer sentido para filmes muito pouco avaliados.")
