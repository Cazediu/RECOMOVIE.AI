from pathlib import Path
import streamlit as st
from src.config import default_config
from src.data import MovieRepository
from src.model import ClusterModel
from src.recommender import MovieRecommender

BASE=Path(__file__).resolve().parent
CFG=default_config(BASE)

st.set_page_config(page_title='RECOMOVIE.AI', page_icon='🎬', layout='wide')

@st.cache_resource(show_spinner=False)
def carregar():
    repo=MovieRepository(CFG.data_path, CFG.movies_path)
    model,meta=ClusterModel.load(CFG.model_path)
    current_genres=repo.genre_columns
    if current_genres != meta['genre_columns']:
        raise RuntimeError('O modelo e o dataset possuem características diferentes. Rode train.py novamente.')
    return repo,MovieRecommender(repo,model),meta

st.title('RECOMOVIE.AI')
st.caption('Recomendador de filmes por agrupamento + similaridade de cosseno')

try:
    repo,recommender,meta=carregar()
except Exception as e:
    st.error(f'Não foi possível carregar o sistema: {e}')
    st.stop()

with st.sidebar:
    st.header('Como usar')
    st.write('1. Pesquise um filme\n2. Escolha o título\n3. Ajuste os filtros\n4. Veja recomendações parecidas')
    st.divider()
    quantity=st.slider('Quantidade de recomendações',3,20,10)
    min_ratings=st.slider('Mínimo de avaliações',0,5000,50,step=50)
    genre_options=['Todos']+repo.genre_columns
    genre=st.selectbox('Filtrar por gênero',genre_options)

query=st.text_input('Qual filme você curte?', placeholder='Ex.: Toy Story, Batman, Matrix...')
found=repo.find_titles(query,limit=50)
if found.empty:
    st.info('Nenhum filme encontrado. Tenta outro nome, mano.')
    st.stop()

labels=found.apply(lambda r:f"{r['title']}  ·  nota {r['media_nota_r']:.2f}",axis=1).tolist()
choice=st.selectbox('Selecione um filme',range(len(labels)),format_func=lambda i:labels[i])
selected=found.iloc[choice]

c1,c2,c3=st.columns(3)
c1.metric('Nota média',f"{selected['media_nota_r']:.2f}")
c2.metric('Avaliações',f"{int(selected['quant_avaliadores']):,}")
cluster=int(recommender.get_movie(int(selected['movieId']))['cluster'])
profile=recommender.cluster_profile(cluster)
c3.metric('Perfil encontrado',f"Grupo {cluster+1}")

st.write(f"**Gêneros:** {selected['genres'] if selected['genres'] else 'não informado'}")

if st.button('Encontrar filmes parecidos', type='primary', use_container_width=True):
    with st.spinner('Comparando filmes...'):
        recs=recommender.recommendations(int(selected['movieId']),quantity,min_ratings,genre)
    st.subheader('Recomendações')
    if recs.empty:
        st.warning('Não encontrei candidatos com esses filtros. Diminui o mínimo de avaliações ou troca o gênero.')
    else:
        for _,r in recs.iterrows():
            with st.container(border=True):
                left,right=st.columns([4,1])
                with left:
                    st.markdown(f"### {r['title']}")
                    st.caption(r['genres'] if r['genres'] else 'Gêneros não informados')
                with right:
                    st.metric('Similaridade',f"{r['similaridade']*100:.1f}%")
                    st.caption(f"Nota {r['media_nota_r']:.2f} · {int(r['quant_avaliadores']):,} avaliações")

with st.expander('O que o modelo faz e quais são as limitações'):
    st.write(f"O sistema representa cada filme pelos gêneros e pela média de avaliações, agrupa os filmes em {meta['n_clusters']} clusters com MiniBatch K-Means e depois usa similaridade de cosseno para ranquear candidatos dentro do grupo.")
    st.warning('Isso é um recomendador baseado apenas nas características disponíveis no dataset. Ele não conhece seu histórico pessoal, contexto, humor ou preferências individuais. Filmes com poucos dados podem ter comportamento menos confiável.')

st.caption('Modelo persistido em arquivo: a aplicação não treina novamente ao ser aberta.')
