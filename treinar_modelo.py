"""Script de linha de comando para (re)treinar e salvar o modelo.

Basta rodar `python treinar_modelo.py`: ele cria o `RecomendadorFilmes`
(o que treina o K-Means com os dados atuais) e salva o resultado em
`modelo_recomovie.joblib`, usado depois pela aplicação Streamlit.
"""

from recomendador import RecomendadorFilmes

# Instanciar já treina o modelo (ver RecomendadorFilmes.__init__).
recomendador = RecomendadorFilmes()
recomendador.salvar()

print("Modelo criado: modelo_recomovie.joblib")
