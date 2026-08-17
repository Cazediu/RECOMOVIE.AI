# RECOMOVIE.AI — Etapa 3

## O que foi melhorado
- Interface Streamlit pensada para usuário comum: pesquisa por título, seleção do filme, filtros e resultados visuais.
- Arquitetura em POO separando dados, modelo e recomendação.
- Modelo persistido com `joblib`; a aplicação não treina ao abrir.
- Pacote do modelo sem caminho absoluto do computador original.
- Validação robusta para clusterização: métricas internas, sensibilidade a `k`, estabilidade por seeds e PCA 2D.
- Relatórios automáticos em `reports/`.

## Execução local
```bash
pip install -r requirements.txt
python train.py
python validacao_robusta.py
streamlit run app.py
```

## Observação
O `train.py` recria o modelo com os mesmos princípios do projeto original: MiniBatch K-Means sobre gêneros + média normalizada e recomendação por similaridade de cosseno dentro do cluster.

## Relação com a Etapa 3
A aplicação atende ao requisito de interface, carregamento do modelo persistido e explicação de limitações. Para clusterização, a validação usa métricas internas, sensibilidade ao número de clusters, estabilidade por inicialização e visualização via PCA.
