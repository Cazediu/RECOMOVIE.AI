# Discussão crítica — Etapa 3

## O que a validação revelou

O recomendador apresenta estrutura de agrupamento, mas não deve ser descrito como perfeitamente estável. No modelo oficial com **25 clusters**, avaliado sobre uma amostra de **10.000 filmes**, a Silhouette foi **0,299**, o Davies-Bouldin **1,468** e o Calinski-Harabasz **800,85**. Esses resultados indicam que existe separação mensurável entre os grupos, porém há sobreposição entre agrupamentos.

Na análise de sensibilidade, diferentes valores de `k` produziram resultados diferentes. Por exemplo, na mesma amostra a Silhouette variou de aproximadamente **0,280 em k=20** a **0,323 em k=35**. Portanto, a escolha de 25 clusters é uma decisão de projeto e não uma prova de que 25 seja o único valor correto.

Na análise de estabilidade com cinco seeds, o ARI médio foi **0,615**, com desvio padrão de aproximadamente **0,062** e mínimo de **0,530**. Isso mostra uma estabilidade moderada: o algoritmo encontra estruturas relacionadas, mas não produz exatamente os mesmos agrupamentos em todas as inicializações.

## Limitações

O sistema usa apenas gêneros e média de avaliações para representar os filmes. Ele não conhece o histórico pessoal de um usuário, atores, diretores, enredo, contexto de consumo ou preferências temporais. Por isso, uma recomendação pode ser matematicamente semelhante ao filme consultado e ainda assim não agradar a uma pessoa específica.

Filmes com poucos dados também podem ser menos confiáveis. O filtro de número mínimo de avaliações reduz esse risco, mas não resolve o problema de forma completa.

## Uso real

O grupo poderia confiar no sistema como **protótipo educacional e ferramenta de recomendação exploratória**, mas não trataria as recomendações como decisões de alta confiança. Em produção, seria importante incorporar feedback real de usuários, medir qualidade das recomendações com dados novos e acompanhar estabilidade do agrupamento ao longo do tempo.

## Melhorias futuras

Com mais tempo ou dados, as prioridades seriam: incorporar preferências individuais; adicionar características semânticas dos filmes; testar outros valores e critérios de seleção de `k`; comparar algoritmos de agrupamento; e criar uma avaliação online baseada em cliques, avaliações ou taxa de aceitação das recomendações.
