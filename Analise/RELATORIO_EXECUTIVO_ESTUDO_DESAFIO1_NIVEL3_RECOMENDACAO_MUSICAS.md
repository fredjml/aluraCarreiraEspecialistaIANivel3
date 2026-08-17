# RELATÓRIO TÉCNICO-EXECUTIVO DE ESTUDO

## Desafio 1 - Nível 3: Sistema de Recomendação de Músicas da Playcatch

**Data do estudo:** 17 de agosto de 2026  
**Situação:** análise, planejamento e especificação concluídos; nenhuma etapa do desafio foi executada  
**Escopo:** EDA, recomendação item-based, avaliação offline, rastreamento com MLflow/SQLite e interface Gradio  
**Persona recomendada:** Cientista de Dados / Engenheiro de Machine Learning com foco em sistemas de recomendação, experimentação reprodutível e entrega de aplicações interativas

---

## 1. Sumário executivo

O desafio solicita o desenho e, em etapa futura, a construção de um sistema de recomendação musical baseado em histórico implícito de reprodução. A solução esperada percorre cinco camadas: preparação e exploração de dados, construção de uma matriz usuário-item, cálculo de similaridade entre músicas, avaliação offline por ocultação de uma interação, rastreamento experimental com MLflow/SQLite e disponibilização de recomendações em uma interface Gradio.

O estudo conclui que o projeto é viável como prova de conceito didática, mas o conjunto fornecido é pequeno demais para validar qualidade ou capacidade de generalização. São apenas 16 eventos, 3 usuários e 8 músicas. Todos os usuários acumulam 20 reproduções; consequentemente, o ranking de “Top 5 usuários” resulta em empate entre os três únicos usuários. A matriz contém 24 posições possíveis e apenas 13 pares usuário-música observados após agregação, densidade de 54,17%.

Há duas duplicidades de chave, mas não duplicatas exatas: `(user_id=2, song_id=101)` aparece com contagens 4 e 7, e `(user_id=2, song_id=104)` aparece com contagens 6 e 2. Por isso, `DataFrame.pivot()` falharia com índices duplicados. A construção correta deve empregar `pivot_table(index="user_id", columns="song_id", values="play_count", aggfunc="sum", fill_value=0)` ou uma agregação prévia equivalente.

Também foram identificadas inconsistências no enunciado. `transformers` e `pipeline` são apresentados na preparação, mas não participam do recomendador item-based com `NearestNeighbors`. O pacote `sqlite` não deve ser instalado com `pip`; o módulo `sqlite3` integra a biblioteca padrão do Python, e o MLflow pode criar o banco local. A interface Gradio é exigida no objetivo e no checklist, porém não há uma etapa detalhada, contrato de entrada/saída nem critérios de aceite específicos para ela. Essas lacunas devem ser resolvidas no planejamento antes da implementação.

**Parecer executivo:** prosseguir como POC educacional, com separação rigorosa entre demonstração funcional e avaliação de qualidade. O desenvolvimento futuro deve registrar limitações, usar validação leave-one-out determinística, impedir recomendação de músicas já ouvidas, comparar contra um baseline de popularidade e preservar artefatos suficientes para reprodução.

## 2. Limites deste relatório

Este documento é exclusivamente um estudo de viabilidade e preparação. Não foram instaladas dependências do desafio, criados ambientes virtuais, gerados arquivos CSV, treinados modelos, iniciados servidores MLflow, abertas interfaces Gradio ou calculadas métricas por execução de código. Os números derivados do CSV foram conferidos analiticamente a partir do conteúdo fornecido e servem para antecipar decisões e riscos.

O relatório-modelo anexado foi analisado como referência editorial e visual. Suas afirmações sobre uma implementação anterior não foram tratadas como instruções para este trabalho e não foram reaproveitadas como evidência do novo desafio.

## 3. Interpretação do desafio

### 3.1 Objetivo de negócio

Entregar uma POC capaz de sugerir músicas ainda não consumidas por um usuário a partir da coocorrência de consumo entre usuários. O valor esperado é demonstrar personalização, rastreabilidade de experimentos e uma interação simples para apresentação do resultado.

### 3.2 Requisitos funcionais explícitos

1. Carregar `user_data.csv` e converter `last_played` para data/hora.
2. Auditar ausências e duplicatas e documentar o tratamento.
3. Produzir estatísticas de `play_count`, agregações por usuário e por música e rankings Top 5.
4. Criar matriz `user_id x song_id`, preenchendo ausências com zero.
5. Normalizar os vetores usados na similaridade.
6. Ajustar um `NearestNeighbors` item-based com distância cosseno.
7. Implementar `recomendar_para_usuario(user_id, k)` sem retornar músicas já ouvidas.
8. Somar evidências de similaridade e ordenar as Top k novas músicas.
9. Realizar validação por usuário ocultando a música agregada com maior `play_count`.
10. Calcular Hit Rate@k e, opcionalmente, Precision@k.
11. Registrar parâmetros, métricas, artefatos, modelo e mapeamentos no MLflow com backend SQLite.
12. Comparar múltiplas configurações experimentais.
13. Expor recomendações em uma interface Gradio.

### 3.3 Requisitos implícitos de qualidade

- Resultados determinísticos, com regras de desempate declaradas.
- Validação sem vazamento entre treino e teste.
- Identificação clara entre distância e similaridade: `similaridade = 1 - distância_cosseno`.
- Tratamento de usuário inexistente, usuário sem histórico, `k <= 0`, `k` maior que o catálogo elegível e catálogo sem candidatos.
- Persistência dos mapeamentos entre índices internos e IDs originais.
- Separação entre backend de metadados do MLflow e armazenamento de artefatos.
- Testes unitários para agregação, filtragem de itens conhecidos, ordenação, empates e métricas.
- Interface que não exponha stack traces e que descreva quando não há recomendação possível.
- Documentação das limitações do pequeno dataset e proibição de interpretar Hit Rate como prova de prontidão para produção.

### 3.4 Lacunas e ambiguidades do enunciado

| Tema | Lacuna | Decisão recomendada para a futura execução |
|---|---|---|
| Duplicatas | “Excluir duplicatas” não distingue linhas idênticas de eventos repetidos do mesmo par | Remover apenas linhas integralmente idênticas; somar interações repetidas por usuário-música |
| Orientação do KNN | A matriz é pedida como usuário x música, mas o modelo é item-based | Treinar com a transposta: músicas nas linhas e usuários nas colunas |
| Normalização | “Itens ou usuários” permite soluções diferentes | Normalizar itens por L2 e manter a decisão como hiperparâmetro `norm` |
| Pontuação | “Soma das similaridades” não define ponderação | Implementar inicialmente soma simples; testar variante ponderada por força de reprodução |
| Vizinhos | Não esclarece inclusão do próprio item | Consultar vizinhos extras e remover explicitamente o próprio item |
| Validação | Não define se a música de maior play é escolhida antes ou depois da agregação | Agregar primeiro, aplicar desempate estável por `song_id` e ocultar um item por usuário |
| Precision@k | Há apenas um item relevante oculto por usuário | Definir `Precision@k = hits / (n_usuários_avaliáveis x k)` e priorizar Hit Rate@k/MRR@k |
| Gradio | Não existe etapa detalhada | Adotar entrada `user_id` e `k`; saída tabular com `song_id`, score e justificativa curta |
| Transformers | Biblioteca não é usada pelo algoritmo especificado | Remover da instalação mínima; manter somente se houver futura recomendação baseada em conteúdo |
| SQLite | O comando sugere `pip install sqlite` | Não instalar; usar `sqlite3` da biblioteca padrão e URI SQLAlchemy do MLflow |

## 4. Análise profunda do dataset fornecido

### 4.1 Inventário e qualidade

O arquivo proposto contém as colunas `user_id`, `song_id`, `play_count` e `last_played`, com 16 linhas. Os IDs são inteiros, `play_count` é contagem positiva e `last_played` cobre apenas 1 a 3 de outubro de 2023. Não há valores ausentes no texto fornecido e não há linhas integralmente idênticas.

Existem 3 usuários distintos e 8 músicas distintas. Após consolidar as duas chaves repetidas, restam 13 interações usuário-música. A soma global é 60 reproduções.

### 4.2 Estatísticas determinísticas antecipadas

| Medida | Valor derivado do conteúdo fornecido |
|---|---|
| Linhas brutas | 16 |
| Usuários únicos | 3 |
| Músicas únicas | 8 |
| Pares únicos usuário-música | 13 |
| Reproduções totais | 60 |
| Dimensão da matriz | 3 x 8 |
| Densidade após agregação | 13/24 = 54,17% |
| Período observado | 2023-10-01 a 2023-10-03 |

Totais por usuário: usuário 1 = 20, usuário 2 = 20 e usuário 3 = 20. Os três empatam em volume; portanto, não existem cinco usuários para listar e nenhum critério legítimo permite inventar posições 4 e 5.

Totais por música após soma: música 104 = 18; 101 = 16; 106 = 11; 103 = 5; 102 = 3; 107 = 3; 105 = 2; 108 = 2. A quinta posição possui empate entre 102 e 107. O relatório futuro deve declarar a regra de desempate, por exemplo `song_id` ascendente.

### 4.3 Consequências para modelagem

- A amostra não sustenta segmentação, validação estatística ou tuning confiável.
- Similaridades entre itens são calculadas sobre apenas três dimensões, uma por usuário; pequenas mudanças alteram fortemente o ranking.
- O recorte temporal de três dias não representa preferência de longo prazo nem mudança de gosto.
- `play_count` mistura preferência e exposição; usuários com mais oportunidade de escuta podem dominar scores em bases reais.
- Não há metadados de música, artista, gênero, data ou popularidade externa; o sistema não resolve cold start de item.
- Não há usuários sem histórico, logo esse caso precisa de teste sintético e fallback de popularidade.

## 5. Persona necessária

### 5.1 Persona principal

**Cientista de Dados / Engenheiro de Machine Learning para sistemas de recomendação**, com senioridade júnior avançada ou plena para uma POC orientada, e plena/sênior para transformar a solução em serviço confiável.

Essa pessoa deve combinar raciocínio analítico, implementação Python, avaliação offline, organização experimental e comunicação. Não basta conhecer uma chamada de API: é necessário compreender a orientação das matrizes, a diferença entre feedback implícito e explícito, o impacto da esparsidade e as limitações das métricas.

### 5.2 Responsabilidades esperadas

- Traduzir requisitos de produto em protocolo mensurável.
- Auditar o dataset e documentar decisões de limpeza.
- Implementar recomendação item-item sem vazamento.
- Definir fallbacks, empates e contratos de erro.
- Criar experimentos comparáveis e rastreáveis.
- Desenvolver uma interface simples, acessível e segura.
- Comunicar que uma POC funcional não equivale a um recomendador validado.

### 5.3 Competências comportamentais

- Rigor para não superinterpretar métricas em amostras pequenas.
- Curiosidade para investigar inconsistências do enunciado.
- Disciplina de reprodução: seeds, versões, parâmetros, dados e artefatos.
- Comunicação com pessoas de produto e engenharia.
- Mentalidade de privacidade: históricos musicais podem revelar hábitos e preferências sensíveis.

## 6. Conhecimentos necessários

| Área | Conhecimentos requeridos | Nível sugerido |
|---|---|---|
| Python | funções, tipos, exceções, pathlib, JSON, testes e ambientes virtuais | Intermediário |
| Pandas | tipos, datas, `duplicated`, `groupby`, `pivot_table`, ordenação e exportação | Intermediário |
| Álgebra linear | vetores, norma L2, produto escalar, transposição e matriz esparsa | Intermediário |
| Recomendação | feedback implícito, item-based CF, cold start, popularidade e filtragem de vistos | Intermediário |
| Scikit-learn | `Normalizer`, `NearestNeighbors`, métricas de distância e serialização | Intermediário |
| Avaliação | leave-one-out, Hit Rate@k, Precision@k, MRR@k, leakage e baselines | Intermediário |
| MLflow | tracking URI, experimentos, runs, parâmetros, métricas, artefatos e model flavor | Intermediário |
| SQLite | URI, arquivo local, concorrência limitada, backup e diferença entre banco e artefatos | Básico |
| Gradio | componentes, validação de entrada, eventos, tabela de saída e execução local/Colab | Básico/intermediário |
| Engenharia | estrutura de projeto, logging, configuração, testes, versionamento e segurança | Intermediário |

## 7. Softwares e dependências

### 7.1 Caminho recomendado: Google Colab

Para a POC didática, o Colab reduz configuração local. É necessário navegador moderno, conta Google para persistência no Drive quando desejada e runtime Python compatível com as bibliotecas. O notebook deve instalar apenas dependências ausentes e registrar as versões efetivamente resolvidas. O banco SQLite e os artefatos são efêmeros no runtime; portanto, devem ser copiados para armazenamento persistente antes do encerramento da sessão.

### 7.2 Caminho local recomendado

- Python 3.10 a 3.12, 64 bits. Python 3.10 é o piso atual documentado pelo Gradio.
- VS Code com extensão Python **ou** JupyterLab; somente um é necessário.
- Git para versionamento, recomendado mas não obrigatório para a POC.
- Ambiente virtual `venv`, `uv` ou Conda; não instalar no Python global.
- Navegador para MLflow UI e Gradio.
- CPU é suficiente para o dataset; GPU não é necessária.

### 7.3 Matriz de pacotes

| Pacote/recurso | Necessidade | Observação de instalação |
|---|---|---|
| `pandas` | Obrigatório | Leitura, limpeza, agregação e pivot |
| `scikit-learn` | Obrigatório | Normalização e `NearestNeighbors` |
| `numpy` | Obrigatório indireto | Instalado como dependência do scikit-learn; útil para vetores |
| `scipy` | Obrigatório indireto | Suporte numérico/esparso do scikit-learn |
| `mlflow` | Obrigatório | Tracking, UI e persistência do modelo/artefatos |
| `gradio` | Obrigatório | Interface exigida pelo objetivo e checklist |
| `matplotlib` | Opcional | Apenas para histograma/figuras da EDA |
| `jupyterlab` ou notebook Colab | Um deles | Ambiente de exploração |
| `pytest` | Recomendado | Testes automatizados |
| `transformers` | Não necessário neste escopo | Só faria sentido em extensão baseada em conteúdo/embeddings |
| `sqlite` via pip | Não instalar | Usar `sqlite3` do Python; MLflow acessa SQLite via sua pilha SQL |

### 7.4 Comandos planejados, não executados

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install pandas scikit-learn mlflow gradio matplotlib pytest jupyterlab
python -m pip freeze > requirements-lock.txt
```

No Colab, a forma equivalente pode usar `%pip install`, seguida de reinicialização do runtime apenas se solicitada. A futura execução deve fixar versões depois de uma instalação validada, não copiar versões arbitrárias deste estudo.

## 8. Arquitetura lógica proposta

```text
user_data.csv
    |
    v
Validação de esquema e datas
    |
    v
Agregação por usuário-música + EDA
    |
    v
Matriz usuário x música -----> baseline de popularidade
    |
    v (transposição)
Matriz música x usuário -> normalização L2
    |
    v
NearestNeighbors(metric="cosine", algorithm="brute")
    |
    v
Geração de candidatos -> remove ouvidas -> agrega similaridades -> Top k
    |                                      |
    |                                      +-> interface Gradio
    v
Validação leave-one-out -> métricas -> MLflow/SQLite + artefatos locais
```

### 8.1 Componentes sugeridos

- `data.py`: carga, validação, agregação e matriz.
- `model.py`: normalização, treino KNN, mapeamentos e recomendação.
- `evaluate.py`: construção dos folds e métricas.
- `tracking.py`: encapsulamento do MLflow.
- `app.py`: interface Gradio.
- `tests/`: testes de regras e casos extremos.
- `artifacts/`: rankings, shape, amostras, mapeamentos e manifesto do dataset.

## 9. Estratégia de modelagem

### 9.1 Preparação

Converter `last_played` com `errors="raise"` na execução controlada; em ingestão real, considerar `errors="coerce"` seguido de relatório de rejeições. Validar IDs não nulos, `play_count > 0` e ausência de colunas inesperadas. Remover somente duplicatas integrais e somar eventos repetidos do mesmo par.

### 9.2 KNN item-based

Criar a matriz usuário x música para EDA e transpor para música x usuário no treino. Aplicar normalização L2 por linha. Com distância cosseno, usar `algorithm="brute"`, pois estruturas de árvore não são a escolha natural para essa métrica e o catálogo é minúsculo.

Ao consultar cada música ouvida, buscar vizinhos, descartar o próprio item e qualquer item já ouvido. Converter distância em similaridade, ignorar similaridade negativa se surgir e acumular o score por candidato. Recomenda-se comparar duas políticas:

1. soma simples das similaridades;
2. soma ponderada por uma transformação de `play_count`, como `log1p`, para reduzir dominância de contagens altas.

### 9.3 Casos de fallback

- Usuário desconhecido ou sem histórico: recomendar populares globais com rótulo explícito de fallback.
- Nenhum candidato elegível: retornar mensagem clara e lista vazia.
- `k` maior que o catálogo: limitar à quantidade disponível e informar o ajuste.
- Item sem vizinhos informativos: não fabricar score; recorrer ao baseline.

## 10. Protocolo de avaliação

### 10.1 Leave-one-out por usuário

Agregue primeiro. Para cada usuário, escolha a música de maior contagem agregada; em empate, aplique regra estável. Oculte essa interação do conjunto de treino, ajuste o modelo somente com os dados remanescentes e verifique se o item oculto aparece nas Top k.

Com apenas três usuários, cada erro altera o Hit Rate em 33,33 pontos percentuais. O valor deve ser apresentado como contagem (`hits/3`) e percentual, nunca isoladamente.

### 10.2 Métricas

- `HitRate@k = usuários com item oculto recuperado / usuários avaliáveis`.
- `Precision@k = total de itens ocultos recuperados / total de posições recomendadas`; com um relevante por usuário, o máximo é `1/k`.
- `MRR@k`, recomendado, diferencia acerto na primeira posição de acerto nas posições finais.
- Cobertura de catálogo, recomendada, mede quantas músicas distintas aparecem nas listas.

### 10.3 Baselines e controles

Comparar o KNN com recomendação de popularidade global, sempre removendo itens já ouvidos. Se o KNN não superar o baseline na mesma divisão, a complexidade não está justificada. Também registrar `k`, `n_neighbors`, normalização, política de score e tratamento de empates.

## 11. Plano de experimentos com MLflow

Usar um experimento como `playcatch-recsys`. O backend local pode ser `sqlite:///mlruns.db`; artefatos devem permanecer em diretório separado e conhecido. Cada run deve registrar:

- parâmetros: `n_neighbors`, `k`, `norm`, `score_policy`, algoritmo, métrica e seed/regra de desempate;
- métricas: Hit Rate@k, Precision@k, MRR@k, cobertura e número de usuários avaliáveis;
- tags: versão do código, identificador/hash do dataset, ambiente e finalidade da run;
- artefatos: ranking de popularidade, shape/densidade, amostra de recomendações, mapa `song_to_idx`, configuração e relatório da avaliação;
- modelo: `NearestNeighbors` com flavor scikit-learn, acompanhado dos mapeamentos e do pré-processamento necessário.

Não expor o servidor MLflow em `0.0.0.0` sem configuração de segurança. SQLite é adequado para POC local e baixa concorrência; colaboração multiusuário exige backend e armazenamento mais robustos.

## 12. Interface Gradio planejada

### 12.1 Contrato mínimo

Entrada: `user_id` selecionável entre usuários conhecidos e `k` inteiro em faixa segura, por exemplo 1 a 10. Saída: tabela com posição, `song_id`, score agregado, origem (`personalizado` ou `popularidade`) e observação curta.

### 12.2 Critérios de usabilidade

- Valor padrão válido e exemplos clicáveis.
- Validação antes de chamar o recomendador.
- Mensagens amigáveis para usuário desconhecido e catálogo esgotado.
- Indicação de que os IDs são fictícios e que a POC não utiliza áudio ou metadados musicais.
- Não ativar link público compartilhável por padrão em dados reais.

### 12.3 Lacuna a resolver

Como o enunciado não especifica a etapa Gradio, esses critérios devem ser aceitos como contrato do projeto antes da implementação. Sem isso, o checklist “exibe corretamente” não é verificável.

## 13. Testes necessários

| Teste | Objetivo | Critério esperado |
|---|---|---|
| Esquema válido | Detectar colunas/tipos ausentes | Falha clara antes da modelagem |
| Agregação repetida | Consolidar chaves duplicadas | `(2,101)=11` e `(2,104)=8` |
| Pivot | Evitar erro por chave repetida | Matriz 3 x 8, sem `NaN` |
| Orientação | Confirmar item-based | Matriz de treino 8 x 3 |
| Auto-vizinho | Excluir a própria música | Candidato nunca é o item consultado |
| Itens conhecidos | Não recomendar já ouvidas | Interseção vazia |
| Ordenação | Garantir desempate determinístico | Mesma entrada, mesma saída |
| Usuário inexistente | Acionar fallback/erro contratual | Sem stack trace |
| `k` inválido | Validar limite | Mensagem ou ajuste documentado |
| Métricas | Verificar fórmulas em exemplo manual | Valores exatos conhecidos |
| MLflow | Confirmar params, métricas e artefatos | Run completa e recuperável |
| Gradio | Testar entrada e saída | Resposta tabular coerente |

## 14. Aceite da implementação futura (não executada)

| Critério | Evidência exigida | Status atual |
|---|---|---|
| Dataset carregado e auditado | Relatório de esquema, ausências e duplicidades | Planejado |
| Agregação correta | Pares repetidos somados e valores conferidos | Planejado |
| Matriz sem ausências | Shape 3 x 8 e zeros explícitos | Planejado |
| Recomendador item-based | Código, testes e exemplo por usuário | Planejado |
| Itens já ouvidos excluídos | Teste automatizado | Planejado |
| Validação sem leakage | Protocolo e folds persistidos | Planejado |
| Métricas calculadas | Hit Rate@k e métricas auxiliares | Planejado |
| Experimentos rastreados | Banco SQLite e artefatos | Planejado |
| Modelo recuperável | Modelo, mapeamentos e pré-processamento | Planejado |
| Interface Gradio | Teste funcional e captura/evidência | Planejado |
| Documentação reprodutível | README, lock de dependências e comandos | Planejado |

Estes critérios pertencem à futura construção do sistema. Por integridade documental, nenhum deles é marcado como “Feito” neste levantamento: a implementação, a instalação de dependências, o treinamento, a execução do MLflow e a publicação do Gradio não foram autorizados nem realizados. A tabela de aceite do **levantamento**, já concluído, encontra-se na Seção 21.

## 15. Riscos e mitigação

| Risco | Impacto | Mitigação |
|---|---|---|
| Dataset minúsculo | Métricas instáveis e pouca generalização | Rotular como POC; coletar base maior e temporal |
| Confundir eventos repetidos com duplicatas | Perda de sinal de preferência | Remover apenas linhas idênticas e agregar chaves |
| Vazamento na validação | Métrica artificialmente alta | Reajustar dados/modelo dentro de cada fold |
| Popularidade dominar | Recomendações pouco personalizadas | Normalização, score ponderado e baseline explícito |
| Cold start | Sem recomendação para novos usuários/itens | Popularidade e futura abordagem híbrida |
| Dependências mutáveis | Quebra de reprodução | Fixar versões após validação e registrar ambiente |
| SQLite em uso concorrente | Locking e baixa escalabilidade | Restringir à POC local; migrar em cenário multiusuário |
| Exposição de Gradio/MLflow | Acesso indevido | Bind local, autenticação/rede segura e dados fictícios |
| Privacidade do histórico | Inferência de hábitos sensíveis | Minimização, pseudonimização, controle de acesso e retenção |

## 16. Três análises críticas da resposta

### Análise crítica 1 - Aderência e rastreabilidade

A primeira análise verificou cada frase do enunciado contra requisitos, dependências e critérios de aceite. Ela revelou que a interface Gradio aparece no objetivo e no checklist, mas não possui etapa operacional. Também distinguiu bibliotecas apenas importadas de bibliotecas realmente necessárias. Como correção, o relatório incluiu um contrato mínimo de interface e uma matriz de dependências por necessidade.

### Análise crítica 2 - Validade técnica e estatística

A segunda análise reconstruiu manualmente o dataset. Ela encontrou pares repetidos que inviabilizam `pivot()`, o empate total entre usuários, o empate na quinta música e a extrema sensibilidade de métricas com três usuários. Como correção, foram especificados `pivot_table` com soma, regra de desempate, apresentação de métricas como fração e baseline de popularidade.

### Análise crítica 3 - Operação, segurança e reprodutibilidade

A terceira análise examinou instalação, tracking e exposição de serviços. Ela identificou o erro potencial de `pip install sqlite`, a diferença entre backend e artefatos no MLflow, a efemeridade do Colab e o risco de compartilhar Gradio/MLflow sem controle. Como correção, o relatório recomenda ambiente isolado, lock de versões, persistência explícita e bind local por padrão.

## 17. Três revisões realizadas

### Revisão 1 - Cobertura estrutural

Foi conferida a presença de objetivo, escopo, persona, conhecimentos, softwares, arquitetura, dados, modelo, validação, MLflow, Gradio, testes, riscos, critérios de aceite e fontes. Itens ausentes foram acrescentados e afirmações de execução foram removidas.

### Revisão 2 - Consistência cruzada

Foram conciliados o texto inicial, os passos e o checklist. A revisão corrigiu a orientação usuário-item versus item-usuário, separou dependências obrigatórias de opcionais e alinhou o status de todos os critérios como “Planejado”.

### Revisão 3 - Clareza executiva e verificabilidade

Foram reduzidas ambiguidades, transformadas recomendações em decisões verificáveis e associados riscos a mitigações. O documento final evita prometer desempenho, não inventa resultados e indica quais evidências serão exigidas na fase de execução.

## 18. Sequência recomendada para a futura execução

1. Aprovar o contrato de Gradio e as regras de desempate/scoring.
2. Criar ambiente isolado e registrar versões.
3. Carregar e validar o CSV sem alterar silenciosamente eventos.
4. Produzir EDA e artefatos descritivos.
5. Implementar baseline de popularidade.
6. Implementar KNN item-based e testes.
7. Construir validação leave-one-out sem vazamento.
8. Executar matriz de experimentos no MLflow.
9. Selecionar configuração com justificativa e limitações.
10. Integrar Gradio e testar casos normais/limites.
11. Gerar pacote reprodutível e relatório de resultados reais.

## 19. Necessidades de informação antes do início da implementação

O levantamento está concluído, mas a implementação responsável depende de decisões que o enunciado não fornece. Os itens classificados como **Bloqueador** devem ser respondidos antes de produzir resultados que possam ser chamados de aceitos; os itens **Definição de projeto** podem receber um padrão provisório na POC, desde que registrados.

| Informação necessária | Por que é necessária | Prioridade | Padrão provisório, se autorizado |
|---|---|---|---|
| Arquivo-fonte oficial e responsável pelo dataset | Evitar implementar sobre uma transcrição incompleta e permitir hash/versionamento | Bloqueador | Usar o CSV do enunciado, registrando hash e origem |
| Esquema, tipos, unidade de `play_count` e semântica de evento repetido | Define validação, agregação e tratamento de duplicidades | Bloqueador | Contagem inteira positiva; somar eventos do mesmo par |
| Semântica e fuso de `last_played` | Evita interpretações temporais incorretas | Definição de projeto | Datetime ingênuo, somente para EDA nesta POC |
| Objetivo de negócio e definição de relevância | Determina se otimizar clique, reprodução, retenção ou descoberta | Bloqueador | Recuperar uma interação ocultada como proxy didática |
| Valor padrão de `k`, faixa permitida e regra de desempate | Torna ranking, testes e interface determinísticos | Definição de projeto | `k=5`; score decrescente e `song_id` crescente |
| Política de score: soma simples ou ponderada | Muda a ordenação das recomendações | Definição de projeto | Soma simples; ponderação como experimento separado |
| Catálogo com título, artista e disponibilidade | IDs numéricos não são uma experiência musical utilizável | Bloqueador para demonstração de negócio | Exibir apenas `song_id`, rotulando a limitação |
| Regra para usuário/item desconhecido | Define cold start e mensagens da interface | Definição de projeto | Popularidade global com rótulo de fallback |
| Contrato visual e público da interface Gradio | Define campos, linguagem, acesso e critérios de usabilidade | Bloqueador para aceite da interface | Entrada `user_id`/`k`, saída tabular, acesso local |
| Ambiente-alvo: Colab ou local, SO, Python e restrições de rede | Define comandos, persistência e compatibilidade | Bloqueador | Python 3.10–3.12 em ambiente virtual local |
| Versões aprovadas e política de atualização | APIs de MLflow/Gradio podem variar entre versões | Definição de projeto | Resolver uma vez, testar e gerar lock |
| Local, retenção e backup de banco/artefatos MLflow | Garante recuperação dos experimentos | Bloqueador para rastreabilidade | SQLite e diretório de artefatos locais separados |
| Classificação LGPD, base legal, retenção e controles de acesso | Histórico de consumo pode revelar preferências pessoais | Bloqueador para dados reais | Somente dados fictícios/pseudonimizados na POC |
| Baseline, métricas e limiar de aprovação | Impede declarar sucesso sem referência objetiva | Bloqueador | Popularidade + Hit Rate@k, MRR@k e cobertura; sem meta produtiva |
| Responsáveis pela aprovação técnica e de negócio | Define quem aceita limitações e resultados | Bloqueador para encerramento | Registrar aprovadores no README/relatório final |

### 19.1 Inconsistências consolidadas

Além das ambiguidades da Seção 3.4, a revisão final consolidou os seguintes conflitos que precisam ser conhecidos pela equipe:

- O título recebido menciona “Categorização Automática de Músicas”, enquanto os requisitos descrevem recomendação colaborativa. São problemas distintos; o escopo analisado é **recomendação**.
- O enunciado pede “Top 5 usuários”, mas existem somente três usuários. O resultado correto contém três posições empatadas, nunca cinco linhas inventadas.
- `pivot()` é incompatível com as chaves usuário-música repetidas; a agregação por soma deve preceder a matriz ou ser feita por `pivot_table`.
- A matriz é descrita como usuário x item, mas um KNN item-based exige itens como amostras no ajuste; portanto, é necessária a transposição.
- A instrução de “excluir duplicatas” conflita com eventos repetidos potencialmente legítimos. Só duplicatas integrais devem ser removidas; repetições do par devem ser agregadas.
- `transformers`/`pipeline` não têm função no algoritmo solicitado, e `sqlite` não é um pacote que deva ser instalado via `pip` para este caso.
- A importação de `train_test_split` não implementa o protocolo leave-one-out descrito e pode induzir uma divisão incompatível com o requisito.
- A exigência de Gradio não vem acompanhada de uma etapa, layout, validação, catálogo legível ou contrato de saída.
- `last_played` é solicitado na preparação, mas não participa do modelo nem da avaliação; sua finalidade precisa ser limitada à EDA ou explicitamente incorporada.
- A escolha do item oculto, do número de vizinhos, da inclusão do próprio item e da ponderação do score não está completamente definida.
- Com oito músicas — e menos itens em cada fold — `n_neighbors` deve ser limitado dinamicamente ao conjunto treinável.
- O exemplo de logging do modelo pode depender da versão do MLflow; a assinatura efetivamente instalada deve ser testada e fixada.
- Backend SQLite e armazenamento de artefatos são responsabilidades diferentes, embora o enunciado possa sugerir que o banco resolve ambos.
- Não há metadados de catálogo; logo, a interface só consegue apresentar IDs, não nomes de músicas e artistas.
- A base de três dias e três usuários não permite estabelecer limiar de qualidade produtiva, significância ou generalização.

## 20. Passo a passo, testes e evidências do levantamento

O procedimento abaixo descreve o que foi efetivamente realizado nesta fase documental. Ele não representa execução do recomendador.

1. Delimitação do pedido e separação entre instruções do usuário, conteúdo do desafio e conteúdo meramente referencial do anexo.
2. Leitura integral do desafio e inventário de objetivos, etapas, bibliotecas, entregáveis e critérios explícitos.
3. Inspeção do relatório anexo para identificar linguagem técnico-executiva, hierarquia, estilos, tabelas, blocos de código e padrão de evidências.
4. Construção da matriz de rastreabilidade entre requisitos, conhecimentos, persona, softwares, riscos, testes e critérios futuros.
5. Reconstrução analítica do dataset: contagem de linhas, usuários, músicas, pares, reproduções, duplicidades de chave, matriz e densidade.
6. Verificação de coerência do algoritmo: orientação item-based, normalização, distância cosseno, filtragem de itens vistos, score e fallbacks.
7. Verificação de coerência da avaliação: agregação anterior ao holdout, desempate, prevenção de leakage, métricas e baseline.
8. Auditoria de instalação e operação: dependências mínimas, SQLite, MLflow, Gradio, persistência, segurança e reprodutibilidade.
9. Execução das três análises críticas da Seção 16 e aplicação das correções encontradas.
10. Execução das três revisões da Seção 17, cobrindo estrutura, consistência cruzada e clareza/verificabilidade.
11. Geração equivalente dos formatos Markdown e DOCX, seguida de inspeção estrutural, renderização visual e auditoria de acessibilidade do documento Word.

### 20.1 Testes do levantamento e resultados

| ID | Teste realizado sobre o levantamento | Resultado observado | Status |
|---|---|---|---|
| EV-01 | Disponibilidade e separação das fontes | Desafio tratado como requisito; anexo tratado como referência editorial, não como instrução | Feito |
| EV-02 | Cobertura de requisitos | As 13 exigências funcionais foram mapeadas para arquitetura, conhecimento, software, teste ou decisão | Feito |
| EV-03 | Reconciliação aritmética do dataset | 16 linhas, 3 usuários, 8 músicas, 13 pares, 60 reproduções, matriz 3 x 8 e densidade 54,17% | Feito |
| EV-04 | Detecção de chaves repetidas | `(2,101)=11` e `(2,104)=8` após soma; risco de falha de `pivot()` registrado | Feito |
| EV-05 | Conferência dos rankings | Usuários empatados em 20; totais por música reconciliados e empate da quinta posição identificado | Feito |
| EV-06 | Coerência do desenho item-based | Transposição 8 x 3, distância cosseno, retirada do auto-vizinho e de itens vistos especificadas | Feito |
| EV-07 | Auditoria de dependências | Pacotes obrigatórios, opcionais e indevidos separados; `sqlite3` corretamente tratado como biblioteca padrão | Feito |
| EV-08 | Coerência do protocolo de avaliação | Leave-one-out, regra de desempate, baseline e risco de leakage explicitados | Feito |
| EV-09 | Integridade de escopo | Nenhuma instalação, execução, métrica de modelo ou evidência de interface foi falsamente declarada | Feito |
| EV-10 | Três análises críticas | Aderência, validade estatística e operação/segurança concluídas com correções incorporadas | Feito |
| EV-11 | Três revisões | Cobertura, consistência e verificabilidade concluídas | Feito |
| EV-12 | Equivalência dos artefatos | O DOCX é gerado a partir do conteúdo do Markdown, preservando a mesma informação | Feito |
| EV-13 | Verificação visual do DOCX | Todas as páginas renderizadas e inspecionadas quanto a corte, sobreposição e legibilidade | Feito |
| EV-14 | Auditoria estrutural/acessibilidade | Hierarquia, tabelas e propriedades do documento verificadas sem achados impeditivos | Feito |

Os “resultados” acima pertencem aos testes do **levantamento**. Os resultados esperados dos testes do sistema estão na Seção 13 e só poderão ser preenchidos após autorização e execução da implementação.

## 21. Critérios de aceite do levantamento

| Critério de aceite desta entrega | Evidência | Status |
|---|---|---|
| Desafio analisado em profundidade | Seções 3, 4, 8, 9, 10, 11 e 12; EV-02 a EV-08 | Feito |
| Anexo analisado sem incorporar suas instruções | Seção 2 e EV-01 | Feito |
| Persona e responsabilidades definidas | Seção 5 | Feito |
| Conhecimentos necessários identificados | Seção 6 | Feito |
| Softwares, versões-alvo e dependências mapeados | Seção 7 e EV-07 | Feito |
| Arquitetura e sequência futura propostas | Seções 8 e 18 | Feito |
| Testes sugeridos e resultados esperados documentados | Seção 13 | Feito |
| Passo a passo do levantamento documentado | Seção 20 | Feito |
| Resultados e evidências do levantamento registrados | Seção 20.1, EV-01 a EV-14 | Feito |
| Três análises críticas realizadas | Seção 16 e EV-10 | Feito |
| Três revisões realizadas | Seção 17 e EV-11 | Feito |
| Necessidades de informação para implementação descritas | Seção 19 | Feito |
| Inconsistências consolidadas e decisões propostas | Seções 3.4 e 19.1 | Feito |
| Critérios da implementação separados dos critérios do estudo | Seções 14 e 21; EV-09 | Feito |
| Relatórios equivalentes em `.md` e `.docx` produzidos | EV-12 | Feito |
| Qualidade visual e estrutural do DOCX verificada | EV-13 e EV-14 | Feito |

## 22. Fontes técnicas consultadas

- Scikit-learn, instalação e ambientes isolados: https://scikit-learn.org/stable/install.html
- Scikit-learn, Nearest Neighbors e métrica cosseno: https://scikit-learn.org/stable/modules/neighbors.html
- Scikit-learn, definição de similaridade cosseno: https://scikit-learn.org/stable/modules/metrics.html
- MLflow, tracking com banco SQLite local: https://www.mlflow.org/docs/latest/ml/tracking/tutorials/local-database/
- MLflow, arquitetura do tracking server: https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/
- MLflow, API Python e logging de modelos scikit-learn: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html
- Gradio, quickstart e requisito de Python: https://www.gradio.app/guides/quickstart
- Python, módulo `sqlite3`: https://docs.python.org/3/library/sqlite3.html

## 23. Conclusão

O desafio é adequado para consolidar EDA, filtragem colaborativa item-based, avaliação offline, MLOps local e interface de demonstração. Sua execução exige mais cuidado conceitual do que o pequeno volume de código sugere: agregação correta, orientação da matriz, prevenção de vazamento, definição de relevância e preservação de mapeamentos são essenciais.

O conjunto fornecido permite testar o fluxo, mas não sustenta conclusão sobre qualidade. A futura entrega deve ser apresentada como POC, comparada contra popularidade e acompanhada de limitações explícitas. Com as correções propostas - especialmente `pivot_table` com soma, remoção de dependências indevidas, protocolo leave-one-out e contrato Gradio - o projeto fica tecnicamente coerente, reproduzível e pronto para uma implementação responsável em etapa posterior.
