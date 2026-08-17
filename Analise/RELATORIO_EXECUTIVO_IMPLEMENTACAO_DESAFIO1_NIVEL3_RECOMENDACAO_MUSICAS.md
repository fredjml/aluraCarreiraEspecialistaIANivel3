# RELATÓRIO TÉCNICO-EXECUTIVO DE IMPLEMENTAÇÃO

## Desafio 1 - Nível 3: Sistema de Recomendação de Músicas da Playcatch

**Data da implementação:** 17 de agosto de 2026  
**Situação:** implementação concluída, testada e documentada  
**Escopo executado:** dados, EDA, KNN item-based, validação leave-one-out, MLflow/SQLite, Gradio, testes e evidências  
**Expansão autorizada:** 10 usuários e 10 músicas sintéticos, preservando e identificando os dados originais

---

## 1. Sumário executivo

O desafio foi implementado integralmente como prova de conceito reproduzível. O repositório contém geração determinística dos dados, validação de esquema, agregação de eventos repetidos, matriz usuário-item, recomendador item-based com distância cosseno, fallback de popularidade, avaliação leave-one-out, três experimentos rastreados pelo MLflow em SQLite, interface Gradio e sete testes automatizados.

A base original foi preservada com 16 eventos, 3 usuários, 8 músicas, 14 pares únicos e 60 reproduções. A expansão sintética com `seed=42` acrescentou exatamente 10 usuários e 10 músicas novas, resultando em 79 eventos, 13 usuários, 18 músicas, 77 pares únicos e 465 reproduções. A coluna `data_origin` distingue `original` de `synthetic`.

Os sete testes passaram. A interface Gradio respondeu HTTP 200 e apresentou o título Playcatch. O banco MLflow contém três runs finalizadas do experimento `playcatch-recsys`, com parâmetros, métricas, modelo e artefatos. A melhor configuração sintética obteve Hit Rate@5 de 1,0, Precision@5 de 0,20, MRR@5 de 0,8769 e cobertura de 83,33%. Esse resultado comprova o funcionamento do cenário controlado, não desempenho em produção.

## 2. Correção resultante da implementação

A execução revelou e corrigiu uma contagem do levantamento anterior. A base original possui **14 pares únicos usuário-música**, não 13: usuário 1 possui cinco pares, usuário 2 possui três e usuário 3 possui seis. Logo, a densidade correta da matriz 3 x 8 é `14/24 = 58,33%`. As chaves `(2,101)` e `(2,104)` continuam sendo eventos repetidos e são somadas para 11 e 8, respectivamente.

## 3. Arquitetura implementada

```text
dados originais + gerador sintético(seed=42)
                 |
                 v
       validação e auditoria
                 |
                 v
   agregação usuário-música + EDA
                 |
                 v
       matriz usuário x item
                 |
                 v
 matriz item x usuário + normalização
                 |
                 v
 NearestNeighbors(metric="cosine", algorithm="brute")
          |                       |
          v                       v
 recomendação/fallback     leave-one-out
          |                       |
          +----------+------------+
                     v
           MLflow + SQLite
                     |
                     v
                Gradio local
```

| Componente | Responsabilidade | Arquivo principal |
|---|---|---|
| Dados | Originais, sintéticos, validação e matriz | `src/playcatch/data.py` |
| Modelo | KNN, candidatos, filtragem e fallback | `src/playcatch/recommender.py` |
| Avaliação | Holdout por usuário e métricas | `src/playcatch/evaluate.py` |
| Pipeline | EDA, artefatos e resumos | `src/playcatch/pipeline.py` |
| Tracking | Runs, métricas, modelo e artefatos | `src/playcatch/tracking.py` |
| Interface | Entrada de usuário/k e saída tabular | `app.py` |
| Automação | Execução completa de três configurações | `scripts/run_pipeline.py` |
| Verificação | Banco e manifesto de artefatos | `scripts/verify_repository.py` |

## 4. Passo a passo executado

1. O enunciado e o estudo anterior foram convertidos em requisitos verificáveis.
2. Foi criado um projeto Python isolado, com dependências declaradas em `pyproject.toml` e resolvidas em `uv.lock`.
3. Os 16 registros originais foram codificados sem alteração e receberam `data_origin=original`.
4. Foi implementado um gerador determinístico com `seed=42`, dez usuários (4 a 13), dez músicas novas (109 a 118), grupos de preferência e itens-ponte.
5. Foram produzidos `user_data_original.csv` e `user_data_expanded.csv`.
6. O carregamento valida colunas, ausências, datas, tipos e contagens positivas; duplicatas integrais são removidas.
7. Eventos repetidos do mesmo par são somados antes da matriz dinâmica.
8. A matriz usuário-item é transposta para item-usuário e normalizada por L1 ou L2.
9. O `NearestNeighbors` foi ajustado com cosseno e força bruta, limitando vizinhos ao catálogo treinável.
10. A recomendação exclui auto-vizinho e itens já consumidos, soma similaridades, aplica desempate por `song_id` e usa popularidade como fallback.
11. A avaliação oculta o item agregado mais consumido de cada usuário, reajusta o modelo e calcula Hit Rate, Precision, MRR e cobertura.
12. O pipeline original foi executado separadamente para preservar uma referência sem dados sintéticos.
13. Três configurações foram executadas e registradas no MLflow/SQLite.
14. O modelo scikit-learn, parâmetros, métricas, hashes, mapeamentos e CSVs de evidência foram persistidos.
15. Sete testes automatizados foram executados e aprovados.
16. A aplicação Gradio foi iniciada localmente, consultada por HTTP e encerrada após a comprovação.
17. O banco e o manifesto de artefatos foram validados por script independente.

## 5. Dados e EDA executados

| Medida | Base original | Base expandida |
|---|---:|---:|
| Eventos brutos | 16 | 79 |
| Usuários | 3 | 13 |
| Músicas | 8 | 18 |
| Pares únicos | 14 | 77 |
| Reproduções | 60 | 465 |
| Dimensão da matriz | 3 x 8 | 13 x 18 |
| Densidade | 58,33% | 32,91% |
| Ausências | 0 | 0 |
| Duplicatas integrais removidas | 0 | 0 |
| Linhas pertencentes a pares repetidos | 4 | 4 |

O hash SHA-256 da base original é `a6bcef334c7fb0c871f8daaf2bea45fae4642986bc7393b1b676c425de610878`. O hash da base expandida é `a3727f7087fa3c902d4d5f80dfc15c1a76bb0b3a53a16c57b399d2ee196c0d63`.

## 6. Estratégia sintética aplicada

Os usuários sintéticos foram distribuídos por três grupos de preferência. Cada grupo combina músicas originais e novas, com contagens decrescentes e pequena variação pseudoaleatória. Itens-ponte criam coocorrência entre grupos. A seed fixa garante que duas execuções produzam o mesmo CSV.

Essa expansão tem três finalidades: testar mais usuários que o “Top 5” solicitado, garantir candidatos não ouvidos e permitir comparar hiperparâmetros. Ela não representa comportamento real, não serve para inferência de produto e não deve ser usada para declarar prontidão produtiva.

## 7. Modelo e regras implementadas

- Matriz de treino: músicas nas linhas, usuários nas colunas.
- Normalização: L1 ou L2 por vetor de item.
- Modelo: `NearestNeighbors(metric="cosine", algorithm="brute")`.
- Similaridade: `max(0, 1 - distância)`.
- Score simples: soma das similaridades.
- Score ponderado: similaridade multiplicada por `log1p(play_count)`.
- Filtro: nenhum item já ouvido pode aparecer.
- Desempate: score decrescente e `song_id` crescente.
- Usuário desconhecido ou ausência de candidatos: popularidade global rotulada como fallback.
- `k <= 0`: erro contratual claro.

## 8. Avaliação e resultados

### 8.1 Base original

Com `n_neighbors=5`, `k=5`, normalização L2 e score simples, foram avaliados três usuários: 2 acertos em 3, Hit Rate@5 de 66,67%, Precision@5 de 13,33%, MRR@5 de 0,15 e cobertura de catálogo de 100%. Cada erro altera o Hit Rate em 33,33 pontos percentuais, portanto o resultado é apenas uma verificação funcional.

### 8.2 Base expandida e MLflow

| Run ID | Vizinhos | k | Norma | Score | Hit Rate | Precision | MRR | Cobertura |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `39228a3ee3b94562943ad44e24cdfb8b` | 3 | 3 | L2 | simples | 38,46% | 12,82% | 0,3333 | 50,00% |
| `21ebd2be0c7f47a68ee898f7285254e4` | 5 | 5 | L2 | simples | 46,15% | 9,23% | 0,3385 | 83,33% |
| `a643c1b137a7444ea60c43ae46d06a01` | 7 | 5 | L1 | `weighted_log1p` | 100,00% | 20,00% | 0,8769 | 83,33% |

A terceira configuração venceu no cenário controlado. O valor perfeito é consequência dos padrões sintéticos deliberadamente recuperáveis e deve ser interpretado como teste de integração, não estimativa de qualidade real.

## 9. Testes executados

Comando executado: `uv run pytest -q -p no:cacheprovider`.

| Teste | Evidência verificada | Resultado | Status |
|---|---|---|---|
| Agregação original | Matriz 3 x 8, `(2,101)=11`, `(2,104)=8`, total 60 | Passou | Feito |
| Escopo sintético | 10 usuários, músicas 109–118 e origem sintética | Passou | Feito |
| Reprodutibilidade | Duas gerações com seed 42 idênticas | Passou | Feito |
| Exclusão de itens vistos | Interseção entre histórico e recomendações vazia | Passou | Feito |
| Ordenação | Scores em ordem decrescente | Passou | Feito |
| Usuário desconhecido | Três itens de fallback de popularidade | Passou | Feito |
| `k` inválido | `ValueError` claro para zero | Passou | Feito |
| Métricas | 13 usuários e métricas no intervalo [0,1] | Passou | Feito |
| Callback Gradio | Esquema `rank/song_id/score/origin` | Passou | Feito |

Resultado consolidado do pytest: **7 passed in 10.18s**.

## 10. Evidências de execução

| ID | Evidência | Resultado | Status |
|---|---|---|---|
| EV-I01 | Ambiente resolvido pelo uv | 119 pacotes instalados em `.venv` | Feito |
| EV-I02 | Testes automatizados | 7/7 aprovados | Feito |
| EV-I03 | Dataset original | CSV criado e hash registrado | Feito |
| EV-I04 | Dataset expandido | 13 usuários, 18 músicas e origem identificada | Feito |
| EV-I05 | EDA e matriz | CSVs de agregação, ranking, matriz e resumo | Feito |
| EV-I06 | Avaliação original | 3 usuários, 2 hits e métricas persistidas | Feito |
| EV-I07 | Experimentos | 3 runs MLflow finalizadas | Feito |
| EV-I08 | Banco de tracking | `mlruns.db` consultado com sucesso | Feito |
| EV-I09 | Modelos | KNN registrado em cada run | Feito |
| EV-I10 | Artefatos | Mapeamentos, rankings, amostras e detalhes leave-one-out | Feito |
| EV-I11 | Gradio | HTTP 200 e título Playcatch encontrado | Feito |
| EV-I12 | Segurança local | Gradio sem share e instrução MLflow em 127.0.0.1 | Feito |
| EV-I13 | Documentação | README, relatório Markdown e relatório Word | Feito |
| EV-I14 | Reprodutibilidade | `pyproject.toml`, `uv.lock`, seed e hashes | Feito |

## 11. Critérios de aceite

| Critério | Evidência | Status |
|---|---|---|
| Dataset carregado, tipado e auditado | EV-I03 a EV-I05 | Feito |
| Valores ausentes e duplicatas tratados | Auditoria com zero ausências e zero duplicatas integrais | Feito |
| Pares repetidos agregados corretamente | Teste de 101/104 do usuário 2 | Feito |
| Estatísticas, Top 5 e distribuições gerados | `song_popularity.csv` e `user_volume.csv` | Feito |
| Matriz usuário-item sem NaN | CSV e teste de shape | Feito |
| Dez usuários e dez músicas adicionados | EV-I04 | Feito |
| Dados originais e sintéticos distinguíveis | Coluna `data_origin` | Feito |
| KNN item-based com cosseno ajustado | Código, testes e modelo MLflow | Feito |
| Itens ouvidos excluídos | Teste automatizado | Feito |
| Fallback e entradas inválidas tratados | Testes automatizados | Feito |
| Leave-one-out sem vazamento implementado | Detalhes persistidos por usuário | Feito |
| Hit Rate e Precision calculados | Resumos original e expandido | Feito |
| MRR e cobertura calculados | Resumos e MLflow | Feito |
| SQLite configurado como backend | `mlruns.db` válido | Feito |
| Múltiplos experimentos registrados | Três run IDs finalizadas | Feito |
| Parâmetros, métricas, modelos e artefatos logados | EV-I07 a EV-I10 | Feito |
| Interface Gradio funcional | HTTP 200 e teste do callback | Feito |
| Testes automatizados aprovados | 7/7 | Feito |
| Instruções de comprovação documentadas | Seção 14 e README | Feito |
| Três análises críticas realizadas | Seção 12 | Feito |
| Três revisões realizadas | Seção 13 | Feito |
| Relatórios `.md` e `.docx` produzidos | Arquivos no diretório `Analise` | Feito |

## 12. Três análises críticas da implementação

### Análise crítica 1 - Correção dos dados e do algoritmo

Foi revisado o fluxo entre evento, par agregado, matriz e orientação item-based. A execução detectou a contagem correta de 14 pares originais e impediu que `pivot()` fosse usado sobre chaves repetidas. Também foram examinados auto-vizinhos, itens vistos, desempates e limites de vizinhos. Resultado: regras consolidadas em funções testadas e correção documental registrada.

### Análise crítica 2 - Validade da avaliação

Foi verificado se o item oculto é removido antes do ajuste de cada fold e se as métricas são calculadas somente sobre recomendações produzidas pelo treino correspondente. O Hit Rate perfeito da melhor configuração foi confrontado com a construção sintética. Resultado: o relatório separa explicitamente comprovação funcional de validade externa e mantém os resultados originais em bloco próprio.

### Análise crítica 3 - Operação, segurança e reprodutibilidade

Foram avaliados ambiente, lock, seed, hashes, persistência, tracking e exposição HTTP. O Gradio é iniciado sem link público, o MLflow é orientado para `127.0.0.1`, e o banco é local. Resultado: repositório reproduzível para POC, com alerta de que SQLite e a interface local não são arquitetura produtiva multiusuário.

## 13. Três revisões realizadas

### Revisão 1 - Cobertura de requisitos

Cada requisito do enunciado foi associado a código, teste, artefato ou comando de verificação. A revisão confirmou dados, EDA, modelo, avaliação, MLflow, SQLite, Gradio, expansão sintética e documentação.

### Revisão 2 - Consistência cruzada

Foram comparados os valores de `summary.json`, `execution_summary.json`, CSVs, testes e banco MLflow. Foram conciliados 79 eventos, 13 usuários, 18 músicas, 77 pares, 465 reproduções e os três conjuntos de métricas.

### Revisão 3 - Entrega e verificabilidade

Foram revisados comandos do README, nomes de arquivos, mensagens de erro, evidências e critérios de aceite. A revisão garantiu que todos os itens “Feito” possuem uma evidência verificável e que limitações sintéticas não foram omitidas.

## 14. Como testar e comprovar

No PowerShell, a partir da raiz do projeto:

```text
$env:UV_CACHE_DIR='E:\ProjAlura\ProjNivel3\.uv-cache'
uv sync --extra dev
uv run pytest -q -p no:cacheprovider
uv run python scripts\run_pipeline.py
uv run python scripts\verify_repository.py
uv run python app.py
```

Na interface Gradio, selecione usuários 1, 4 e 999 por meio do callback/teste; para a interface visual, use os usuários oferecidos no dropdown e varie `k` entre 1 e 10. Confirme que itens já ouvidos não aparecem e que a tabela contém rank, música, score e origem.

Para consultar o MLflow:

```text
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
```

Abra `http://127.0.0.1:5000`, selecione `playcatch-recsys` e compare as três runs. Os artefatos também podem ser inspecionados em `artifacts/mlflow/`.

## 15. Limitações e próximos passos

- Substituir dados sintéticos por histórico real anonimizado e autorizado.
- Adicionar catálogo com título, artista, gênero e disponibilidade.
- Criar divisão temporal e avaliação online/A-B antes de decisões de produto.
- Empregar armazenamento e backend adequados a concorrência real.
- Empacotar o recomendador completo, incluindo pré-processamento e mapeamentos, como modelo de serviço.
- Incluir autenticação, observabilidade e política LGPD antes de exposição externa.

## 16. Conclusão

A implementação satisfaz o desafio como POC: o pipeline é executável, testado, rastreável e acessível por interface local. Os artefatos comprovam o processamento, a recomendação, a avaliação, o tracking e a apresentação. A expansão autorizada resolve a limitação operacional de apenas três usuários para fins de teste, preservando a distinção entre dados reais do enunciado e dados sintéticos.

O resultado mais importante não é o Hit Rate perfeito da configuração sintética, mas a existência de um fluxo verificável que pode ser reexecutado, auditado e substituído por dados reais sem alterar o contrato principal.

## 17. Repositório

**Repositório remoto:** https://github.com/fredjml/aluraCarreiraEspecialistaIA  
**Diretório do projeto:** `ProjNivel3/`  
**Raiz local:** `E:\ProjAlura\ProjNivel3`  
**Documentação operacional:** `README.md`
