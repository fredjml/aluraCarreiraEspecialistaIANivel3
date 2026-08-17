# Playcatch — Sistema de Recomendação de Músicas

Prova de conceito de um sistema de recomendação musical **item-based** desenvolvido para o Desafio 1 — Nível 3 da formação de Especialista em Inteligência Artificial da Alura.

A solução cobre todo o fluxo: preparação dos dados, análise exploratória, expansão sintética controlada, recomendação por similaridade cosseno, avaliação *leave-one-out*, rastreamento de experimentos com MLflow/SQLite, interface Gradio, testes automatizados e geração de evidências auditáveis.

> **Importante:** os dados sintéticos demonstram o funcionamento técnico do pipeline. Os resultados não representam desempenho em produção nem comportamento real de usuários.

## Visão geral

- Preserva os 16 eventos originais do desafio.
- Acrescenta exatamente 10 usuários e 10 músicas sintéticas.
- Identifica a procedência dos dados pela coluna `data_origin`.
- Agrega eventos repetidos antes de construir a matriz usuário-item.
- Recomenda músicas por KNN item-based com distância cosseno.
- Exclui músicas já consumidas pelo usuário.
- Usa popularidade global como fallback para usuários desconhecidos ou sem candidatos.
- Avalia o modelo sem reutilizar o item ocultado no treinamento de cada fold.
- Registra parâmetros, métricas, modelos e artefatos no MLflow.
- Disponibiliza recomendações em uma interface Gradio local.

## Arquitetura

```text
dados originais + gerador sintético (seed=42)
                       |
                       v
             validação e auditoria
                       |
                       v
       agregação por usuário e música + EDA
                       |
                       v
             matriz usuário x item
                       |
                       v
       matriz item x usuário + normalização
                       |
                       v
 NearestNeighbors(metric="cosine", algorithm="brute")
             |                          |
             v                          v
 recomendação + fallback         leave-one-out
             |                          |
             +------------+-------------+
                          v
                   MLflow + SQLite
                          |
                          v
                     Gradio local
```

## Dados

| Medida | Base original | Base expandida |
|---|---:|---:|
| Eventos brutos | 16 | 79 |
| Usuários | 3 | 13 |
| Músicas | 8 | 18 |
| Pares únicos usuário-música | 14 | 77 |
| Reproduções | 60 | 465 |
| Dimensão da matriz | 3 × 8 | 13 × 18 |
| Densidade | 58,33% | 32,91% |
| Valores ausentes | 0 | 0 |
| Duplicatas integrais removidas | 0 | 0 |

A expansão usa `seed=42` e cria os usuários 4 a 13 e as músicas 109 a 118. Grupos de preferência e itens-ponte produzem coocorrências adequadas à validação funcional, sem alterar os registros originais.

Os arquivos gerados ficam em `data/user_data_original.csv` e `data/user_data_expanded.csv`.

## Funcionamento do recomendador

1. Os eventos de cada par usuário-música são somados.
2. A matriz usuário-item é transposta para item-usuário.
3. Os vetores são normalizados por L1 ou L2.
4. O KNN encontra músicas vizinhas usando similaridade cosseno.
5. Os candidatos recebem score simples ou ponderado por `log1p(play_count)`.
6. Auto-vizinhos e músicas já ouvidas são removidos.
7. Os resultados são ordenados por score decrescente e `song_id` crescente.
8. Quando não há histórico ou candidatos, aplica-se o fallback de popularidade.

## Resultados da avaliação

A avaliação *leave-one-out* oculta o item agregado mais consumido de cada usuário, reajusta o modelo em cada fold e calcula Hit Rate, Precision, MRR e cobertura de catálogo.

### Base original

Com 5 vizinhos, `k=5`, normalização L2 e score simples:

- Hit Rate@5: **66,67%**
- Precision@5: **13,33%**
- MRR@5: **0,15**
- Cobertura: **100%**
- Usuários avaliados: **3**

Como a amostra possui apenas três usuários, cada erro altera o Hit Rate em 33,33 pontos percentuais. Esses números são uma verificação funcional, não uma estimativa estatística de qualidade.

### Base expandida

| Configuração | Vizinhos | k | Norma | Score | Hit Rate | Precision | MRR | Cobertura |
|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 3 | 3 | L2 | simples | 38,46% | 12,82% | 0,3333 | 50,00% |
| 2 | 5 | 5 | L2 | simples | 46,15% | 9,23% | 0,3385 | 83,33% |
| 3 | 7 | 5 | L1 | `weighted_log1p` | 100,00% | 20,00% | 0,8769 | 83,33% |

A terceira configuração apresentou o melhor resultado no cenário sintético controlado. O Hit Rate perfeito decorre de padrões deliberadamente recuperáveis e não deve ser interpretado como prontidão produtiva.

## Pré-requisitos

- Python 3.12 ou compatível com o intervalo definido em `pyproject.toml`.
- [uv](https://docs.astral.sh/uv/) para instalar e executar o ambiente reproduzível.
- PowerShell, Prompt de Comando ou terminal equivalente.

As principais bibliotecas utilizadas são Pandas, Scikit-learn, MLflow, Gradio e Pytest. As versões resolvidas estão registradas em `uv.lock`.

## Instalação

No PowerShell, a partir da raiz do projeto:

```powershell
$env:UV_CACHE_DIR='E:\ProjAlura\ProjNivel3\.uv-cache'
uv sync --extra dev
```

O uso de `UV_CACHE_DIR` é opcional. Ele apenas mantém o cache dentro do diretório do projeto.

## Execução completa do pipeline

```powershell
uv run python scripts\run_pipeline.py
```

O comando recria os datasets, executa a análise, treina e avalia as configurações, registra os experimentos e atualiza as evidências em `artifacts/`.

Para validar o banco de tracking e o manifesto de artefatos:

```powershell
uv run python scripts\verify_repository.py
```

## Testes automatizados

```powershell
uv run pytest -q -p no:cacheprovider
```

Resultado registrado na implementação: **7 testes aprovados em 10,18 segundos**.

Os testes verificam agregação de pares repetidos, escopo e reprodutibilidade dos dados sintéticos, exclusão de músicas ouvidas, ordenação dos scores, fallback, validação de `k`, métricas e contrato da interface.

## Interface Gradio

```powershell
uv run python app.py
```

Abra no navegador o endereço local informado no terminal. Selecione um usuário, altere `k` entre 1 e 10 e confirme que a tabela contém `rank`, `song_id`, `score` e `origin`.

A interface é iniciada sem link público. Usuários desconhecidos são cobertos pelos testes automatizados e pelo fallback de popularidade; a interface visual oferece os usuários existentes no dropdown.

## MLflow

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
```

Acesse [http://127.0.0.1:5000](http://127.0.0.1:5000), abra o experimento `playcatch-recsys` e compare as três execuções. O backend SQLite é local e adequado à prova de conceito, mas não a uma implantação concorrente de produção.

## Estrutura do projeto

```text
ProjNivel3/
├── Analise/                    # relatórios executivo-técnicos em MD e DOCX
├── artifacts/                  # métricas, matrizes e evidências das execuções
├── data/                       # datasets original e expandido
├── scripts/
│   ├── run_pipeline.py         # execução completa dos experimentos
│   └── verify_repository.py    # validação do tracking e dos artefatos
├── src/playcatch/
│   ├── data.py                 # dados, validação, agregação e matriz
│   ├── evaluate.py             # avaliação leave-one-out e métricas
│   ├── pipeline.py             # EDA, execução e persistência
│   ├── recommender.py          # KNN, recomendação e fallback
│   └── tracking.py             # integração com MLflow/SQLite
├── tests/                      # testes automatizados
├── app.py                      # interface Gradio
├── pyproject.toml              # configuração e dependências
└── uv.lock                     # ambiente reproduzível
```

## Critérios de aceite atendidos

| Grupo | Entrega | Status |
|---|---|---|
| Dados | carga, tipagem, auditoria, agregação e matriz sem NaN | Feito |
| Expansão | 10 usuários e 10 músicas com origem identificável | Feito |
| Recomendação | KNN item-based, filtro de vistos, desempate e fallback | Feito |
| Avaliação | leave-one-out, Hit Rate, Precision, MRR e cobertura | Feito |
| Tracking | SQLite, três runs MLflow, parâmetros, métricas e artefatos | Feito |
| Interface | aplicação Gradio local e callback testado | Feito |
| Qualidade | sete testes automatizados aprovados | Feito |
| Evidências | datasets, hashes, matrizes, resumos e recomendações | Feito |
| Documentação | README e relatórios em Markdown e Word | Feito |

A matriz completa de critérios e suas evidências está no relatório de implementação.

## Evidências e relatórios

- [`artifacts/execution_summary.json`](artifacts/execution_summary.json): consolidação das execuções.
- [`artifacts/original/`](artifacts/original/): evidências obtidas apenas com a base original.
- [`artifacts/mlflow/`](artifacts/mlflow/): evidências de cada configuração rastreada.
- [`Analise/RELATORIO_EXECUTIVO_IMPLEMENTACAO_DESAFIO1_NIVEL3_RECOMENDACAO_MUSICAS.md`](Analise/RELATORIO_EXECUTIVO_IMPLEMENTACAO_DESAFIO1_NIVEL3_RECOMENDACAO_MUSICAS.md): relatório completo de implementação.
- [`Analise/RELATORIO_EXECUTIVO_IMPLEMENTACAO_DESAFIO1_NIVEL3_RECOMENDACAO_MUSICAS.docx`](Analise/RELATORIO_EXECUTIVO_IMPLEMENTACAO_DESAFIO1_NIVEL3_RECOMENDACAO_MUSICAS.docx): versão executiva em Word.
- [`Analise/RELATORIO_EXECUTIVO_ESTUDO_DESAFIO1_NIVEL3_RECOMENDACAO_MUSICAS.md`](Analise/RELATORIO_EXECUTIVO_ESTUDO_DESAFIO1_NIVEL3_RECOMENDACAO_MUSICAS.md): estudo e levantamento prévio.

## Limitações e próximos passos

- Substituir os dados sintéticos por histórico real anonimizado e autorizado.
- Adicionar metadados de catálogo, como título, artista, gênero e disponibilidade.
- Adotar divisão temporal e avaliação online/A-B antes de decisões de produto.
- Empregar backend e armazenamento adequados à concorrência real.
- Empacotar pré-processamento, mapeamentos e modelo como uma única unidade de serviço.
- Incluir autenticação, observabilidade e controles de LGPD antes de exposição externa.

## Repositório

[github.com/fredjml/aluraCarreiraEspecialistaIANivel3](https://github.com/fredjml/aluraCarreiraEspecialistaIANivel3)
