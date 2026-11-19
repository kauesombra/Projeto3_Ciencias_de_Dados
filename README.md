# Projeto 3 – Classificação de Cenas Aéreas com Redes Neurais (NWPU-RESISC45)

Disciplina de Ciência de Dados – Unidade III (Redes neurais e Deep Learning).
**Equipe:** _(nomes dos integrantes)_ · **Repositório:** _(link do GitHub)_

Compara duas MLPs, duas CNNs (128×128) e o efeito da resolução (CNNs em 256×256) em
10 classes do NWPU-RESISC45, com pesos novos (sem pré-treinamento).

## Classes (ordem fixa em todo o projeto)
`airport, beach, circular_farmland, dense_residential, desert, forest, harbor, mountain, river, storage_tank`

## Versões
Python 3.12 · TensorFlow/Keras (ver `requirements.txt` e `resultados/historicos/*_config.json`, que
registram as versões exatas usadas em cada treino).

## Instalação
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
GPU: TensorFlow ≥ 2.11 **não** usa GPU no Windows nativo. Use WSL2 (`pip install "tensorflow[and-cuda]"`)
ou o Google Colab. Em CPU os treinos funcionam, mas as CNNs em 256×256 são lentas.

## Dados
Baixe o NWPU-RESISC45 (links na Seção 7 do enunciado) e coloque **somente as 10 pastas** em
`dados/raw/NWPU-RESISC45/<classe>/`. Se você extraiu o dataset completo:
```bash
python -m src.filtrar_classes "/caminho/para/NWPU-RESISC45"
```
Os originais não são alterados nem versionados no Git.

## Ordem de execução
| # | Comando / notebook | O que faz |
|---|---|---|
| 1 | `python -m src.data_prep` | Inventário, duplicatas, partição 70/15/15 (semente 42), estatísticas só do treino |
| 2 | `python -m src.exploracao` | Figuras da análise exploratória (só treino) |
| 3 | `python -m src.train --nome MLP-A-128` (e os outros 5) | Treina e salva o melhor checkpoint por F1 macro de validação |
| 4 | `python -m src.relatorio_validacao --nome <NOME>` | Curvas, matriz de confusão e métricas **de validação** |

Nomes dos seis treinos: `MLP-A-128, MLP-B-128, CNN-A-128, CNN-B-128, CNN-A-256, CNN-B-256`.
Os notebooks em `notebooks/` executam as mesmas etapas com comentários.

> Teste de fumaça (não é treino final, não grava em `experimentos.csv`):
> `python -m src.train --nome CNN-A-256 --max-epocas 2 --fumaca`

## Protocolo (resumo)
- Partição estratificada 70/15/15, semente 42, caminhos ordenados antes do sorteio; imagens duplicadas
  ou quase idênticas ficam sempre na mesma partição (`dados/splits/particoes.csv`).
- Entrada RGB redimensionada **a partir do original 256×256**; pixels/255 e padronização por canal
  com média/desvio calculados **somente no treino** (`dados/splits/estatisticas_treino.json`).
- Aumento de dados (flips e rotação) apenas no treino.
- Mesmos hiperparâmetros nos seis treinos (ver `src/config.py`); seleção de configuração e checkpoint
  pelo **F1 macro da validação**. O teste só é usado na avaliação final dos seis modelos.

## Estrutura
```
src/        config, data_prep, exploracao, datasets, models, train, relatorio_validacao
notebooks/  exploração e preparação; treino e validação
dados/      raw/ (originais, fora do Git) · splits/ (particoes.csv, inventario.csv, estatísticas)
modelos/    checkpoints (.keras)  ·  resultados/  experimentos.csv, históricos, figuras
docs/       acompanhamento.pdf, artigo.pdf, pitch.pdf
```
