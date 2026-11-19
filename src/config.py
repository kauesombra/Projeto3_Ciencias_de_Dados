"""Configurações centrais do Projeto 3 (RESISC45, 10 classes).

Tudo que precisa ser idêntico entre experimentos (classes, ordem, semente,
caminhos) fica aqui para evitar divergências entre notebooks e scripts.
"""
from pathlib import Path

# --- Caminhos (relativos à raiz do repositório) ---------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "dados" / "raw" / "NWPU-RESISC45"
DIR_SPLITS = RAIZ / "dados" / "splits"
DIR_MODELOS = RAIZ / "modelos"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"
DIR_HISTORICOS = DIR_RESULTADOS / "historicos"

ARQ_SPLIT = DIR_SPLITS / "particoes.csv"
ARQ_STATS = DIR_SPLITS / "estatisticas_treino.json"
ARQ_EXPERIMENTOS = DIR_RESULTADOS / "experimentos.csv"

# --- Classes: nomes originais das pastas, ORDEM FIXA em todo o projeto ----
CLASSES = [
    "airport", "beach", "circular_farmland", "dense_residential", "desert",
    "forest", "harbor", "mountain", "river", "storage_tank",
]
NOMES_PT = {
    "airport": "Aeroporto", "beach": "Praia",
    "circular_farmland": "Áreas agrícolas circulares",
    "dense_residential": "Área residencial densa", "desert": "Deserto",
    "forest": "Floresta", "harbor": "Porto", "mountain": "Montanha",
    "river": "Rio", "storage_tank": "Tanques de armazenamento",
}
CLASSE_PARA_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

# --- Dados ------------------------------------------------------------------
SEMENTE = 42
FRAC_TREINO, FRAC_VAL, FRAC_TESTE = 0.70, 0.15, 0.15
TAMANHO_ORIGINAL = 256

# --- Protocolo de treino (igual para todos os 6 treinos finais) ------------
TREINO = dict(
    otimizador="Adam",
    taxa_aprendizagem=1e-3,
    lote=32,
    max_epocas=40,
    paciencia=8,            # parada antecipada monitorando F1 macro da validação
    metrica_selecao="val_f1_macro",
)

# Os seis treinos obrigatórios: nome -> (arquitetura, resolução)
EXPERIMENTOS = {
    "MLP-A-128": ("mlp_a", 128),
    "MLP-B-128": ("mlp_b", 128),
    "CNN-A-128": ("cnn_a", 128),
    "CNN-B-128": ("cnn_b", 128),
    "CNN-A-256": ("cnn_a", 256),
    "CNN-B-256": ("cnn_b", 256),
}
