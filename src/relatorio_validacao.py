"""Resultados de VALIDAÇÃO de um modelo treinado (usar no acompanhamento; o teste
NÃO deve ser avaliado nesta etapa).

Uso: python -m src.relatorio_validacao --nome CNN-A-128
Gera em resultados/figuras e resultados/:
    curvas_<NOME>.png, matriz_val_<NOME>.png, metricas_val_<NOME>.csv
"""
import argparse

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from . import config as C
from .datasets import carregar_particao, montar_dataset


def curvas(nome, sufixo=""):
    h = pd.read_csv(C.DIR_HISTORICOS / f"{nome}{sufixo}.csv")
    melhor = int(h["val_f1_macro"].idxmax()) + 1
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, (tr, va, tit) in zip(ax, [("loss", "val_loss", "Perda"), ("accuracy", "val_accuracy", "Acurácia"),
                                     (None, "val_f1_macro", "F1 macro (validação)")]):
        if tr: a.plot(h["epoca"], h[tr], label="treino")
        a.plot(h["epoca"], h[va], label="validação")
        a.axvline(melhor, ls="--", c="gray", lw=1, label=f"checkpoint (época {melhor})")
        a.set_title(f"{nome} - {tit}"); a.set_xlabel("Época"); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(C.DIR_FIGURAS / f"curvas_{nome}{sufixo}.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nome", required=True, choices=list(C.EXPERIMENTOS))
    a = ap.parse_args()
    _, res = C.EXPERIMENTOS[a.nome]
    modelo = keras.models.load_model(C.DIR_MODELOS / f"{a.nome}.keras")
    df = carregar_particao("validacao")
    prob = modelo.predict(montar_dataset("validacao", res, 64), verbose=0)
    y, p = df["y"].to_numpy(), prob.argmax(1)

    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(C.N_CLASSES), zero_division=0)
    tab = pd.DataFrame(dict(classe=C.CLASSES, precisao=pr, recall=rc, f1=f1, suporte=sup))
    tab.loc[len(tab)] = ["acuracia", np.nan, np.nan, accuracy_score(y, p), len(y)]
    tab.loc[len(tab)] = ["f1_macro", np.nan, np.nan, f1_score(y, p, average="macro", zero_division=0), len(y)]
    tab.to_csv(C.DIR_RESULTADOS / f"metricas_val_{a.nome}.csv", index=False); print(tab.round(3))

    cm = confusion_matrix(y, p, labels=range(C.N_CLASSES))   # linhas = real, colunas = prevista
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for a_, m, t, fmt in [(ax[0], cm, "Contagens", "d"), (ax[1], cmn, "Normalizada por classe real", ".2f")]:
        im = a_.imshow(m, cmap="Blues"); a_.set_title(f"{a.nome} - validação: {t}")
        a_.set_xticks(range(C.N_CLASSES)); a_.set_yticks(range(C.N_CLASSES))
        a_.set_xticklabels(C.CLASSES, rotation=60, ha="right", fontsize=7); a_.set_yticklabels(C.CLASSES, fontsize=7)
        a_.set_xlabel("Prevista"); a_.set_ylabel("Real")
        for i in range(C.N_CLASSES):
            for j in range(C.N_CLASSES):
                a_.text(j, i, format(m[i, j], fmt), ha="center", va="center", fontsize=6,
                        color="white" if m[i, j] > m.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(C.DIR_FIGURAS / f"matriz_val_{a.nome}.png", dpi=150); plt.close()
    curvas(a.nome)
    print("Figuras e métricas salvas em resultados/")


if __name__ == "__main__":
    main()
