"""Figuras da análise exploratória. USA SOMENTE IMAGENS DE TREINO (a validação e o
teste só entram em contagens/integridade, conforme o enunciado).

Uso: python -m src.exploracao
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . import config as C

C.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
FONTE = "Fonte: NWPU-RESISC45 (Cheng et al., 2017); elaboração própria."


def _img(rel, tam=None):
    im = Image.open(C.RAIZ / rel).convert("RGB")
    return im.resize((tam, tam), Image.BILINEAR) if tam else im


def fig_contagens(part):
    tab = part.pivot_table(index="classe", columns="particao", values="arquivo",
                           aggfunc="count").reindex(C.CLASSES)[["treino", "validacao", "teste"]]
    ax = tab.plot(kind="bar", stacked=True, figsize=(9, 4), color=["#2f6b8a", "#7fb3c8", "#c9dfe8"])
    ax.set_ylabel("Imagens"); ax.set_xlabel(""); ax.set_title("Imagens por classe e partição")
    plt.xticks(rotation=40, ha="right"); plt.figtext(0.01, 0.005, FONTE, fontsize=7)
    plt.tight_layout(); plt.savefig(C.DIR_FIGURAS / "eda_contagens.png", dpi=150); plt.close()


def fig_exemplos(treino, n=5):
    fig, axs = plt.subplots(C.N_CLASSES, n, figsize=(n * 1.6, C.N_CLASSES * 1.6))
    rng = np.random.default_rng(C.SEMENTE)
    for i, c in enumerate(C.CLASSES):
        arqs = treino[treino["classe"] == c]["arquivo"].tolist()
        for j, rel in enumerate(rng.choice(arqs, n, replace=False)):
            axs[i, j].imshow(_img(rel)); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            if j == 0:
                axs[i, j].set_ylabel(C.NOMES_PT[c], rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Exemplos aleatórios do treino (semente 42)", y=0.995)
    plt.figtext(0.01, 0.002, FONTE, fontsize=6)
    plt.tight_layout(); plt.savefig(C.DIR_FIGURAS / "eda_exemplos_classes.png", dpi=130); plt.close()


def fig_resolucoes(treino):
    """Mesma imagem em 256 (original), 128 e 128 exibida no mesmo tamanho físico."""
    rng = np.random.default_rng(C.SEMENTE)
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i, c in enumerate(["airport", "circular_farmland", "storage_tank"]):
        rel = rng.choice(treino[treino["classe"] == c]["arquivo"].tolist())
        orig = _img(rel); pequena = _img(rel, 128)
        axs[i, 0].imshow(orig); axs[i, 0].set_title("256 × 256 (original)" if i == 0 else "")
        axs[i, 1].imshow(pequena); axs[i, 1].set_title("128 × 128" if i == 0 else "")
        # recorte ampliado para evidenciar a perda de detalhe fina
        axs[i, 2].imshow(np.asarray(pequena)[:64, :64], interpolation="nearest")
        axs[i, 2].set_title("128: recorte 64×64 (zoom)" if i == 0 else "")
        axs[i, 0].set_ylabel(C.NOMES_PT[c], fontsize=9)
        for a in axs[i]:
            a.set_xticks([]); a.set_yticks([])
    plt.figtext(0.01, 0.005, FONTE, fontsize=7)
    plt.tight_layout(); plt.savefig(C.DIR_FIGURAS / "eda_resolucoes.png", dpi=140); plt.close()


def fig_duplicatas(inv, part):
    d = inv.merge(part[["arquivo", "particao"]], on="arquivo")
    g = d.groupby("grupo").filter(lambda x: len(x) > 1).sort_values(["grupo", "arquivo"])
    grupos = list(g.groupby("grupo"))
    if not grupos:
        return
    fig, axs = plt.subplots(len(grupos), 2, figsize=(4, 2 * len(grupos)))
    axs = np.atleast_2d(axs)
    for i, (gid, x) in enumerate(grupos):
        for j, rel in enumerate(x["arquivo"].tolist()[:2]):
            axs[i, j].imshow(_img(rel)); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            axs[i, j].set_title(rel.split("/")[-1], fontsize=7)
    fig.suptitle("Grupos de imagens duplicadas/quase idênticas", fontsize=9)
    plt.tight_layout(); plt.savefig(C.DIR_FIGURAS / "eda_duplicatas.png", dpi=130); plt.close()


def fig_cor_por_classe(treino):
    """Média de cor (R,G,B) por classe no treino: ajuda a ver quais classes são
    separáveis só por cor (relevante para o MLP)."""
    medias = []
    for c in C.CLASSES:
        arqs = treino[treino["classe"] == c]["arquivo"].tolist()[:150]
        medias.append(np.mean([np.asarray(_img(r, 64), dtype=float).mean((0, 1)) for r in arqs], 0) / 255)
    medias = np.array(medias)
    fig, ax = plt.subplots(figsize=(8, 3.5)); x = np.arange(C.N_CLASSES)
    for k, (nome, cor) in enumerate(zip("RGB", ["#c0392b", "#27ae60", "#2980b9"])):
        ax.bar(x + (k - 1) * 0.27, medias[:, k], 0.27, label=nome, color=cor)
    ax.set_xticks(x); ax.set_xticklabels(C.CLASSES, rotation=40, ha="right")
    ax.set_ylabel("Intensidade média [0–1]"); ax.legend(title="Canal")
    ax.set_title("Cor média por classe (150 imagens de treino por classe)")
    plt.figtext(0.01, 0.005, FONTE, fontsize=7)
    plt.tight_layout(); plt.savefig(C.DIR_FIGURAS / "eda_cor_por_classe.png", dpi=150); plt.close()


def mosaico_para_ambiguidades(treino, n=12):
    """Mosaico de candidatas a ambíguas: imagens cuja cor média está mais perto
    do centro de OUTRA classe do que da própria. É só um ponto de partida: a
    equipe deve olhar e escolher/justificar as ambiguidades manualmente."""
    cent, feats, arqs = {}, [], []
    for c in C.CLASSES:
        for r in treino[treino["classe"] == c]["arquivo"].tolist()[:200]:
            feats.append(np.asarray(_img(r, 32), dtype=float).mean((0, 1))); arqs.append((r, c))
    F = np.array(feats); cls = np.array([c for _, c in arqs])
    for c in C.CLASSES:
        cent[c] = F[cls == c].mean(0)
    cand = []
    for f, (r, c) in zip(F, arqs):
        d = {k: np.linalg.norm(f - v) for k, v in cent.items()}
        outra = min((k for k in d if k != c), key=d.get)
        cand.append((d[c] - d[outra], r, c, outra))   # >0: mais perto de outra classe
    cand.sort(reverse=True); cand = cand[:n]
    fig, axs = plt.subplots(2, n // 2, figsize=(n // 2 * 2, 4.4))
    for a, (_, r, c, o) in zip(axs.ravel(), cand):
        a.imshow(_img(r)); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{c}\n(parece {o})", fontsize=7)
    fig.suptitle("Candidatas a ambiguidade visual (cor média; revisar manualmente)", fontsize=9)
    plt.tight_layout(); plt.savefig(C.DIR_FIGURAS / "eda_ambiguidades_candidatas.png", dpi=130); plt.close()


def main():
    part = pd.read_csv(C.ARQ_SPLIT); inv = pd.read_csv(C.DIR_SPLITS / "inventario.csv")
    treino = part[part["particao"] == "treino"]
    fig_contagens(part)            # só contagens: permitido para todas as partições
    fig_exemplos(treino); fig_resolucoes(treino)
    fig_duplicatas(inv, part); fig_cor_por_classe(treino); mosaico_para_ambiguidades(treino)
    print("Figuras salvas em", C.DIR_FIGURAS)


if __name__ == "__main__":
    main()
