"""Inventário, verificação de duplicatas, partição e estatísticas de treino.

Uso (da raiz do repositório):
    python -m src.data_prep

Gera:
    dados/splits/inventario.csv          (um registro por arquivo + checagens)
    dados/splits/particoes.csv           (arquivo, classe, particao)  <- reutilizar sempre
    dados/splits/estatisticas_treino.json (média/desvio por canal, SOMENTE treino)
    resultados/inventario_resumo.json

Não altera nenhum arquivo original.
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

from . import config as C


def _dhash(img: Image.Image, n: int = 8) -> int:
    """Hash perceptual de diferença (detecta cópias/versões quase idênticas)."""
    g = img.convert("L").resize((n + 1, n), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def inventariar() -> pd.DataFrame:
    linhas = []
    for classe in C.CLASSES:
        pasta = C.DIR_RAW / classe
        if not pasta.is_dir():
            sys.exit(f"Pasta não encontrada: {pasta}")
        for caminho in sorted(pasta.iterdir()):
            if caminho.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rel = caminho.relative_to(C.RAIZ).as_posix()
            reg = dict(arquivo=rel, classe=classe, leitura_ok=True, largura=None,
                       altura=None, modo=None, md5=None, dhash=None, erro="")
            try:
                reg["md5"] = hashlib.md5(caminho.read_bytes()).hexdigest()
                with Image.open(caminho) as im:
                    im.load()
                    reg["largura"], reg["altura"] = im.size
                    reg["modo"] = im.mode
                    reg["dhash"] = _dhash(im)
            except Exception as e:  # arquivo corrompido
                reg["leitura_ok"], reg["erro"] = False, str(e)
            linhas.append(reg)
    return pd.DataFrame(linhas)


def agrupar_duplicatas(inv: pd.DataFrame, limiar_hamming: int = 2) -> pd.Series:
    """Atribui um grupo a cada imagem. Duplicatas exatas (md5) e quase
    duplicatas (dHash com distância de Hamming <= limiar) ficam no MESMO grupo."""
    n = len(inv)
    pai = list(range(n))

    def achar(i):
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = pai[i]
        return i

    def unir(i, j):
        pi, pj = achar(i), achar(j)
        if pi != pj:
            pai[max(pi, pj)] = min(pi, pj)

    # 1) duplicatas exatas
    for _, idx in inv.groupby("md5").indices.items():
        for k in idx[1:]:
            unir(idx[0], k)
    # 2) quase duplicatas (comparação par a par via XOR vetorizado; 7.000 imagens é viável)
    h = inv["dhash"].fillna(0).astype("uint64").to_numpy()
    for i in range(n):
        xor = h[i] ^ h[i + 1:]
        dist = np.array([bin(int(v)).count("1") for v in xor]) if len(xor) else np.array([])
        for j in np.nonzero(dist <= limiar_hamming)[0]:
            unir(i, i + 1 + int(j))
    return pd.Series([achar(i) for i in range(n)], index=inv.index, name="grupo")


def particionar(inv: pd.DataFrame) -> pd.DataFrame:
    """70/15/15 estratificado por classe, semente 42, caminhos ordenados antes do
    sorteio, grupos de duplicatas inteiros em uma única partição."""
    rng = np.random.default_rng(C.SEMENTE)
    inv = inv.sort_values("arquivo").reset_index(drop=True)
    partes = []
    for classe in C.CLASSES:
        d = inv[inv["classe"] == classe]
        grupos = sorted(d["grupo"].unique())          # ordem determinística
        rng.shuffle(grupos)                            # sorteio com semente 42
        tam = d.groupby("grupo").size().to_dict()
        n = len(d)
        alvo = {"teste": round(n * C.FRAC_TESTE), "validacao": round(n * C.FRAC_VAL)}
        alvo["treino"] = n - alvo["teste"] - alvo["validacao"]
        cont = {"treino": 0, "validacao": 0, "teste": 0}
        atrib = {}
        for g in grupos:   # preenche teste, depois validação, resto no treino
            for p in ("teste", "validacao", "treino"):
                if cont[p] + tam[g] <= alvo[p] or p == "treino":
                    atrib[g] = p
                    cont[p] += tam[g]
                    break
        d = d.assign(particao=d["grupo"].map(atrib))
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


def estatisticas_treino(part: pd.DataFrame) -> dict:
    """Média e desvio por canal (escala [0,1]) calculados SOMENTE no treino,
    sobre as imagens originais 256x256."""
    tr = part[part["particao"] == "treino"]["arquivo"]
    soma = np.zeros(3); soma2 = np.zeros(3); n_pix = 0
    for rel in tr:
        a = np.asarray(Image.open(C.RAIZ / rel).convert("RGB"), dtype=np.float64) / 255.0
        soma += a.sum((0, 1)); soma2 += (a ** 2).sum((0, 1)); n_pix += a.shape[0] * a.shape[1]
    media = soma / n_pix
    desvio = np.sqrt(soma2 / n_pix - media ** 2)
    return dict(media=media.tolist(), desvio=desvio.tolist(), n_imagens_treino=int(len(tr)),
                escala="pixels/255 antes de padronizar", ordem_canais="RGB")


def main():
    C.DIR_SPLITS.mkdir(parents=True, exist_ok=True)
    print("1/4 Inventariando (leitura, dimensões, canais, hashes)...")
    inv = inventariar()
    inv["grupo"] = agrupar_duplicatas(inv)
    inv.to_csv(C.DIR_SPLITS / "inventario.csv", index=False)

    n_grupo = inv.groupby("grupo").size()
    resumo = dict(
        total_arquivos=int(len(inv)),
        por_classe=inv.groupby("classe").size().reindex(C.CLASSES).to_dict(),
        falhas_leitura=int((~inv["leitura_ok"]).sum()),
        dimensoes=inv.groupby(["largura", "altura"]).size().rename("n").reset_index().to_dict("records"),
        modos=inv["modo"].value_counts().to_dict(),
        duplicatas_exatas_md5=int(inv["md5"].duplicated().sum()),
        grupos_com_mais_de_uma_imagem=int((n_grupo > 1).sum()),
        imagens_em_grupos_duplicados=int(n_grupo[n_grupo > 1].sum()),
    )
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    if resumo["falhas_leitura"]:
        print("ATENÇÃO: há arquivos ilegíveis; registre e trate (exclusão justificada) antes de seguir.")

    print("2/4 Particionando (70/15/15, estratificado, semente 42)...")
    part = particionar(inv[inv["leitura_ok"]])
    part[["arquivo", "classe", "particao"]].to_csv(C.ARQ_SPLIT, index=False)
    tab = part.pivot_table(index="classe", columns="particao", values="arquivo",
                           aggfunc="count").reindex(C.CLASSES)[["treino", "validacao", "teste"]]
    print(tab)
    resumo["contagens_particao"] = {k: tab[k].to_dict() for k in tab.columns}
    resumo["totais_particao"] = tab.sum().astype(int).to_dict()

    # verificação de vazamento: nenhum grupo em duas partições
    vaz = part.groupby("grupo")["particao"].nunique().gt(1).sum()
    resumo["grupos_em_mais_de_uma_particao"] = int(vaz)
    assert vaz == 0, "Vazamento: grupo duplicado em mais de uma partição"

    print("3/4 Estatísticas de normalização (somente treino)...")
    stats = estatisticas_treino(part)
    C.ARQ_STATS.write_text(json.dumps(stats, indent=2))
    print(stats)

    print("4/4 Salvando resumo...")
    C.DIR_RESULTADOS.mkdir(exist_ok=True)
    (C.DIR_RESULTADOS / "inventario_resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
