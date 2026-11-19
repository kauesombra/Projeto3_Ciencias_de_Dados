"""Pipeline tf.data: lê os ORIGINAIS 256x256, converte para RGB, redimensiona a
partir do original, padroniza com estatísticas do TREINO. O aumento de dados
ocorre dentro do modelo (camadas de augmentation) e só fica ativo no treino."""
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from . import config as C


def carregar_particao(nome: str) -> pd.DataFrame:
    """nome: 'treino' | 'validacao' | 'teste'. Ordem estável (por arquivo)."""
    df = pd.read_csv(C.ARQ_SPLIT)
    df = df[df["particao"] == nome].sort_values("arquivo").reset_index(drop=True)
    df["y"] = df["classe"].map(C.CLASSE_PARA_IDX)
    return df


def _estatisticas():
    s = json.loads(C.ARQ_STATS.read_text())
    return (np.array(s["media"], dtype="float32"), np.array(s["desvio"], dtype="float32"))


def montar_dataset(nome: str, resolucao: int, lote: int, embaralhar: bool = False,
                   semente: int = C.SEMENTE, cache: bool = False) -> tf.data.Dataset:
    df = carregar_particao(nome)
    media, desvio = _estatisticas()
    caminhos = [str(C.RAIZ / p) for p in df["arquivo"]]
    ds = tf.data.Dataset.from_tensor_slices((caminhos, df["y"].to_numpy("int32")))
    if embaralhar:
        ds = ds.shuffle(len(df), seed=semente, reshuffle_each_iteration=True)

    def ler(caminho, y):
        img = tf.io.decode_image(tf.io.read_file(caminho), channels=3, expand_animations=False)  # RGB
        img.set_shape([C.TAMANHO_ORIGINAL, C.TAMANHO_ORIGINAL, 3])
        img = tf.cast(img, tf.float32) / 255.0                          # [0,1]
        if resolucao != C.TAMANHO_ORIGINAL:                              # resize A PARTIR DO ORIGINAL
            img = tf.image.resize(img, (resolucao, resolucao), method="bilinear", antialias=True)
        img = (img - media) / desvio                                     # estatísticas do treino
        return img, y

    ds = ds.map(ler, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()   # cuidado com memória em 256x256 (≈ 4,9k imgs * 786 KB ≈ 3,9 GB no treino)
    return ds.batch(lote).prefetch(tf.data.AUTOTUNE)
