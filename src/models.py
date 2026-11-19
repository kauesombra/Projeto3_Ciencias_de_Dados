"""Arquiteturas. A e B de cada família diferem ESTRUTURALMENTE.

MLP A: 1 camada oculta larga (256)         -> hipótese: baseline raso
MLP B: 3 camadas ocultas (512-256-128)     -> hipótese: mais profundidade/largura ajudam?
CNN A: 3 blocos (1 conv por bloco)         -> hipótese: baseline convolucional leve
CNN B: 4 blocos (2 conv por bloco, + largo) -> hipótese: profundidade/largura compensam o custo?

As CNNs usam GlobalAveragePooling2D antes do classificador, então o número de
parâmetros é IDÊNTICO em 128x128 e 256x256 (apenas o custo computacional muda).
Pré-processamento (RGB, padronização) é feito no tf.data; a augmentation fica no
modelo e só atua em training=True.
"""
import keras
from keras import layers

from . import config as C


def _augmentation():
    # Imagens aéreas não têm "cima": flips e rotações são transformações plausíveis.
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25, fill_mode="reflect"),   # até ±90°
    ], name="augmentation")


def _bloco_conv(x, filtros, n_conv):
    for _ in range(n_conv):
        x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    return layers.MaxPooling2D(2)(x)


def construir_modelo(arquitetura: str, resolucao: int, aumento: bool = True) -> keras.Model:
    entrada = keras.Input((resolucao, resolucao, 3))
    x = _augmentation()(entrada) if aumento else entrada

    if arquitetura == "mlp_a":
        x = layers.Flatten()(x)                       # RGB como vetor
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    elif arquitetura == "mlp_b":
        x = layers.Flatten()(x)
        for u in (512, 256, 128):
            x = layers.Dense(u, activation="relu")(x)
            x = layers.Dropout(0.3)(x)
    elif arquitetura == "cnn_a":
        for f in (32, 64, 128):
            x = _bloco_conv(x, f, n_conv=1)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
    elif arquitetura == "cnn_b":
        for f in (32, 64, 128, 256):
            x = _bloco_conv(x, f, n_conv=2)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
    else:
        raise ValueError(f"Arquitetura desconhecida: {arquitetura}")

    saida = layers.Dense(C.N_CLASSES, activation="softmax")(x)
    return keras.Model(entrada, saida, name=f"{arquitetura}_{resolucao}")
