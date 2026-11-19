"""Treina UM dos seis modelos obrigatórios (pesos novos, sem pré-treinamento).

Uso:
    python -m src.train --nome CNN-A-128
    python -m src.train --nome MLP-A-128 --max-epocas 3 --fumaca   # teste rápido (não é treino final)

Salva:
    modelos/<NOME>.keras                         melhor checkpoint (maior F1 macro de validação)
    resultados/historicos/<NOME>.csv             histórico por época (+ tempo)
    resultados/historicos/<NOME>_config.json     configuração e ambiente
    resultados/experimentos.csv                  uma linha por treino final
"""
import argparse
import json
import platform
import time

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

from . import config as C
from .datasets import carregar_particao, montar_dataset
from .models import construir_modelo


class TempoPorEpoca(keras.callbacks.Callback):
    """Mede o tempo de treino+validação de cada época (exclui o cálculo extra do F1)."""
    def on_epoch_begin(self, epoch, logs=None): self._t = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        logs["tempo_epoca_s"] = time.perf_counter() - self._t


class F1MacroCheckpoint(keras.callbacks.Callback):
    """Calcula o F1 macro da validação a cada época, salva o melhor checkpoint e
    aplica parada antecipada com base nele (métrica principal de seleção)."""
    def __init__(self, ds_val, y_val, caminho, paciencia):
        super().__init__()
        self.ds_val, self.y_val, self.caminho, self.paciencia = ds_val, y_val, caminho, paciencia
        self.melhor, self.melhor_epoca, self.espera = -1.0, -1, 0

    def on_epoch_end(self, epoch, logs=None):
        prob = self.model.predict(self.ds_val, verbose=0)
        f1 = f1_score(self.y_val, prob.argmax(1), average="macro", labels=range(C.N_CLASSES), zero_division=0)
        logs["val_f1_macro"] = f1
        if f1 > self.melhor:
            self.melhor, self.melhor_epoca, self.espera = f1, epoch + 1, 0
            self.model.save(self.caminho)
        else:
            self.espera += 1
            if self.espera >= self.paciencia:
                self.model.stop_training = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True, choices=list(C.EXPERIMENTOS))
    ap.add_argument("--max-epocas", type=int, default=C.TREINO["max_epocas"])
    ap.add_argument("--lote", type=int, default=C.TREINO["lote"])
    ap.add_argument("--cache", action="store_true", help="cache em RAM (rápido, mas usa muita memória em 256)")
    ap.add_argument("--fumaca", action="store_true", help="teste de fumaça: 2 lotes/época, NÃO grava em experimentos.csv")
    a = ap.parse_args()

    arq, res = C.EXPERIMENTOS[a.nome]
    keras.utils.set_random_seed(C.SEMENTE)

    C.DIR_MODELOS.mkdir(exist_ok=True); C.DIR_HISTORICOS.mkdir(parents=True, exist_ok=True)
    ds_tr = montar_dataset("treino", res, a.lote, embaralhar=True, cache=a.cache)
    ds_va = montar_dataset("validacao", res, a.lote, cache=a.cache)
    y_val = carregar_particao("validacao")["y"].to_numpy()
    if a.fumaca:
        ds_tr, ds_va, y_val = ds_tr.take(2), ds_va.take(2), y_val[: 2 * a.lote]

    modelo = construir_modelo(arq, res)
    modelo.compile(optimizer=keras.optimizers.Adam(C.TREINO["taxa_aprendizagem"]),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.summary()

    destino = C.DIR_MODELOS / (a.nome + ("_fumaca" if a.fumaca else "") + ".keras")
    ckpt = F1MacroCheckpoint(ds_va, y_val, destino, C.TREINO["paciencia"])
    t0 = time.perf_counter()
    hist = modelo.fit(ds_tr, validation_data=ds_va, epochs=a.max_epocas,
                      callbacks=[TempoPorEpoca(), ckpt], verbose=2, shuffle=False)  # embaralhamento já feito no tf.data
    tempo_total = time.perf_counter() - t0

    h = pd.DataFrame(hist.history); h.insert(0, "epoca", range(1, len(h) + 1))
    sufixo = "_fumaca" if a.fumaca else ""
    h.to_csv(C.DIR_HISTORICOS / f"{a.nome}{sufixo}.csv", index=False)

    gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    cfg = dict(nome=a.nome, arquitetura=arq, resolucao=res, parametros=int(modelo.count_params()),
               **{**C.TREINO, "lote": a.lote, "max_epocas": a.max_epocas}, semente=C.SEMENTE,
               perda="sparse_categorical_crossentropy",
               augmentation="RandomFlip(h+v), RandomRotation(0.25) - só treino",
               normalizacao="pixels/255, depois (x-media)/desvio com estatísticas do treino",
               hardware=dict(gpu=gpus or "nenhuma", cpu=platform.processor() or platform.machine(),
                             so=platform.platform()),
               versoes=dict(python=platform.python_version(), tensorflow=tf.__version__,
                            keras=keras.__version__, numpy=np.__version__, pandas=pd.__version__),
               metodo_tempo="time.perf_counter; épocas = treino + validação (loss/acc), exclui o F1 extra; "
                            "total = fit() inteiro; exclui download e preparação prévia")
    (C.DIR_HISTORICOS / f"{a.nome}{sufixo}_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    linha = dict(nome=a.nome, arquitetura=arq, resolucao=res, parametros=cfg["parametros"],
                 tamanho_modelo_mb=round(destino.stat().st_size / 1e6, 2),
                 epocas_executadas=len(h), epoca_checkpoint=ckpt.melhor_epoca,
                 val_f1_macro=round(ckpt.melhor, 4),
                 val_acuracia_no_checkpoint=round(float(h.loc[ckpt.melhor_epoca - 1, "val_accuracy"]), 4),
                 tempo_total_s=round(tempo_total, 1), tempo_medio_epoca_s=round(h["tempo_epoca_s"].mean(), 1),
                 lote=a.lote, hardware=str(gpus or "CPU"))
    print(linha)
    if not a.fumaca:
        df = pd.DataFrame([linha])
        if C.ARQ_EXPERIMENTOS.exists():
            ant = pd.read_csv(C.ARQ_EXPERIMENTOS); ant = ant[ant["nome"] != a.nome]
            df = pd.concat([ant, df], ignore_index=True)
        df.to_csv(C.ARQ_EXPERIMENTOS, index=False)


if __name__ == "__main__":
    main()
