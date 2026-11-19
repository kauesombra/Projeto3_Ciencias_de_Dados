"""Copia só as 10 classes do projeto de uma pasta NWPU-RESISC45 completa (45 classes)
para dados/raw/NWPU-RESISC45/ (os originais NÃO são alterados).

Uso: python -m src.filtrar_classes "C:/caminho/para/NWPU-RESISC45"
"""
import shutil
import sys
from pathlib import Path

from . import config as C

origem = Path(sys.argv[1])
for c in C.CLASSES:
    destino = C.DIR_RAW / c
    shutil.copytree(origem / c, destino, dirs_exist_ok=True)
    print(f"{c}: {len(list(destino.glob('*.jpg')))} imagens")
