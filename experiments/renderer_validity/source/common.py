"""Do not import older bootstrap modules: all new writes stay in main_python."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / 'artifacts'
OUT.mkdir(exist_ok=True)
DATA = ROOT / 'main_python' / 'dataset' / 'CUB_200_2011'
PREVIOUS = ROOT / '临时'
for key in ('TMP', 'TEMP', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'NUMBA_CACHE_DIR'):
    target = OUT / 'runtime_cache' / key.lower()
    target.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(target)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def write_json(path, value):
    import json
    target = Path(path).resolve()
    assert target.is_relative_to(BASE.resolve()), target
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def configure(seed=0):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
