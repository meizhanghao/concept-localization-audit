"""Package small frozen inputs and the executed sources for numerical review."""
from .common import ROOT, OUT
from .frontier_protocol import DEST
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np


def main():
    dest=Path(__file__).resolve().parent/'frontier_bundle'
    data=dest/'data';data.mkdir(exist_ok=True)
    core=ROOT/'github_cun/main/data/core_audit/data'
    shutil.copyfile(core/'FB_mask_context.npz',data/'context.npz')
    banks=[]
    for seed in [821,2026111909,2026111910]:
        banks.append(dict(np.load(OUT/f'head_robustness_2026-11-19/fb_{seed}.npz')))
    banks.append(dict(np.load(DEST/'fb_channel_peaks.npz')))
    random=np.stack([np.stack([np.stack([bank[f'head{h}__{name}'].reshape(1000,2) for h in range(3)])
                               for name in ['cam','gradient']]) for bank in banks])
    trained=np.stack([np.load(core/f'FB_mask_{name}.npz')['peaks'][0] for name in ['cam','gradient']])
    head_indices=np.load(DEST/'reference_order_draws.npz')['head_indices']
    np.savez_compressed(data/'reference_inputs.npz',random=random,trained=trained,head_indices=head_indices,
        bank_names=np.array(['gaussian_821','gaussian_2026111909','gaussian_2026111910','channel']))
    for name in ['training_lambda_0.0.npz','training_lambda_5.0.npz','training_inference.npz','criterion_inputs.npz','protocol.json']:
        shutil.copyfile(DEST/name,data/name)
    source=dest/'source';source.mkdir(exist_ok=True)
    modules=['frontier_protocol.py','frontier_criterion.py','frontier_references.py',
             'frontier_training.py','frontier_training_check.py','frontier_interactions.py',
             'frontier_export.py','models.py','funnybirds.py','common.py']
    for name in modules:shutil.copyfile(Path(__file__).resolve().parent/name,source/name)
    files={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (dest/'manifest.json').write_text(json.dumps({'scope':'Later exploratory numerical extension, separate from the immutable base snapshot.',
        'files':files},indent=2),encoding='utf-8')
    print(dest,len(files),'files',sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'bytes')


if __name__=='__main__':main()
