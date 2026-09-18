import os, sys, shutil
sys.path.insert(0, '/Users/sohamanand/Unrealistic')
sys.path.insert(0, '/Users/sohamanand/Unrealistic/scripts')
os.chdir('/Users/sohamanand/Unrealistic')
import mlx.core as mx
from src.training.checkpointing import load_checkpoint
from huggingface_hub import HfApi
api = HfApi()
REPO = 'SohamProgrammer/Unrealistic-v1'
def flat(d, pre, out):
    if isinstance(d, dict):
        for k, v in d.items(): flat(v, pre + '.' + k, out)
    elif isinstance(d, list):
        for i, v in enumerate(d): flat(v, f'{pre}.{i}', out)
    else: out[pre] = d
jobs = [('final', 'checkpoints/step_362289'), ('post_phase7f', 'baselines/post_phase7f'), ('post_phase7g', 'baselines/post_phase7g')]
os.makedirs('.tools/mlx_stage', exist_ok=True)
for name, src in jobs:
    out = f'.tools/mlx_stage/{name}'
    os.makedirs(out, exist_ok=True)
    print(f'repack {name}...', flush=True)
    params, _, _ = load_checkpoint(src, return_opt=True)
    f = {}
    flat(params, 'params', f)
    mx.savez(os.path.join(out, 'model.npz'), **f)
    for fn in ('config.json', 'meta.json'):
        p = os.path.join(src, fn)
        if os.path.exists(p): shutil.copy(p, os.path.join(out, fn))
    api.upload_folder(folder_path=out, path_in_repo=f'mlx/{name}', repo_id=REPO)
    print(f'uploaded mlx/{name}', flush=True)
    shutil.rmtree(out, ignore_errors=True)
print('MLX REFRESH DONE', flush=True)
