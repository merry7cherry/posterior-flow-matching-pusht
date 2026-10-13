"""Public local training and native PushT evaluation entrypoints."""
import argparse
import json
import math
import os
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.pfm import log_variance

@hydra.main(version_base=None, config_path='config', config_name='pusht')
def train(cfg):
    """Hydra overrides configure local data, output directory, and device."""
    torch.set_num_threads(int(os.environ.get('PFM_CPU_THREADS', '2')))
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    checkpoint = workspace.get_checkpoint_path()
    if checkpoint.exists() and not cfg.training.resume:
        raise FileExistsError(f'{checkpoint} exists; choose a fresh output directory or training.resume=true')
    workspace.run()


def evaluate():
    parser=argparse.ArgumentParser(description='Native image PushT rollout, video and optional reference diagnostics')
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--solver',choices=['ode','sde'],default='ode')
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--diffusion-scale',type=float,default=None)
    parser.add_argument('--seed',type=int,default=321)
    parser.add_argument('--n-test',type=int,default=50)
    parser.add_argument('--n-envs',type=int,default=1)
    parser.add_argument('--max-steps',type=int,default=300)
    parser.add_argument('--data',type=Path,help='Optional native Zarr dataset for reference-state diagnostics')
    parser.add_argument('--diagnostic-batches',type=int,default=2)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    if min(args.steps,args.n_test,args.n_envs,args.max_steps,args.diagnostic_batches)<1:
        parser.error('Counts must be positive')
    scale=args.diffusion_scale if args.diffusion_scale is not None else (0.1 if args.solver=='sde' else 0.)
    if not math.isfinite(scale) or scale < 0 or (args.solver == 'ode' and scale != 0):
        parser.error('Diffusion scale must be finite and nonnegative; ODE requires zero')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(os.environ.get('PFM_CPU_THREADS','2')))
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    # Only load trusted checkpoints: the inherited checkpoint format contains dill objects.
    payload=torch.load(args.checkpoint,map_location='cpu',pickle_module=dill)
    cfg=payload['cfg']
    if cfg.policy.get('training_variant','pfm_delta') != 'pfm_delta':
        raise ValueError('Only standard PFM checkpoints are supported')
    if 'training_variant' in cfg.policy:del cfg.policy.training_variant
    policy=hydra.utils.instantiate(cfg.policy)
    key='ema_model' if cfg.training.use_ema else 'model'
    policy.load_state_dict(payload['state_dicts'][key])
    policy.to(args.device).eval()
    policy.sampling.update(solver=args.solver,steps=args.steps,diffusion_scale=scale,seed=args.seed)
    cfg.task.env_runner.update(n_train=0,n_train_vis=0,n_test=args.n_test,n_test_vis=min(1,args.n_test),n_envs=args.n_envs,max_steps=args.max_steps)
    runner=hydra.utils.instantiate(cfg.task.env_runner,output_dir=str(args.output))
    try:result=runner.run(policy)
    finally:runner.env.close()
    result.update(sampling=policy.sampling,device=args.device,checkpoint=str(args.checkpoint),weights=key)
    (args.output/'evaluation.json').write_text(json.dumps(result,indent=2))
    if args.data:
        cfg.task.dataset.zarr_path=str(args.data)
        dataset=hydra.utils.instantiate(cfg.task.dataset).get_validation_dataset()
        loader=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=False)
        records=[]
        for time in [.1,.5,.9,.99]:
            values=[]
            for i,batch in enumerate(loader):
                if i>=args.diagnostic_batches:break
                batch=dict_apply(batch,lambda x:x.to(args.device))
                with torch.no_grad():
                    obs=policy.normalizer.normalize(batch['obs'])
                    image=dict_apply(obs,lambda x:x[:,:2].reshape(-1,*x.shape[2:]))
                    condition=policy.obs_encoder(image).reshape(len(batch['action']),-1)
                    x1=policy.normalizer['action'].normalize(batch['action']).float()
                    x0=policy.noise_scale*torch.randn_like(x1)
                    t=torch.full((len(x1),),time,device=args.device)
                    v,raw=policy._field((1-time)*x0+time*x1,t,condition)
                    ell=log_variance(raw,policy.logvar_min,policy.logvar_max)
                    residual=(x1-x0-v.float()).square()
                    values.append([float(ell.exp().mean()),float(residual.mean()),float(.5*(ell+residual*ell.neg().exp()).mean())])
            if not values:raise ValueError('No validation windows available')
            var,mse,nll=np.mean(values,axis=0)
            records.append(dict(time=time,velocity_variance=var,residual_mse=mse,gaussian_nll_without_constant=nll,endpoint_variance=(1-time)**2*var))
        (args.output/'diagnostics.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(result,indent=2))
