import os
import torch
from omegaconf import OmegaConf

from .detr_vae import build as build_vae
from omegaconf import DictConfig  # Hydra uses OmegaConf's DictConfig

import IPython
e = IPython.embed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_QIL_model(args):
    print(f"-------------{args}----------------")
    return build_vae(args)


def _to_namespace(cfg):

    class ConfigNamespace:
        pass

    cfg_obj = ConfigNamespace()
   
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg

    for k, v in cfg_dict.items():
        setattr(cfg_obj, k, v)
    return cfg_obj


def build_QIL_model_and_optimizer(cfg: DictConfig):
    # Use the Hydra DictConfig directly.
    model = build_QIL_model(cfg)
    model.to(device)

    # Build parameter groups: one group for non-backbone params, one for backbone params.
    # Assumes that lr, lr_backbone, and weight_decay are stored in cfg.policy.
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() 
                       if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() 
                       if "backbone" in n and p.requires_grad],
            "lr": cfg.policy.lr_backbone,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=cfg.policy.lr,
        weight_decay=cfg.policy.weight_decay
    )
    return model, optimizer
