import os
import pickle
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import wandb
from utils import (
    make_policy,
    make_optimizer,
    set_seed,
    load_data,
    detach_dict,
    compute_dict_mean,
)
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime

def forward_pass(data, policy, device):
    image_data, qpos_data, action_data, is_pad = data
    qpos_data = qpos_data.float()
    action_data = action_data.float()
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label='validation')
        plt.title(key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        wandb.log({f"train_val_{key}_plot": wandb.Image(plot_path)})
    # print(f'Saved plots to {ckpt_dir}')


def train_bc(train_dataloader, val_dataloader, cfg, device):
    train_cfg = cfg.train
    task_cfg = cfg.task
    policy = make_policy(cfg.policy.policy_class, cfg)
    policy.to(device)
    optimizer = make_optimizer(cfg.policy.policy_class, policy)

    # Convert the Hydra config to a plain dictionary for wandb
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    
    # Generate a unique name for each run
    run_name = f"{task_cfg.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(project=task_cfg.task_name, config=wandb_config, name=run_name)
    
    train_history = []
    validation_history = []
    min_val_loss = float('inf')
    best_ckpt_info = None

    for epoch in tqdm(range(train_cfg.num_epochs), desc="Training Epochs"):
        
        # Evaluation loop with inference mode
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for data in val_dataloader:
                forward_dict = forward_pass(data, policy, device)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            wandb.log({"epoch": epoch, "Validation Loss": epoch_val_loss, **{f'val_{k}': v.item() for k, v in epoch_summary.items()}})
        
        # Training loop
        policy.train()
        optimizer.zero_grad()
        for data in train_dataloader:
            forward_dict = forward_pass(data, policy, device)
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            wandb.log({"epoch": epoch, "Training Loss": loss.item(), **{k: v.item() for k, v in forward_dict.items()}})
            
            # Log both training and validation losses together after each epoch
            wandb.log({
                "epoch": epoch,
                "Training Loss": loss.item(),
                "Validation Loss": epoch_val_loss
            })
            
        # Optionally, save checkpoint periodically
        if epoch % 200 == 0:
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg.seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, train_cfg.checkpoint_dir, train_cfg.seed)
            wandb.save(ckpt_path)

    final_ckpt_path = os.path.join(train_cfg.checkpoint_dir, 'policy_last.ckpt')
    torch.save(policy.state_dict(), final_ckpt_path)
    wandb.save(final_ckpt_path)
    print(f'Final checkpoint saved at {final_ckpt_path}')
    wandb.finish()

