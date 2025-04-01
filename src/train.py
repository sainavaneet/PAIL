import os
import pickle
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from utils import set_seed, load_data
from src.main import train_bc

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))

    device = cfg.policy.device if "device" in cfg.policy else "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["DEVICE"] = device
    print(f"Using device: {device}")

    # Set random seed
    set_seed(cfg.train.seed)

    # Create checkpoint directory
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    # Load dataset
    data_dir = cfg.task.dataset_dir
    num_episodes = cfg.task.TOTAL_EPISODES
    train_dataloader, val_dataloader, norm_stats, is_sim = load_data(
        data_dir,
        num_episodes,
        cfg.task.camera_names,
        cfg.train.batch_size_train,
        cfg.train.batch_size_val,
    )

    # Save dataset stats
    stats_path = os.path.join(cfg.train.checkpoint_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(norm_stats, f)
    print(f"Dataset stats saved to {stats_path}")

    # Start training
    train_bc(train_dataloader, val_dataloader, cfg, device)

if __name__ == '__main__':
    main()
