import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import sys


sys.path.append('/home/navaneet/vqvae_transformer/')

from detr.main import build_QIL_model_and_optimizer  



import IPython
e = IPython.embed  # For debugging purposes

class QIL(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        # Build model and optimizer using the Hydra configuration.
        model, optimizer = build_QIL_model_and_optimizer(cfg)
        self.model = model  # The underlying VQ-VAE model (or other model architecture)
        self.optimizer = optimizer
        self.kl_weight = cfg.policy.kl_weight  # Weight for vector quantization loss (or commitment loss)
        print(f'KL Weight {self.kl_weight}')

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        Forward pass.
        At training time (if actions is provided), computes losses.
        At inference time, returns the predicted actions.
        """
        env_state = None
        # Normalize images using standard ImageNet statistics.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, vq_loss = self.model(qpos, image, env_state, actions, is_pad)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            # Only compute loss for non-padded timesteps.
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1  # Regression loss.
            loss_dict['vq'] = vq_loss  # Vector quantization loss.
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['vq'] * self.kl_weight  # Total loss.
            return loss_dict
        else:  # inference time
            a_hat, _, _ = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        """Return the optimizer for training."""
        return self.optimizer

    # Optional: Allow the instance to be callable via forward.
    __call__ = forward
