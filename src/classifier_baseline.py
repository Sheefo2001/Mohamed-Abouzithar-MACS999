import os
import logging
from typing import Dict, Any, Tuple, Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BehaveFormerMAE(nn.Module):
    """
    Cross-Modal Masked Autoencoder (MAE) for Behavioral Biometrics.
    Uses Anomaly Detection (Reconstruction Error) instead of Classification.
    """
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, nhead: int = 4, mask_ratio: float = 0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio
        
        # Encoders (Linear projection to hidden_dim)
        self.mouse_proj = nn.Linear(4, hidden_dim)
        self.key_proj = nn.Linear(2, hidden_dim)
        
        # Learnable mask tokens
        self.mouse_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.key_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Positional Embeddings (simple learnable 1D for max length 2048)
        self.mouse_pos_embed = nn.Parameter(torch.zeros(1, 2048, hidden_dim))
        self.key_pos_embed = nn.Parameter(torch.zeros(1, 2048, hidden_dim))
        
        # Modality type embeddings (to distinguish mouse from key in the fused sequence)
        self.modality_embed = nn.Parameter(torch.zeros(1, 2, hidden_dim)) # 0: mouse, 1: key
        
        # Shared Fusion Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoders (Linear projection back to original dimensions)
        self.mouse_decoder = nn.Linear(hidden_dim, 4)
        self.key_decoder = nn.Linear(hidden_dim, 2)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mouse_mask_token, std=.02)
        nn.init.normal_(self.key_mask_token, std=.02)
        nn.init.normal_(self.mouse_pos_embed, std=.02)
        nn.init.normal_(self.key_pos_embed, std=.02)
        nn.init.normal_(self.modality_embed, std=.02)
        
    def forward(self, batch: Dict[str, torch.Tensor], apply_mask: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mouse_s = batch.get('mouse_spatial')      # (B, N, 2)
        mouse_k = batch.get('mouse_kinematics')   # (B, N, 2)
        key_t = batch.get('keystroke_timing')     # (B, M, 2)
        
        mouse_pad_mask = batch.get('mouse_mask') # (B, N) True if padding
        key_pad_mask = batch.get('key_mask')     # (B, M) True if padding
        
        device = next(self.parameters()).device
        B = mouse_s.shape[0] if mouse_s is not None else (key_t.shape[0] if key_t is not None else 1)
        
        mouse_features = None
        key_features = None
        N = 0
        M = 0
        
        if mouse_s is not None and mouse_k is not None:
            N = mouse_s.shape[1]
            mouse_raw = torch.cat([mouse_s, mouse_k], dim=-1) # (B, N, 4)
            mouse_features = self.mouse_proj(mouse_raw) + self.mouse_pos_embed[:, :N, :] + self.modality_embed[:, 0:1, :]
            
            if apply_mask:
                # Random masking (only mask unpadded elements)
                rand = torch.rand(B, N, device=device)
                if mouse_pad_mask is not None:
                    rand[mouse_pad_mask] = 1.0 # Never mask padded tokens (so they don't get loss)
                mask_bool = rand < self.mask_ratio
                mouse_features[mask_bool] = self.mouse_mask_token
            else:
                mask_bool = torch.zeros(B, N, dtype=torch.bool, device=device)
                
            mouse_mask_bool = mask_bool

        if key_t is not None:
            M = key_t.shape[1]
            key_features = self.key_proj(key_t) + self.key_pos_embed[:, :M, :] + self.modality_embed[:, 1:2, :]
            
            if apply_mask:
                rand = torch.rand(B, M, device=device)
                if key_pad_mask is not None:
                    rand[key_pad_mask] = 1.0
                mask_bool = rand < self.mask_ratio
                key_features[mask_bool] = self.key_mask_token
            else:
                mask_bool = torch.zeros(B, M, dtype=torch.bool, device=device)
                
            key_mask_bool = mask_bool

        # Fuse modalities by concatenating along sequence length
        if mouse_features is not None and key_features is not None:
            fused_seq = torch.cat([mouse_features, key_features], dim=1) # (B, N+M, hidden)
            
            # Combine pad masks
            if mouse_pad_mask is None:
                mouse_pad_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            if key_pad_mask is None:
                key_pad_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
                
            fused_pad_mask = torch.cat([mouse_pad_mask, key_pad_mask], dim=1) # (B, N+M)
        elif mouse_features is not None:
            fused_seq = mouse_features
            fused_pad_mask = mouse_pad_mask if mouse_pad_mask is not None else torch.zeros(B, N, dtype=torch.bool, device=device)
        else:
            fused_seq = key_features
            fused_pad_mask = key_pad_mask if key_pad_mask is not None else torch.zeros(B, M, dtype=torch.bool, device=device)

        # Apply shared transformer
        fused_encoded = self.transformer(fused_seq, src_key_padding_mask=fused_pad_mask)
        
        # Split back and decode
        mouse_recon = None
        key_recon = None
        
        if mouse_features is not None and key_features is not None:
            mouse_encoded = fused_encoded[:, :N, :]
            key_encoded = fused_encoded[:, N:, :]
            mouse_recon = self.mouse_decoder(mouse_encoded)
            key_recon = self.key_decoder(key_encoded)
            return mouse_recon, key_recon, mouse_mask_bool, key_mask_bool
        elif mouse_features is not None:
            mouse_recon = self.mouse_decoder(fused_encoded)
            return mouse_recon, None, mouse_mask_bool, None
        else:
            key_recon = self.key_decoder(fused_encoded)
            return None, key_recon, None, key_mask_bool

# --- Evaluation and Perturbation ---
try:
    from src.offensive_engine import BiologicalDecayModule
    # Aggressive parameters for deepfake evaluation
    decay_module = BiologicalDecayModule(mu=0.01, sigma=0.4, act_r_d=0.35)
except ImportError as e:
    logger.error(f"Failed to import BiologicalDecayModule: {e}")
    decay_module = None

def apply_adversarial_perturbation(batch: Dict[str, torch.Tensor], apply: bool = False) -> Dict[str, torch.Tensor]:
    if apply and decay_module is not None:
        device = next(decay_module.parameters()).device
        if batch.get('mouse_spatial') is not None:
            decay_module.to(batch['mouse_spatial'].device)
        return decay_module(batch)
    return batch

def compute_reconstruction_loss(recon: torch.Tensor, target: torch.Tensor, pad_mask: torch.Tensor, is_training: bool = True) -> torch.Tensor:
    """
    Computes MSE loss only on valid (unpadded) tokens.
    """
    if recon is None or target is None:
        return torch.tensor(0.0, device=recon.device if recon is not None else target.device)
        
    loss = F.mse_loss(recon, target, reduction='none')
    
    # Mask out padded tokens
    loss = loss * (~pad_mask).unsqueeze(-1).float()
    
    # Average over valid tokens
    valid_tokens = (~pad_mask).sum().clamp(min=1)
    return loss.sum() / valid_tokens

def collate_biometric_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    MAX_LEN = 2048
    mouse_s_list = [item['mouse_spatial'][:MAX_LEN] for item in batch]
    mouse_k_list = [item['mouse_kinematics'][:MAX_LEN] for item in batch]
    key_t_list = [item['keystroke_timing'][:MAX_LEN] for item in batch]
    
    from torch.nn.utils.rnn import pad_sequence
    
    mouse_s_padded = pad_sequence(mouse_s_list, batch_first=True, padding_value=0.0)
    mouse_k_padded = pad_sequence(mouse_k_list, batch_first=True, padding_value=0.0)
    key_t_padded = pad_sequence(key_t_list, batch_first=True, padding_value=0.0)
    
    mouse_mask = torch.zeros(len(batch), mouse_s_padded.shape[1], dtype=torch.bool)
    for i, t in enumerate(mouse_s_list):
        if t.shape[0] < mouse_s_padded.shape[1]:
            mouse_mask[i, t.shape[0]:] = True
            
    key_mask = torch.zeros(len(batch), key_t_padded.shape[1], dtype=torch.bool)
    for i, t in enumerate(key_t_list):
        if t.shape[0] < key_t_padded.shape[1]:
            key_mask[i, t.shape[0]:] = True
            
    labels_list = [item['label'] for item in batch]
    labels_padded = torch.stack(labels_list)
            
    return {
        'mouse_spatial': mouse_s_padded,
        'mouse_kinematics': mouse_k_padded,
        'keystroke_timing': key_t_padded,
        'mouse_mask': mouse_mask,
        'key_mask': key_mask,
        'label': labels_padded
    }

if __name__ == '__main__':
    logger.info("Running BehaveFormerMAE smoke test...")
    model = BehaveFormerMAE(hidden_dim=64)
    
    mock_batch = [
        {
            'mouse_spatial': torch.rand(120, 2),
            'mouse_kinematics': torch.rand(120, 2),
            'keystroke_timing': torch.rand(50, 2)
        },
        {
            'mouse_spatial': torch.rand(150, 2),
            'mouse_kinematics': torch.rand(150, 2),
            'keystroke_timing': torch.rand(40, 2)
        }
    ]
    
    collated = collate_biometric_batch(mock_batch)
    
    m_recon, k_recon, m_mask, k_mask = model(collated)
    logger.info(f"Mouse Recon shape: {m_recon.shape}")
    logger.info(f"Key Recon shape: {k_recon.shape}")
    logger.info("Smoke test passed.")
