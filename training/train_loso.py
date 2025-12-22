import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from training.wesad_dataset import WESADDataset
from neural_cde.neural_cde_model import NeuralCDE
from continuous_path.cubic_spline_construction import build_spline
from latent_discretization.temporal_sampling import generate_fixed_timeline, sample_latent_trajectory
from fusion_transformer.multimodal_transformer import MultimodalTransformer
from classifier.emotion_classifier import EmotionClassifier
from normalization.intensity_channel import add_intensity_channel
from normalization.zscore_normalization import zscore_normalize

class FullPipelineModel(nn.Module):
    def __init__(self, config):
        super(FullPipelineModel, self).__init__()
        # Simplified for brevity in this file summary
        self.ncde_ecg = NeuralCDE(1+1, config['hidden_dim']) # +1 for intensity
        self.ncde_eda = NeuralCDE(2+1, config['hidden_dim']) 
        self.ncde_acc = NeuralCDE(4+1, config['hidden_dim'])
        
        self.transformer = MultimodalTransformer(config['hidden_dim'], config['num_heads'], config['num_layers'])
        self.classifier = EmotionClassifier(config['hidden_dim'], 3) # 3 classes: Baseline, Stress, Amusement

    def forward(self, ecg_path, eda_path, acc_path, timeline):
        # 1. NCDE + Discretization
        z_ecg = sample_latent_trajectory(self.ncde_ecg, ecg_path, timeline)
        z_eda = sample_latent_trajectory(self.ncde_eda, eda_path, timeline)
        z_acc = sample_latent_trajectory(self.ncde_acc, acc_path, timeline)
        
        # 2. Fusion
        fused = self.transformer(z_ecg, z_eda, z_acc)
        
        # 3. Classification
        logits = self.classifier(fused)
        return logits

def collate_paths(batch):
    """
    Collate function to build continuous paths for a batch of samples.
    """
    def pad_and_stack(seqs, times):
        max_len = max([len(s) for s in seqs])
        n_feats = seqs[0].shape[1]
        
        # Output tensors
        data_padded = torch.zeros(len(seqs), max_len, n_feats)
        times_padded = torch.zeros(len(seqs), max_len)
        
        for i, (s, t) in enumerate(zip(seqs, times)):
            L = len(s)
            s_tensor = torch.tensor(s, dtype=torch.float32)
            t_tensor = torch.tensor(t, dtype=torch.float32)
            
            data_padded[i, :L, :] = s_tensor
            times_padded[i, :L] = t_tensor
            
            if L < max_len:
                # LOCF: Carry the last observation forward to replace zero-padding
                last_val = s_tensor[-1, :]
                data_padded[i, L:, :] = last_val
                
                # LINEAR EXTENSION for Time: Continue time increments to keep the path monotonic
                last_t = t_tensor[-1]
                dt = (t_tensor[-1] - t_tensor[-2]) if L > 1 else 1.0
                times_padded[i, L:] = last_t + torch.arange(1, max_len - L + 1).float() * dt
            
        return data_padded, times_padded

    # Extract sequences
    ecg_seqs = [b['ecg_seq'] for b in batch]
    eda_seqs = [b['eda_seq'] for b in batch]
    acc_seqs = [b['acc_seq'] for b in batch]
    
    # Pad and stack
    ecg_data, ecg_t = pad_and_stack(ecg_seqs, [b['ecg_time'] for b in batch])
    eda_data, eda_t = pad_and_stack(eda_seqs, [b['eda_time'] for b in batch])
    acc_data, acc_t = pad_and_stack(acc_seqs, [b['acc_time'] for b in batch])
    
    # 1. Z-Score Normalization
    ecg_data, _, _ = zscore_normalize(ecg_data)
    eda_data, _, _ = zscore_normalize(eda_data)
    acc_data, _, _ = zscore_normalize(acc_data)
    
    # 2. Augment with Intensity/Time Channel
    ecg_in = add_intensity_channel(ecg_data, ecg_t)
    eda_in = add_intensity_channel(eda_data, eda_t)
    acc_in = add_intensity_channel(acc_data, acc_t)
    
    # Build continuous paths (Splines)
    ecg_path = build_spline(ecg_in)
    eda_path = build_spline(eda_in)
    acc_path = build_spline(acc_in)
    
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    
    return {
        'ecg_path': ecg_path,
        'eda_path': eda_path,
        'acc_path': acc_path,
        'label': labels
    }

def train_loso():
    # Placeholder for the main loop
    print("Starting LOSO Cross-Validation...")
    # ... Training logic ...

if __name__ == "__main__":
    train_loso()
