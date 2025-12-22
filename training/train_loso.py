import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        self.ncde_ecg = NeuralCDE(1+1, config['hidden_dim']) # +1 for intensity
        self.ncde_eda = NeuralCDE(2+1, config['hidden_dim']) 
        self.ncde_acc = NeuralCDE(4+1, config['hidden_dim'])
        
        self.transformer = MultimodalTransformer(config['hidden_dim'], config['hidden_dim'], config['num_heads'], config['num_layers'])
        self.classifier = EmotionClassifier(config['hidden_dim']*3, 3) # 3 classes: Baseline, Stress, Amusement

    def forward(self, ecg_path, eda_path, acc_path, timeline):
        # 1. NCDE + Discretization
        z_ecg = sample_latent_trajectory(self.ncde_ecg, ecg_path, timeline)
        z_eda = sample_latent_trajectory(self.ncde_eda, eda_path, timeline)
        z_acc = sample_latent_trajectory(self.ncde_acc, acc_path, timeline)
        
        # 2. Fusion
        fused = self.transformer(z_ecg, z_eda, z_acc)
        
        # 3. Classification
        logits, probs = self.classifier(fused)
        return logits

def collate_paths(batch):
    """
    Pads and stacks pre-computed spline coefficients.
    """
    def pad_and_stack_coeffs(coeffs_list):
        # coeffs_list elements are (Length, Channels)
        max_len = max([c.shape[0] for c in coeffs_list])
        n_feats = coeffs_list[0].shape[1]
        
        padded = torch.zeros(len(coeffs_list), max_len, n_feats)
        for i, c in enumerate(coeffs_list):
            L = c.shape[0]
            padded[i, :L, :] = c
            if L < max_len:
                padded[i, L:, :] = c[-1, :]
        return padded

    ecg_b = pad_and_stack_coeffs([b['ecg_coeffs'] for b in batch])
    eda_b = pad_and_stack_coeffs([b['eda_coeffs'] for b in batch])
    acc_b = pad_and_stack_coeffs([b['acc_coeffs'] for b in batch])
    
    # Wrap in Spline objects
    import torchcde
    ecg_path = torchcde.CubicSpline(ecg_b)
    eda_path = torchcde.CubicSpline(eda_b)
    acc_path = torchcde.CubicSpline(acc_b)
    
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    
    return {
        'ecg_path': ecg_path,
        'eda_path': eda_path,
        'acc_path': acc_path,
        'label': labels
    }

def update_live_plot(history, output_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#e74c3c')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs, history['val_loss'], 's-', label='Val Loss', color='#3498db')
    plt.title('Training Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'o-', label='Train Acc', color='#2ecc71')
    if 'val_acc' in history and history['val_acc']:
        plt.plot(epochs, history['val_acc'], 's-', label='Val Acc', color='#f1c40f')
    plt.title('Training Accuracy Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def train_one_subject_loso(target_subject, all_subjects, data_root, config):
    print(f"\n{'='*60}")
    print(f" LOSO TRAINING: Target Subject S{target_subject}")
    print(f"{'='*60}\n")
    
    train_subjects = [s for s in all_subjects if s != target_subject]
    
    cache_root = os.path.join(PROJECT_ROOT, 'data', 'processed', 'WESAD_Coeffs')
    train_ds = WESADDataset(train_subjects, data_root, use_cache=True, cache_root=cache_root)
    val_ds = WESADDataset([target_subject], data_root, use_cache=True, cache_root=cache_root)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_paths, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_paths, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    model = FullPipelineModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    timeline = generate_fixed_timeline(0, 60, config['snapshot_hz']).to(device)
    
    results_dir = os.path.join(PROJECT_ROOT, 'results', f'training_S{target_subject}')
    os.makedirs(results_dir, exist_ok=True)
    plot_file = os.path.join(results_dir, 'live_metrics.png')

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            
            ecg_p = batch['ecg_path'].to(device)
            eda_p = batch['eda_path'].to(device)
            acc_p = batch['acc_path'].to(device)
            y = batch['label'].to(device)
            
            logits = model(ecg_p, eda_p, acc_p, timeline)
            loss = criterion(logits, y)
            
            if torch.isnan(loss):
                print("\n[CRITICAL] NaN Loss detected during training. Stopping.")
                return history

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*correct/total:.1f}%"})

        history['train_loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(100 * correct / total)

        # Validation
        model.eval()
        v_loss = 0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for batch in val_loader:
                ecg_p = batch['ecg_path'].to(device)
                eda_p = batch['eda_path'].to(device)
                acc_p = batch['acc_path'].to(device)
                y = batch['label'].to(device)
                
                logits = model(ecg_p, eda_p, acc_p, timeline)
                loss = criterion(logits, y)
                v_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                v_total += y.size(0)
                v_correct += (predicted == y).sum().item()
        
        history['val_loss'].append(v_loss / len(val_loader))
        history['val_acc'].append(100 * v_correct / v_total)
        
        print(f"Epoch {epoch+1} Results: Train Acc: {history['train_acc'][-1]:.1f}% | Val Acc: {history['val_acc'][-1]:.1f}%")
        
        # Update live plot
        update_live_plot(history, plot_file)
        print(f"Live plot updated at {plot_file}")
        
    return history

if __name__ == "__main__":
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
    
    # Subjects to include (Full dataset S2-S17, excluding S12)
    ALL_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    
    CONFIG = {
        'hidden_dim': 16,
        'num_heads': 4,
        'num_layers': 2,
        'batch_size': 4,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 10,
        'snapshot_hz': 1.0 # 1 snapshot per second
    }
    
    # Carry out LOSO for Subject 2
    history = train_one_subject_loso(target_subject=2, all_subjects=ALL_SUBJECTS, data_root=DATA_ROOT, config=CONFIG)
