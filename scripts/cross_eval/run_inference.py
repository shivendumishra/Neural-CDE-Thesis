import torch
import os
import sys
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from training.train_loso import FullPipelineModel
from cross_eval.affective_road_loader import AffectiveRoadDataset
from latent_discretization.temporal_sampling import generate_fixed_timeline
from normalization.intensity_channel import add_intensity_channel
import torchcde

def run_cross_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")
    
    # 1. Load Model
    config = {'hidden_dim': 16, 'num_heads': 4, 'num_layers': 2}
    model = FullPipelineModel(config).to(device)
    
    weights_path = os.path.join(PROJECT_ROOT, 'cross_eval', 'wesad_model.pth')
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}. Run train_and_save.py first.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 2. Load AffectiveRoad Data
    data_root = os.path.join(PROJECT_ROOT, 'data', 'AffectiveROAD_Data')
    dataset = AffectiveRoadDataset(data_root)
    
    if len(dataset) == 0:
        print("No samples found in AffectiveRoad.")
        return
        
    from torch.utils.data import DataLoader
    from training.train_loso import collate_paths
    
    # Use batch size 16 for fast inference
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_paths)
    
    timeline = generate_fixed_timeline(0, 60, 1.0).to(device)
    
    print("\n--- Cross-Dataset Evaluation on Affective Road ---")
    print(f"Total samples found: {len(dataset)}")
    
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            ecg_p = batch['ecg_path'].to(device)
            eda_p = batch['eda_path'].to(device)
            acc_p = batch['acc_path'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(ecg_p, eda_p, acc_p, timeline)
            _, predicted = torch.max(logits, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    accuracy = 100 * correct / total
    
    print(f"\n{'='*40}")
    print(f" FINAL RESULTS (Affective Road)")
    print(f"{'='*40}")
    print(f"Total Samples:  {total}")
    print(f"Correct:        {correct}")
    print(f"Accuracy:       {accuracy:.2f}%")
    print(f"{'='*40}")
    
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    target_names = ["Baseline", "Stress", "Amusement"]
    present_labels = sorted(list(set(y_true) | set(y_pred)))
    labels_to_show = [target_names[l] for l in present_labels]
    
    print(classification_report(y_true, y_pred, target_names=labels_to_show, labels=present_labels))

if __name__ == "__main__":
    run_cross_inference()
