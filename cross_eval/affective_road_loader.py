import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from utils.config import FS_EDA, FS_ACC

class AffectiveRoadDataset(Dataset):
    def __init__(self, data_root, window_size_sec=60, step_size_sec=10):
        self.samples = []
        self.window_size_sec = window_size_sec
        
        e4_root = os.path.join(data_root, 'Database', 'E4')
        annot_path = os.path.join(e4_root, 'Annot_E4_Left.csv')
        
        if not os.path.exists(annot_path):
            print(f"Error: Annotation file not found at {annot_path}")
            return

        annots = pd.read_csv(annot_path)
        
        # For the fast demo, we only process the first drive (Drv1)
        for _, row in annots.iloc[:1].iterrows():
            drive_id = row['Drive-id']
            drive_folder = None
            for d in os.listdir(e4_root):
                if d.endswith(drive_id):
                    drive_folder = os.path.join(e4_root, d, 'Left')
                    break
            
            if not drive_folder or not os.path.exists(drive_folder):
                continue
                
            print(f"Pre-loading {drive_id} for inference...")
            
            try:
                # Load Signals
                eda_df = pd.read_csv(os.path.join(drive_folder, 'EDA.csv'), skiprows=2, header=None)
                acc_df = pd.read_csv(os.path.join(drive_folder, 'ACC.csv'), skiprows=2, header=None)
                ibi_df = pd.read_csv(os.path.join(drive_folder, 'IBI.csv'), skiprows=1, header=None)
                
                eda_raw = eda_df[0].values.astype(np.float32)
                acc_raw = acc_df.values.astype(np.float32)
                ibi_offsets = ibi_df[0].values.astype(np.float32)
                ibi_values = ibi_df[1].values.astype(np.float32)
                
                duration_sec = len(eda_raw) / 4.0
                
                # Windowing with Ground Truth Labels
                current_time = 0.0
                
                # Create a list of (start, end, label) from the row
                # Mapping: Rest -> 0 (Baseline), City/Hwy -> 1 (Stress)
                intervals = []
                cols = annots.columns[1:]
                for i in range(0, len(cols), 2):
                    label_str = cols[i].split('_')[0]
                    start = row[cols[i]]
                    end = row[cols[i+1]]
                    
                    if label_str == 'Rest':
                        l = 0
                    elif label_str in ['City1', 'City2', 'Hwy']:
                        l = 1
                    else:
                        continue # Skip 'Z' or others
                    intervals.append((start, end, l))

                while current_time + window_size_sec <= duration_sec:
                    # Find label for this window
                    # A window is valid if it's entirely within one interval
                    win_label = None
                    for start, end, l in intervals:
                        if current_time >= start and (current_time + window_size_sec) <= end:
                            win_label = l
                            break
                    
                    if win_label is None:
                        current_time += step_size_sec
                        continue

                    start_idx_4hz = int(current_time * 4.0)
                    end_idx_4hz = start_idx_4hz + int(window_size_sec * 4.0)
                    start_idx_32hz = int(current_time * 32.0)
                    end_idx_32hz = start_idx_32hz + int(window_size_sec * 32.0)
                    
                    if end_idx_4hz > len(eda_raw) or end_idx_32hz > len(acc_raw): break
                        
                    mask = (ibi_offsets >= current_time) & (ibi_offsets < current_time + window_size_sec)
                    win_ibi = ibi_values[mask]
                    win_ibi_times = ibi_offsets[mask] - current_time
                    
                    if len(win_ibi) >= 2:
                        self.samples.append({
                            'ibi': win_ibi,
                            'ibi_times': win_ibi_times,
                            'eda': eda_raw[start_idx_4hz:end_idx_4hz],
                            'acc': acc_raw[start_idx_32hz:end_idx_32hz],
                            'label': win_label
                        })
                    current_time += step_size_sec
            except Exception as e:
                print(f"Error: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        from preprocessing.eda_preprocessing_cvxeda import eda_preprocessing
        from preprocessing.acc_preprocessing_filtering import preprocess_accelerometer
        from preprocessing.acc_statistical_features import extract_acc_features
        
        _, phasic, tonic = eda_preprocessing(sample['eda'], 4.0)
        mag = preprocess_accelerometer(sample['acc'], 32.0)
        
        acc_feats = []
        for i in range(0, len(mag), 32):
            w = mag[i:i+32]
            if len(w) < 32: break
            f = extract_acc_features(w)
            acc_feats.append([f['acc_mean'], f['acc_var'], f['acc_energy'], f['acc_entropy']])
            
        # Compute Coefficients for CDE
        import torchcde
        from normalization.intensity_channel import add_intensity_channel
        
        def build_coeffs(data, times):
            d_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            t_tensor = torch.tensor(times, dtype=torch.float32).unsqueeze(0)
            d_in = add_intensity_channel(d_tensor, t_tensor)
            return torchcde.natural_cubic_coeffs(d_in).squeeze(0)
            
        return {
            'ecg_coeffs': build_coeffs(sample['ibi'].reshape(-1, 1), sample['ibi_times']),
            'eda_coeffs': build_coeffs(np.stack([phasic, tonic], axis=1), np.arange(len(phasic))/4.0),
            'acc_coeffs': build_coeffs(np.array(acc_feats), np.arange(len(acc_feats))),
            'label': sample['label']
        }
