import torch
import torch.nn as nn
from evaluation.metrics_accuracy_auc_f1 import compute_metrics

def validate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    
    Args:
        model: The full pipeline model (FusionTransformer + components).
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: 'cuda' or 'cpu'.
        
    Returns:
        avg_loss: Scalar float
        metrics_dict: Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch
            # Assuming batch structure: [X_ecg, X_eda, X_acc], labels, (timelines if needed)
            
            # Since our model takes Spline objects, the dataloader might yield coefficients 
            # and we build splines here, or yields raw data and we build splines.
            # "Continuous Path Construction" is a stage. Ideally, it's part of the model forward 
            # or done just before.
            # The NeuralCDE expects Spline. 
            
            # Let's assume dataloader provides:
            # ecg_data, eda_data, acc_data, labels
            # And we need to build splines and valid timelines.
            
            ecg_data, eda_data, acc_data, labels = batch
            
            ecg_data = ecg_data.to(device)
            eda_data = eda_data.to(device)
            acc_data = acc_data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            # The 'model' passed here is likely a wrapper 'FullModel' or we handle the pipeline here.
            # The prompt implies modularity. 
            # If 'model' is just the FusionTransformer, we need the NCDEs too.
            # We likely pass a 'FullPipelineModel' that contains NCDEs + Transformer.
            
            logits, probs = model(ecg_data, eda_data, acc_data)
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            
            _, preds = torch.max(logits, 1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Concatenate
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    
    # Metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)
    
    return avg_loss, metrics
