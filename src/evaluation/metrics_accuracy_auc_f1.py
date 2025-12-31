import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Computes classification metrics.
    
    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_probs: Predicted probabilities (N, C) (Required for AUC)
        
    Returns:
        dict: {accuracy, f1_macro, f1_weighted, auc (opt), confusion_matrix}
    """
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm
    }
    
    if y_probs is not None:
        try:
            # Handle multiclass AUC
            # 'ovr' (One-vs-Rest) or 'ovo' (One-vs-One)
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            metrics['auc'] = auc
        except ValueError:
            # Can fail if only one class present in y_true
            metrics['auc'] = 0.0
            
    return metrics
