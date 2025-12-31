import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionClassifier(nn.Module):
    """
    Classification head for Emotion Recognition.
    
    Input: Fused embedding vector.
    Output: Class probabilities and confidence scores.
    """
    
    def __init__(self, input_dim, num_classes, hidden_dim=64, dropout=0.3):
        super(EmotionClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)
            
        Returns:
            logits: (batch, num_classes)
            probs: (batch, num_classes)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=1)
        
        return logits, probs
