import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention Layer.
    
    Computes Attention(Q, K, V) where Q comes from Modality A, 
    and K, V come from Modality B.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Args:
            query: (batch, seq_len_q, embed_dim) - From Modality A
            key:   (batch, seq_len_k, embed_dim) - From Modality B
            value: (batch, seq_len_k, embed_dim) - From Modality B
            
        Returns:
            output: (batch, seq_len_q, embed_dim) - Enriched features for Modality A
        """
        
        # Multihead Attention
        # query, key, value
        attn_output, _ = self.multihead_attn(query, key, value, 
                                             key_padding_mask=key_padding_mask, 
                                             attn_mask=attn_mask)
        
        # Residual + Norm
        output = self.norm(query + self.dropout(attn_output))
        
        return output
