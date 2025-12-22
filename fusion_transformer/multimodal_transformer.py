import torch
import torch.nn as nn
from fusion_transformer.cross_modal_attention import CrossModalAttention

class MultimodalTransformer(nn.Module):
    """
    Fuses discretized latent embeddings from ECG, EDA, and ACC using Cross-Modal Transformers.
    """
    
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(FusionTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Projectors to common embedding dimension if needed
        self.proj_ecg = nn.Linear(input_dim, embed_dim)
        self.proj_eda = nn.Linear(input_dim, embed_dim)
        self.proj_acc = nn.Linear(input_dim, embed_dim)
        
        # Position embeddings? 
        # CDE output handles time somewhat, but Transformer might need PE.
        # For simplicity and since we use CDE (which encodes time evolution in state), 
        # explicit PE might be secondary, but usually good to add.
        # We'll skip complex PE for now or assume z(t) carries history.
        
        # Cross-Modal Attention Blocks
        # Strategy:
        # A simple powerful fusion: 
        # 1. Update ECG using EDA and ACC
        # 2. Update EDA using ECG and ACC
        # 3. Update ACC using ECG and EDA
        # Or Just Concat and Self-Attend.
        
        # Prompt: "Cross-modal attention (Q, K, V from different modalities)"
        # "Learn inter-signal dependencies (ECG <-> EDA <-> ACC)"
        
        # We will implement a cycle or star fusion.
        # Let's do:
        # Fused = CrossAttn(Q=ECG, K=EDA) + CrossAttn(Q=ECG, K=ACC) ...?
        # A cleaner "Research-Grade" approach is a multimodal transformer encoder 
        # where tokens from all modalities are fed in, effectively doing cross-attention (Self-Attn over union).
        # BUT, to be explicit about "Cross-modal attention (Q,K,V from DIFFERENT)", 
        # we can use specific layers.
        
        # Implementation:
        # Stack of:
        # ECG_new = CMA(Q=ECG, K=EDA) + CMA(Q=ECG, K=ACC)
        # EDA_new = ...
        
        # Simplified "Cross-Modal" Layer for strict compliance:
        # We will use one Cross-Modal Block that creates a "Fused" representation.
        # Or better: Pairwise cross-attention then concat.
        
        self.cma_ecg_eda = CrossModalAttention(embed_dim, num_heads, dropout)
        self.cma_ecg_acc = CrossModalAttention(embed_dim, num_heads, dropout)
        
        self.cma_eda_ecg = CrossModalAttention(embed_dim, num_heads, dropout)
        self.cma_eda_acc = CrossModalAttention(embed_dim, num_heads, dropout)
        
        self.cma_acc_ecg = CrossModalAttention(embed_dim, num_heads, dropout)
        self.cma_acc_eda = CrossModalAttention(embed_dim, num_heads, dropout)
        
        # Final Self-Attention to mix everything
        self.self_attn_layer = nn.TransformerEncoderLayer(d_model=embed_dim*3, nhead=num_heads, batch_first=True)
        
        self.fusion_norm = nn.LayerNorm(embed_dim*3)

    def forward(self, z_ecg, z_eda, z_acc):
        """
        Args:
            z_ecg: (batch, seq, input_dim)
            z_eda: (batch, seq, input_dim)
            z_acc: (batch, seq, input_dim)
        """
        
        # Project
        h_ecg = self.proj_ecg(z_ecg)
        h_eda = self.proj_eda(z_eda)
        h_acc = self.proj_acc(z_acc)
        
        # Cross-Modal Updates
        # ECG enriched by EDA and ACC
        ecg_from_eda = self.cma_ecg_eda(h_ecg, h_eda, h_eda)
        ecg_from_acc = self.cma_ecg_acc(h_ecg, h_acc, h_acc)
        h_ecg_fused = h_ecg + ecg_from_eda + ecg_from_acc
        
        # EDA enriched by ECG and ACC
        eda_from_ecg = self.cma_eda_ecg(h_eda, h_ecg, h_ecg)
        eda_from_acc = self.cma_eda_acc(h_eda, h_acc, h_acc)
        h_eda_fused = h_eda + eda_from_ecg + eda_from_acc
        
        # ACC enriched by ECG and EDA
        acc_from_ecg = self.cma_acc_ecg(h_acc, h_ecg, h_ecg)
        acc_from_eda = self.cma_acc_eda(h_acc, h_eda, h_eda)
        h_acc_fused = h_acc + acc_from_ecg + acc_from_eda
        
        # Concatenate: (batch, seq, 3*embed_dim)
        fused = torch.cat([h_ecg_fused, h_eda_fused, h_acc_fused], dim=-1)
        
        # Final mixing via Self-Attention (optional but good for temporally aggregating if needed)
        # Or just pooling.
        # The prompt says "LayerNorm + residuals" which we did in CMA.
        # "Output: Fused emotion embedding" -> implies a vector or sequence.
        # Usually for classification we pool.
        
        # Let's apply one self-attention pass over the fused sequence to be sure
        fused_seq = self.self_attn_layer(fused)
        
        # Global Average Pooling to get single embedding vector
        embedding = torch.mean(fused_seq, dim=1) # (batch, 3*embed_dim)
        
        return embedding
