import torch
import torch.nn as nn
from src.layers import *
# Import the necessary components from the paper's code

# --- Define the Spiking Transformer Block ---
class SpikeBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1, T=16, L=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # The MultiheadAttention layer does not need to be aware of the time dimension
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # The MLP is now a simple sequence. The IF neuron handles the temporal dynamics.
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            # The IF neuron is initialized with T > 0 for SNN mode.
            # It will internally loop through the timesteps.
            IF(T=T, L=L),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # The input 'x' now has shape [T*B, Seq, Dim]
        # Attention is applied across all timesteps simultaneously
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output

        # The MLP with the IF neuron is applied
        x = x + self.mlp(self.norm2(x))
        return x

# --- Define the main Spiking Vision Transformer ---
class SpikeVisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10, dim=192, depth=12, heads=3, mlp_ratio=4., dropout=0.1, T=16, L=8):
        super().__init__()
        self.T = T # Timesteps

        # --- Patching and Embedding (Identical to ANN) ---
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.patch_proj = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_dropout = nn.Dropout(p=dropout)

        # --- Temporal Dimension Handlers ---
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)

        # --- Transformer Encoder ---
        # Create a stack of SpikeBlocks, configured for SNN mode (T>0)
        self.blocks = nn.ModuleList([
            SpikeBlock(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, dropout=dropout, T=T, L=L)
            for _ in range(depth)
        ])

        # --- Classification Head ---
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Input 'x' has shape [B, C, H, W]
        B, C, H, W = x.shape
        
        # --- 1. Patching and Embedding (Done once) ---
        patches = x.view(B, C, H // 4, 4, W // 4, 4).permute(0, 2, 4, 1, 3, 5).reshape(B, 64, 48)
        x = self.patch_proj(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.pos_dropout(x) # Shape: [B, Seq, Dim]

        # --- 2. Expand for Temporal Processing ---
        # Repeat the input tensor for each timestep
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1) # Shape: [T, B, Seq, Dim]
        x = self.merge(x) # Shape: [T*B, Seq, Dim]

        # --- 3. Pass through the Transformer Encoder ---
        # The data flows through all blocks. Each IF neuron inside will handle the time dimension.
        for blk in self.blocks:
            x = blk(x)

        # --- 4. Classification ---
        # Reshape back to separate the time and batch dimensions
        x = self.expand(x) # Shape: [T, B, Seq, Dim]
        
        # Average the outputs over all timesteps
        x = x.mean(0) # Shape: [B, Seq, Dim]

        # Take the output corresponding to the [CLS] token
        cls_token_output = self.norm(x[:, 0])
        logits = self.head(cls_token_output)
        return logits

print("SNN Model architecture defined according to the paper's methodology.")