import torch
import torch.nn as nn
from src.layers import *

# In this cell, we define the complete architecture of our Vision Transformer.

# --- Define the Transformer Encoder Block ---
# The Transformer is made of a stack of these blocks.
class Block(nn.Module):
    """A single block of the Transformer Encoder."""
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1, T=0, L=8):
        """
        Args:
            dim (int): The embedding dimension (e.g., 192). This is the size of the vectors.
            num_heads (int): The number of attention heads (e.g., 3).
            mlp_ratio (float): Determines the hidden size of the MLP. hidden_size = dim * mlp_ratio.
            dropout (float): The dropout rate for regularization.
            T (int): Timesteps for SNN simulation. Set to 0 for ANN training.
            L (int): Quantization levels.
        """
        super().__init__()
        # First part of the block: Layer Normalization followed by Multi-Head Self-Attention.
        self.norm1 = nn.LayerNorm(dim) # Normalizes the input vectors for stability.
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,         # The dimension of the input vectors.
            num_heads=num_heads,   # The number of parallel attention mechanisms.
            dropout=dropout,       # Dropout applied to the attention scores.
            batch_first=True,      # IMPORTANT: We expect input tensors of shape [Batch, Sequence, Dim].
        )

        # Second part of the block: Layer Normalization followed by a Multi-Layer Perceptron (MLP).
        self.norm2 = nn.LayerNorm(dim) # Another normalization layer for stability.
        mlp_hidden_dim = int(dim * mlp_ratio) # Calculate the size of the MLP's hidden layer.
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), # First linear layer expands the vector to learn more complex feature.
            IF(T=T, L=L),                   # Replace nn.ReLU() with the IF neuron.
            nn.Dropout(dropout),            # Another dropout layer for regularization.
            nn.Linear(mlp_hidden_dim, dim), # Second linear layer brings the vector back to the original dimension.
            nn.Dropout(dropout),            # Final dropout layer.
        )

    def forward(self, x):
        """The forward pass of the block."""
        # The first residual connection (skip connection).
        # We pass the normalized input to the attention layer and add the original input back.
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output

        # The second residual connection.
        # We pass the result through the MLP and add the input to the MLP back.
        x = x + self.mlp(self.norm2(x))
        return x

# --- Define the main Vision Transformer ---
class VisionTransformer(nn.Module):
    """The main Vision Transformer model."""
    def __init__(self, img_size=32, patch_size=4, num_classes=10, dim=192, depth=12, heads=3, mlp_ratio=4., dropout=0.1, T=0, L=8):
        """
        Args:
            img_size (int): The size of the input image (e.g., 32 for CIFAR-10).
            patch_size (int): The size of each image patch (e.g., 4).
            num_classes (int): The number of output classes (e.g., 10 for CIFAR-10).
            dim (int): The embedding dimension for each patch.
            depth (int): The number of Transformer Blocks to stack.
            heads (int): The number of attention heads in each block.
            mlp_ratio (float): The expansion ratio for the MLP in each block.
            dropout (float): The dropout rate.
            T (int): Timesteps for SNN simulation. Set to 0 for ANN training.
            L (int): Quantization levels.
        """
        super().__init__()

        # --- 1. Patching and Embedding ---
        num_patches = (img_size // patch_size) ** 2 # Calculate the number of patches (e.g., (32/4)^2 = 64).
        patch_dim = 3 * patch_size ** 2 # Calculate the dimension of a flattened patch (e.g., 3 * 4 * 4 = 48).

        # This is a linear layer that takes the flattened patch and projects it to the embedding dimension.
        self.patch_proj = nn.Linear(patch_dim, dim)

        # --- 2. Special Tokens and Positional Embeddings ---
        # The [CLS] token is a learnable parameter that we will use for classification.
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # The positional embedding gives the model information about the position of each patch.
        # It's a learnable parameter. We need one position for each patch + one for the [CLS] token.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_dropout = nn.Dropout(p=dropout)

        # --- 3. Transformer Encoder ---
        # We create a stack of 'depth' (e.g., 12) Transformer Blocks.
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, dropout=dropout, T=T, L=L)
            for _ in range(depth)
        ])

        # --- 4. Classification Head ---
        self.norm = nn.LayerNorm(dim) # Final normalization layer.
        self.head = nn.Linear(dim, num_classes) # The final linear layer that outputs the class logits.

    def forward(self, x):
        """The forward pass of the entire model."""
        # Input 'x' has shape [Batch, Channels, Height, Width], e.g., [128, 3, 32, 32].
        B, C, H, W = x.shape
        
        # --- 1. Patching ---
        # We need to reshape the image into a sequence of flattened patches.
        # Rearrange from [B, C, H, W] to [B, NumPatches, PatchDim].
        # The 'view' and 'permute' operations are efficient ways to do this without copying data.
        patches = x.view(B, C, H // 4, 4, W // 4, 4).permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 4) * (W // 4), C * 4 * 4)
        
        # Project the patches to the embedding dimension.
        x = self.patch_proj(patches) # Shape: [B, NumPatches, Dim], e.g., [128, 64, 192].

        # --- 2. Add Special Tokens and Positional Embeddings ---
        # Prepend the [CLS] token to the sequence of patch embeddings.
        cls_tokens = self.cls_token.expand(B, -1, -1) # Expand the single [CLS] token for the whole batch.
        x = torch.cat((cls_tokens, x), dim=1) # Shape: [B, NumPatches + 1, Dim], e.g., [128, 65, 192].

        # Add the positional embedding to the sequence.
        x = x + self.pos_embedding
        x = self.pos_dropout(x)

        # --- 3. Pass through the Transformer Encoder ---
        # The data flows through all the blocks in sequence.
        for blk in self.blocks:
            x = blk(x)

        # --- 4. Classification ---
        # We take only the output corresponding to the [CLS] token for classification.
        cls_token_output = x[:, 0] # Shape: [B, Dim], e.g., [128, 192].
        cls_token_output = self.norm(cls_token_output) # Apply final normalization.
        
        # Pass the [CLS] token output through the classification head to get the final logits.
        logits = self.head(cls_token_output) # Shape: [B, NumClasses], e.g., [128, 10].
        return logits

print("Model architecture defined.")