import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Original VectorQuantizer from the VQ-VAE paper.
    
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class ImprovedVectorQuantizer(nn.Module):
    """
    Improved VectorQuantizer with EMA updates and better codebook utilization.
    Optimized for 1D calcium imaging data.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, 
                 decay=0.99, epsilon=1e-5):
        super(ImprovedVectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        
        # EMA for codebook updates
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        # Track codebook usage
        self._codebook_usage = torch.zeros(num_embeddings)

    def forward(self, inputs):
        # Handle both 1D and 2D inputs
        if len(inputs.shape) == 3:  # 1D case: (B, C, L)
            inputs = inputs.permute(0, 2, 1).contiguous()  # -> (B, L, C)
            input_shape = inputs.shape
            flat_input = inputs.view(-1, self._embedding_dim)
        else:  # 2D case: (B, C, H, W)
            inputs = inputs.permute(0, 2, 3, 1).contiguous()  # -> (B, H, W, C)
            input_shape = inputs.shape
            flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, 
                               device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Update codebook usage
        if hasattr(self, '_codebook_usage'):
            self._codebook_usage = 0.99 * self._codebook_usage + 0.01 * encodings.sum(0).cpu()
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n
            )
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert back to original format
        if len(input_shape) == 3:  # 1D case
            quantized = quantized.permute(0, 2, 1).contiguous()  # -> (B, C, L)
        else:  # 2D case
            quantized = quantized.permute(0, 3, 1, 2).contiguous()  # -> (B, C, H, W)
        
        return loss, quantized, perplexity, encodings


class GroupedResidualVQ(nn.Module):
    """
    Grouped Residual Vector Quantization for better codebook utilization.
    Implements the technique from "Neural Discrete Representation Learning" improvements.
    """
    
    def __init__(self, num_embeddings, embedding_dim, num_groups=4, 
                 num_residual=2, commitment_cost=0.25):
        super(GroupedResidualVQ, self).__init__()
        
        self.num_groups = num_groups
        self.num_residual = num_residual
        self.group_dim = embedding_dim // num_groups
        
        assert embedding_dim % num_groups == 0, "embedding_dim must be divisible by num_groups"
        
        # Create quantizers for each group and residual level
        self.quantizers = nn.ModuleList([
            nn.ModuleList([
                ImprovedVectorQuantizer(
                    num_embeddings // num_groups,
                    self.group_dim,
                    commitment_cost
                ) for _ in range(num_groups)
            ]) for _ in range(num_residual)
        ])
        
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        
    def forward(self, inputs):
        # Handle both 1D and 2D inputs
        original_shape = inputs.shape
        if len(inputs.shape) == 3:  # 1D case: (B, C, L)
            inputs = inputs.permute(0, 2, 1).contiguous()  # -> (B, L, C)
            is_1d = True
        else:  # 2D case: (B, C, H, W)
            inputs = inputs.permute(0, 2, 3, 1).contiguous()  # -> (B, H, W, C)
            is_1d = False
            
        input_shape = inputs.shape
        B, *spatial_dims, D = input_shape
        
        # Reshape to (B * spatial, D)
        inputs_flat = inputs.view(-1, D)
        
        # Split into groups
        inputs_groups = inputs_flat.view(-1, self.num_groups, self.group_dim)
        
        all_losses = []
        all_perplexities = []
        quantized_groups = []
        
        for g in range(self.num_groups):
            x_g = inputs_groups[:, g, :]  # (B * spatial, group_dim)
            quantized_g = torch.zeros_like(x_g)
            
            # Residual quantization
            residual = x_g
            for r in range(self.num_residual):
                # Reshape for quantizer input
                if is_1d:
                    residual_input = residual.view(B, -1, self.group_dim).permute(0, 2, 1)
                else:
                    H, W = spatial_dims
                    residual_input = residual.view(B, H, W, self.group_dim).permute(0, 3, 1, 2)
                
                loss_r, quantized_r, perplexity_r, encodings_r = self.quantizers[r][g](residual_input)
                
                # Reshape back
                if is_1d:
                    quantized_r = quantized_r.permute(0, 2, 1).contiguous().view(-1, self.group_dim)
                else:
                    quantized_r = quantized_r.permute(0, 2, 3, 1).contiguous().view(-1, self.group_dim)
                
                quantized_g = quantized_g + quantized_r
                residual = residual - quantized_r.detach()
                
                all_losses.append(loss_r)
                all_perplexities.append(perplexity_r)
                
            quantized_groups.append(quantized_g)
        
        # Concatenate groups
        quantized = torch.cat(quantized_groups, dim=-1)  # (B * spatial, D)
        quantized = quantized.view(input_shape)  # Restore spatial dimensions
        
        # Average metrics
        loss = sum(all_losses) / len(all_losses)
        perplexity = sum(all_perplexities) / len(all_perplexities)
        
        # Dummy encodings for compatibility
        total_spatial = np.prod(spatial_dims) if spatial_dims else 1
        encodings = torch.zeros(B * total_spatial, self._num_embeddings, device=inputs.device)
        
        # Convert back to original format
        if is_1d:
            quantized = quantized.permute(0, 2, 1).contiguous()  # -> (B, C, L)
        else:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()  # -> (B, C, H, W)
        
        return loss, quantized, perplexity, encodings


if __name__ == "__main__":
    # Test original VQ
    print("Testing Original VectorQuantizer (2D):")
    x_2d = torch.randn(2, 64, 8, 8)
    vq_original = VectorQuantizer(512, 64, 0.25)
    loss, quantized, perplexity, encodings, indices = vq_original(x_2d)
    print(f"Input: {x_2d.shape}, Output: {quantized.shape}, Perplexity: {perplexity:.2f}")
    
    # Test improved VQ with 1D data
    print("\nTesting ImprovedVectorQuantizer (1D):")
    x_1d = torch.randn(2, 64, 50)  # (batch, channels, time)
    vq_improved = ImprovedVectorQuantizer(512, 64, 0.25)
    loss, quantized, perplexity, encodings = vq_improved(x_1d)
    print(f"Input: {x_1d.shape}, Output: {quantized.shape}, Perplexity: {perplexity:.2f}")
    
    # Test grouped RVQ
    print("\nTesting GroupedResidualVQ (1D):")
    grvq = GroupedResidualVQ(512, 64, num_groups=4, num_residual=2)
    loss, quantized, perplexity, encodings = grvq(x_1d)
    print(f"Input: {x_1d.shape}, Output: {quantized.shape}, Perplexity: {perplexity:.2f}")