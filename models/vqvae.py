import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.encoder import Encoder, CalciumEncoder
from models.quantizer import VectorQuantizer, ImprovedVectorQuantizer, GroupedResidualVQ
from models.decoder import Decoder
from models.behavior import BehaviorHead


class VQVAE(nn.Module):
    """
    Original VQ-VAE implementation for 2D images.
    """
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity


class CalciumDecoder(nn.Module):
    """
    1D Decoder optimized for calcium imaging data.
    Symmetric to CalciumEncoder with transpose convolutions.
    """

    def __init__(self, embedding_dim, num_hiddens, num_residual_layers, 
                 num_residual_hiddens, output_channels, dropout_rate=0.1):
        super(CalciumDecoder, self).__init__()
        
        from models.encoder import ImprovedResidualBlock
        
        # Residual stack at the beginning
        self._residual_stack = nn.ModuleList([
            ImprovedResidualBlock(embedding_dim, embedding_dim, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])
        
        # Progressive upsampling (symmetric to encoder)
        self._conv_transpose_1 = nn.ConvTranspose1d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        
        self._conv_transpose_2 = nn.ConvTranspose1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens//2,
            kernel_size=5, stride=2, padding=2, output_padding=1
        )
        
        self._conv_final = nn.Conv1d(
            in_channels=num_hiddens//2,
            out_channels=output_channels,
            kernel_size=7, stride=1, padding=3
        )
        
        self._dropout = nn.Dropout1d(dropout_rate)

    def forward(self, x):
        # Apply residual blocks
        for block in self._residual_stack:
            x = block(x)
        
        # Upsample
        x = F.relu(self._conv_transpose_1(x))
        x = self._dropout(x)
        x = F.relu(self._conv_transpose_2(x))
        x = self._dropout(x)
        x = self._conv_final(x)
        
        return x


class CalciumVQVAE(nn.Module):
    """
    VQ-VAE optimized for calcium imaging data with optional behavior prediction.
    
    Features:
    - 1D convolutions for temporal neural data
    - Choice of quantization methods (standard, improved, grouped residual)
    - Integrated behavior prediction head
    - Multi-task training capability
    """
    
    def __init__(self, num_neurons, num_hiddens, num_residual_layers, 
                 num_residual_hiddens, num_embeddings, embedding_dim, 
                 commitment_cost, quantizer_type='improved_vq',
                 dropout_rate=0.1, behavior_dim=4, enable_behavior_prediction=True):
        super(CalciumVQVAE, self).__init__()
        
        self.quantizer_type = quantizer_type
        self.enable_behavior_prediction = enable_behavior_prediction
        
        # Encoder
        self.encoder = CalciumEncoder(
            num_neurons, num_hiddens, num_residual_layers, 
            num_residual_hiddens, dropout_rate
        )
        
        # Pre-quantization convolution
        self.pre_quantization_conv = nn.Sequential(
            nn.Conv1d(in_channels=num_hiddens, 
                      out_channels=embedding_dim,
                      kernel_size=1, stride=1),
            nn.Dropout1d(dropout_rate)
        )
        
        # Vector Quantizer
        if quantizer_type == 'standard_vq':
            # Adapt standard VQ for 1D (requires modification to handle 1D input)
            self.vector_quantization = ImprovedVectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
        elif quantizer_type == 'improved_vq':
            self.vector_quantization = ImprovedVectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
        elif quantizer_type == 'grouped_rvq':
            self.vector_quantization = GroupedResidualVQ(
                num_embeddings, embedding_dim, commitment_cost=commitment_cost
            )
        else:
            raise ValueError(f"Unknown quantizer type: {quantizer_type}")
        
        # Decoder
        self.decoder = CalciumDecoder(
            embedding_dim, num_hiddens, num_residual_layers,
            num_residual_hiddens, num_neurons, dropout_rate
        )
        
        # Behavior prediction head (optional)
        if enable_behavior_prediction:
            self.behavior_head = BehaviorHead(
                embedding_dim, behavior_dim, hidden_dim=128, dropout=dropout_rate
            )
        else:
            self.behavior_head = None

    def forward(self, x, return_behavior_pred=True):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (B, num_neurons, time_steps)
            return_behavior_pred: Whether to compute behavior predictions
            
        Returns:
            tuple: (vq_loss, x_recon, perplexity, quantized, encodings, behavior_pred)
        """
        # Encode
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        
        # Quantize
        vq_loss, quantized, perplexity, encodings = self.vector_quantization(z)
        
        # Adjust dimensions for decoder if needed
        if quantized.shape[2] != x.shape[2] // 4:
            quantized = F.interpolate(
                quantized, size=x.shape[2] // 4, mode='linear', align_corners=False
            )
        
        # Decode
        x_recon = self.decoder(quantized)
        
        # Adjust final dimension if needed
        if x_recon.shape[2] != x.shape[2]:
            x_recon = F.interpolate(
                x_recon, size=x.shape[2], mode='linear', align_corners=False
            )

        # Behavior prediction
        behavior_pred = None
        if return_behavior_pred and self.behavior_head is not None:
            behavior_pred = self.behavior_head(quantized)

        return vq_loss, x_recon, perplexity, quantized, encodings, behavior_pred
    
    def encode(self, x):
        """Encode input to quantized representation."""
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        _, quantized, _, _ = self.vector_quantization(z)
        return quantized
    
    def decode(self, quantized):
        """Decode quantized representation to reconstruction."""
        return self.decoder(quantized)
    
    def get_codebook_usage(self):
        """Get codebook usage statistics."""
        if hasattr(self.vector_quantization, '_codebook_usage'):
            usage = self.vector_quantization._codebook_usage
            return {
                'used_codes': (usage > 0.01).sum().item(),
                'total_codes': len(usage),
                'usage_percentage': ((usage > 0.01).sum() / len(usage) * 100).item(),
                'entropy': -(usage * torch.log(usage + 1e-10)).sum().item()
            }
        else:
            return {'usage_percentage': 100.0}  # Assume full usage for grouped RVQ


class MultiScaleCalciumVQVAE(nn.Module):
    """
    Multi-scale VQ-VAE that operates at different temporal resolutions.
    Useful for capturing both fast neural dynamics and slow behavioral patterns.
    """
    
    def __init__(self, num_neurons, base_params, scales=[1, 2, 4]):
        super(MultiScaleCalciumVQVAE, self).__init__()
        
        self.scales = scales
        self.models = nn.ModuleList()
        
        for scale in scales:
            # Adjust parameters for each scale
            params = base_params.copy()
            params['num_embeddings'] = params['num_embeddings'] // scale
            params['embedding_dim'] = params['embedding_dim'] // scale
            
            model = CalciumVQVAE(num_neurons, **params)
            self.models.append(model)
    
    def forward(self, x):
        outputs = []
        
        for i, (model, scale) in enumerate(zip(self.models, self.scales)):
            # Downsample input for higher scales
            if scale > 1:
                x_scaled = F.avg_pool1d(x, kernel_size=scale, stride=scale)
            else:
                x_scaled = x
            
            # Forward pass
            vq_loss, x_recon, perplexity, quantized, encodings, behavior_pred = model(x_scaled)
            
            # Upsample reconstruction back to original size
            if scale > 1:
                x_recon = F.interpolate(x_recon, size=x.shape[2], mode='linear', align_corners=False)
            
            outputs.append({
                'vq_loss': vq_loss,
                'x_recon': x_recon,
                'perplexity': perplexity,
                'quantized': quantized,
                'behavior_pred': behavior_pred,
                'scale': scale
            })
        
        return outputs


def create_calcium_vqvae(config):
    """
    Factory function to create CalciumVQVAE with different configurations.
    
    Args:
        config: dict with model parameters
        
    Returns:
        CalciumVQVAE model
    """
    default_config = {
        'num_hiddens': 128,
        'num_residual_layers': 3,
        'num_residual_hiddens': 64,
        'num_embeddings': 512,
        'embedding_dim': 64,
        'commitment_cost': 0.25,
        'quantizer_type': 'improved_vq',
        'dropout_rate': 0.1,
        'behavior_dim': 4,
        'enable_behavior_prediction': True
    }
    
    # Update with user config
    default_config.update(config)
    
    return CalciumVQVAE(**default_config)


if __name__ == "__main__":
    # Test original VQVAE
    print("Testing Original VQVAE (2D):")
    model_2d = VQVAE(128, 32, 2, 512, 64, 0.25)
    x_2d = torch.randn(2, 3, 32, 32)
    vq_loss, x_recon, perplexity = model_2d(x_2d)
    print(f"Input: {x_2d.shape}, Reconstruction: {x_recon.shape}, Perplexity: {perplexity:.2f}")
    
    # Test CalciumVQVAE
    print("\nTesting CalciumVQVAE (1D):")
    model_1d = CalciumVQVAE(
        num_neurons=50, num_hiddens=128, num_residual_layers=3,
        num_residual_hiddens=64, num_embeddings=512, embedding_dim=64,
        commitment_cost=0.25, quantizer_type='improved_vq'
    )
    x_1d = torch.randn(2, 50, 100)  # (batch, neurons, time)
    vq_loss, x_recon, perplexity, quantized, encodings, behavior_pred = model_1d(x_1d)
    print(f"Input: {x_1d.shape}")
    print(f"Reconstruction: {x_recon.shape}")
    print(f"Quantized: {quantized.shape}")
    print(f"Behavior prediction: {behavior_pred.shape if behavior_pred is not None else None}")
    print(f"Perplexity: {perplexity:.2f}")
    
    # Test different quantizers
    print("\nTesting different quantizers:")
    for qt in ['improved_vq', 'grouped_rvq']:
        model = CalciumVQVAE(
            num_neurons=50, num_hiddens=64, num_residual_layers=2,
            num_residual_hiddens=32, num_embeddings=256, embedding_dim=32,
            commitment_cost=0.25, quantizer_type=qt
        )
        vq_loss, x_recon, perplexity, _, _, _ = model(x_1d)
        print(f"{qt}: Perplexity={perplexity:.2f}, Recon_MSE={F.mse_loss(x_recon, x_1d):.4f}")