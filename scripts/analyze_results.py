#!/usr/bin/env python3
"""
Script per analisi completa e visualizzazione dei risultati VQ-VAE calcium imaging.

Genera:
- Analisi spazio latente (t-SNE, UMAP, PCA)
- Analisi codebook 
- Curve di training dettagliate
- Esempi di ricostruzione
- Correlazioni behavior-neural
- Heatmaps di attivazione
- Distribuzione embeddings
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.vqvae import CalciumVQVAE
from datasets.calcium import create_calcium_dataloaders
from models.behavior import evaluate_behavior_predictions

# Try to import UMAP (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_model_and_data(model_path, experiment_name=None):
    """Load trained model and recreate dataloaders."""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model config
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # Default config
        config = {
            'quantizer_type': 'improved_vq',
            'enable_behavior_prediction': True
        }
    
    # Create dataloaders (same as training)
    train_loader, val_loader, test_loader, dataset_info = create_calcium_dataloaders(
        dataset_type='single',
        batch_size=32,
        session_id=501940850,  # Same session as training
        window_size=50,
        stride=10,
        min_neurons=30
    )
    
    dataloaders = (train_loader, val_loader, test_loader)
    
    # Create model
    num_neurons = dataset_info['neural_shape'][0]
    model = CalciumVQVAE(
        num_neurons=num_neurons,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        embedding_dim=64,
        commitment_cost=0.25,
        quantizer_type=config['quantizer_type'],
        enable_behavior_prediction=config['enable_behavior_prediction']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Dataset: {dataset_info['total_samples']} samples")
    
    return model, (train_loader, val_loader, test_loader), dataset_info, checkpoint


def extract_latent_representations(model, dataloader, max_samples=2000):
    """Extract latent representations and metadata."""
    print("Extracting latent representations...")
    
    device = next(model.parameters()).device
    
    latent_reps = []
    quantized_reps = []
    behavior_data = []
    reconstruction_errors = []
    original_data = []
    
    with torch.no_grad():
        for batch_idx, (neural_data, behavior) in enumerate(dataloader):
            if len(latent_reps) * neural_data.size(0) >= max_samples:
                break
                
            neural_data = neural_data.to(device)
            
            # Forward pass
            vq_loss, neural_recon, perplexity, quantized, encodings, behavior_pred = model(neural_data)
            
            # Extract features
            encoded = model.encoder(neural_data)
            pre_quantized = model.pre_quantization_conv(encoded)
            
            # Store representations (pool over time dimension)
            latent_mean = pre_quantized.mean(dim=2)  # [B, embedding_dim]
            quantized_mean = quantized.mean(dim=2)   # [B, embedding_dim]
            
            latent_reps.append(latent_mean.cpu().numpy())
            quantized_reps.append(quantized_mean.cpu().numpy())
            behavior_data.append(behavior.numpy())
            
            # Reconstruction error per sample
            recon_error = torch.mean((neural_recon - neural_data)**2, dim=(1,2))
            reconstruction_errors.append(recon_error.cpu().numpy())
            
            # Store some original data for visualization
            if len(original_data) < 10:
                original_data.append(neural_data.cpu().numpy())
    
    # Concatenate all
    latent_reps = np.vstack(latent_reps)
    quantized_reps = np.vstack(quantized_reps)
    behavior_data = np.vstack(behavior_data)
    reconstruction_errors = np.hstack(reconstruction_errors)
    
    print(f"✓ Extracted {latent_reps.shape[0]} samples")
    print(f"✓ Latent space dimension: {latent_reps.shape[1]}")
    
    return {
        'latent': latent_reps,
        'quantized': quantized_reps,
        'behavior': behavior_data,
        'recon_errors': reconstruction_errors,
        'original_samples': original_data[:5]  # Keep first 5 for visualization
    }


def analyze_latent_space(representations, save_dir, experiment_name):
    """Comprehensive latent space analysis."""
    print("Analyzing latent space...")
    
    latent = representations['latent']
    quantized = representations['quantized']
    behavior = representations['behavior']
    recon_errors = representations['recon_errors']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. PCA Analysis
    print("  Running PCA...")
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent)
    quantized_pca = pca.fit_transform(quantized)
    
    # 2. t-SNE Analysis  
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    latent_tsne = tsne.fit_transform(latent[:1000])  # Subset for speed
    
    # 3. UMAP Analysis (if available)
    if UMAP_AVAILABLE:
        print("  Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        latent_umap = reducer.fit_transform(latent)
    
    # Color by different behavioral features
    behavior_features = ['Mean Speed', 'Speed Std', 'Max Speed', 'Speed Change']
    
    # Plot 1: PCA colored by behavior
    ax1 = plt.subplot(3, 4, 1)
    scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], 
                         c=behavior[:, 0], cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Mean Speed')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    plt.title('PCA - Latent Space (Mean Speed)')
    
    # Plot 2: PCA colored by reconstruction error
    ax2 = plt.subplot(3, 4, 2)
    scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], 
                         c=recon_errors, cmap='plasma', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Reconstruction Error')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    plt.title('PCA - Colored by Recon Error')
    
    # Plot 3: t-SNE colored by behavior
    ax3 = plt.subplot(3, 4, 3)
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                         c=behavior[:1000, 0], cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Mean Speed')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE - Latent Space')
    
    # Plot 4: UMAP (if available)
    if UMAP_AVAILABLE:
        ax4 = plt.subplot(3, 4, 4)
        scatter = plt.scatter(latent_umap[:, 0], latent_umap[:, 1], 
                             c=behavior[:, 0], cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Mean Speed')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('UMAP - Latent Space')
    else:
        ax4 = plt.subplot(3, 4, 4)
        plt.text(0.5, 0.5, 'UMAP not available\nInstall: pip install umap-learn', 
                ha='center', va='center', transform=ax4.transAxes)
        plt.title('UMAP - Not Available')
    
    # Plot 5-8: PCA colored by different behavioral features
    for i, feature_name in enumerate(behavior_features):
        ax = plt.subplot(3, 4, 5 + i)
        scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], 
                             c=behavior[:, i], cmap='coolwarm', alpha=0.6, s=15)
        plt.colorbar(scatter, label=feature_name)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'PCA - {feature_name}')
    
    # Plot 9: Explained variance
    ax9 = plt.subplot(3, 4, 9)
    pca_full = PCA()
    pca_full.fit(latent)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, min(21, len(cumvar)+1)), cumvar[:20], 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True, alpha=0.3)
    
    # Plot 10: Latent space clustering
    ax10 = plt.subplot(3, 4, 10)
    # K-means clustering on latent space
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(latent_pca)
    scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], 
                         c=clusters, cmap='tab10', alpha=0.6, s=20)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('K-means Clustering (k=5)')
    
    # Plot 11: Behavior correlation heatmap
    ax11 = plt.subplot(3, 4, 11)
    # Correlate latent PCs with behavior
    corr_matrix = np.zeros((min(10, latent.shape[1]), 4))
    pca_components = PCA(n_components=10).fit_transform(latent)
    for i in range(min(10, latent.shape[1])):
        for j in range(4):
            corr_matrix[i, j] = pearsonr(pca_components[:, i], behavior[:, j])[0]
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=behavior_features, 
                yticklabels=[f'PC{i+1}' for i in range(corr_matrix.shape[0])])
    plt.title('PC-Behavior Correlations')
    
    # Plot 12: Latent space density
    ax12 = plt.subplot(3, 4, 12)
    plt.hist2d(latent_pca[:, 0], latent_pca[:, 1], bins=50, cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Latent Space Density')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{experiment_name}_latent_space_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Latent space analysis saved: {save_path}")
    
    return {
        'pca_variance': pca.explained_variance_ratio_,
        'latent_pca': latent_pca,
        'quantized_pca': quantized_pca,
        'pc_behavior_correlations': corr_matrix
    }


def analyze_codebook(model, representations, save_dir, experiment_name):
    """Analyze VQ-VAE codebook usage and embeddings."""
    print("Analyzing codebook...")
    
    device = next(model.parameters()).device
    
    # Get codebook embeddings
    if hasattr(model.vector_quantization, '_embedding'):
        embeddings = model.vector_quantization._embedding.weight.detach().cpu().numpy()
    elif hasattr(model.vector_quantization, 'embedding'):
        embeddings = model.vector_quantization.embedding.weight.detach().cpu().numpy()
    else:
        print("Warning: Could not extract codebook embeddings")
        return
    
    # Get usage statistics
    usage_stats = model.get_codebook_usage()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Codebook embedding PCA
    pca = PCA(n_components=2)
    embed_pca = pca.fit_transform(embeddings)
    
    axes[0, 0].scatter(embed_pca[:, 0], embed_pca[:, 1], alpha=0.7, s=30)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    axes[0, 0].set_title('Codebook Embeddings (PCA)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Embedding similarity heatmap (subset)
    n_show = min(50, len(embeddings))
    similarity_matrix = np.corrcoef(embeddings[:n_show])
    
    im = axes[0, 1].imshow(similarity_matrix, cmap='coolwarm', aspect='auto')
    axes[0, 1].set_title(f'Embedding Similarity (first {n_show})')
    axes[0, 1].set_xlabel('Embedding Index')
    axes[0, 1].set_ylabel('Embedding Index')
    plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Embedding norm distribution
    norms = np.linalg.norm(embeddings, axis=1)
    axes[0, 2].hist(norms, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Embedding Norm')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Embedding Norm Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Codebook usage (if available)
    if hasattr(model.vector_quantization, '_codebook_usage'):
        usage = model.vector_quantization._codebook_usage.cpu().numpy()
        
        axes[1, 0].bar(range(len(usage)), usage, alpha=0.7)
        axes[1, 0].set_xlabel('Codebook Index')
        axes[1, 0].set_ylabel('Usage Count')
        axes[1, 0].set_title('Codebook Usage Distribution')
        
        # Usage histogram
        axes[1, 1].hist(usage, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Usage Count')
        axes[1, 1].set_ylabel('Number of Codes')
        axes[1, 1].set_title('Usage Count Distribution')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Usage statistics\nnot available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Codebook Usage')
        
        axes[1, 1].text(0.5, 0.5, 'Usage histogram\nnot available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Usage Distribution')
    
    # 5. Embedding clustering
    from sklearn.cluster import KMeans
    n_clusters = min(10, len(embeddings) // 10)
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        scatter = axes[1, 2].scatter(embed_pca[:, 0], embed_pca[:, 1], 
                                   c=clusters, cmap='tab10', alpha=0.7, s=30)
        axes[1, 2].set_xlabel('PC1')
        axes[1, 2].set_ylabel('PC2')
        axes[1, 2].set_title(f'Embedding Clusters (k={n_clusters})')
    else:
        axes[1, 2].text(0.5, 0.5, 'Not enough embeddings\nfor clustering', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Embedding Clustering')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{experiment_name}_codebook_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Codebook analysis saved: {save_path}")


def create_detailed_training_curves(checkpoint, save_dir, experiment_name):
    """Create detailed training curves from checkpoint."""
    print("Creating detailed training curves...")
    
    if 'results' not in checkpoint:
        print("Warning: No training metrics found in checkpoint")
        return
    
    results = checkpoint['results']
    metrics = results.get('training_metrics', {})
    
    if not metrics:
        print("Warning: No training metrics available")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    epochs = range(1, len(metrics.get('train_total_loss', [])) + 1)
    
    # 1. Total Loss
    if 'train_total_loss' in metrics and 'val_total_loss' in metrics:
        axes[0, 0].plot(epochs, metrics['train_total_loss'], 'b-', label='Train', alpha=0.8)
        axes[0, 0].plot(epochs, metrics['val_total_loss'], 'r-', label='Validation', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
    
    # 2. Reconstruction Loss
    if 'train_recon_loss' in metrics and 'val_recon_loss' in metrics:
        axes[0, 1].plot(epochs, metrics['train_recon_loss'], 'b-', label='Train', alpha=0.8)
        axes[0, 1].plot(epochs, metrics['val_recon_loss'], 'r-', label='Validation', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Reconstruction Loss')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. VQ Loss
    if 'train_vq_loss' in metrics and 'val_vq_loss' in metrics:
        axes[0, 2].plot(epochs, metrics['train_vq_loss'], 'b-', label='Train', alpha=0.8)
        axes[0, 2].plot(epochs, metrics['val_vq_loss'], 'r-', label='Validation', alpha=0.8)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('VQ Loss')
        axes[0, 2].set_title('Vector Quantization Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Perplexity
    if 'train_perplexity' in metrics and 'val_perplexity' in metrics:
        axes[1, 0].plot(epochs, metrics['train_perplexity'], 'b-', label='Train', alpha=0.8)
        axes[1, 0].plot(epochs, metrics['val_perplexity'], 'r-', label='Validation', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].set_title('Codebook Perplexity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Behavior Loss
    if 'train_behavior_loss' in metrics and 'val_behavior_loss' in metrics:
        # Filter out zeros
        train_behav = [x for x in metrics['train_behavior_loss'] if x > 0]
        val_behav = [x for x in metrics['val_behavior_loss'] if x > 0]
        behav_epochs = epochs[:len(train_behav)]
        
        if train_behav and val_behav:
            axes[1, 1].plot(behav_epochs, train_behav, 'b-', label='Train', alpha=0.8)
            axes[1, 1].plot(behav_epochs, val_behav, 'r-', label='Validation', alpha=0.8)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Behavior Loss')
            axes[1, 1].set_title('Behavior Prediction Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
    
    # 6. Behavior R² (if available)
    if 'val_behavior_r2' in metrics:
        r2_scores = [x for x in metrics['val_behavior_r2'] if x is not None]
        if r2_scores:
            r2_epochs = epochs[:len(r2_scores)]
            axes[1, 2].plot(r2_epochs, r2_scores, 'g-', linewidth=2, alpha=0.8)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Behavior R²')
            axes[1, 2].set_title('Behavior Prediction R²')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 7. Loss components comparison
    if all(key in metrics for key in ['val_recon_loss', 'val_vq_loss', 'val_behavior_loss']):
        val_recon = metrics['val_recon_loss']
        val_vq = metrics['val_vq_loss']
        val_behav = [x for x in metrics['val_behavior_loss']]
        
        # Normalize for comparison
        val_recon_norm = np.array(val_recon) / np.max(val_recon)
        val_vq_norm = np.array(val_vq) / np.max(val_vq)
        val_behav_norm = np.array(val_behav) / np.max([x for x in val_behav if x > 0])
        
        axes[2, 0].plot(epochs, val_recon_norm, label='Reconstruction', alpha=0.8)
        axes[2, 0].plot(epochs, val_vq_norm, label='VQ Loss', alpha=0.8)
        axes[2, 0].plot(epochs[:len(val_behav_norm)], val_behav_norm, label='Behavior', alpha=0.8)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Normalized Loss')
        axes[2, 0].set_title('Loss Components (Normalized)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Convergence analysis
    if 'val_total_loss' in metrics:
        val_loss = metrics['val_total_loss']
        # Calculate moving average
        window = min(10, len(val_loss) // 4)
        if window > 1:
            moving_avg = np.convolve(val_loss, np.ones(window)/window, mode='valid')
            axes[2, 1].plot(epochs, val_loss, alpha=0.3, color='blue', label='Raw')
            axes[2, 1].plot(epochs[window-1:], moving_avg, 'b-', linewidth=2, label=f'Moving Avg ({window})')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Validation Loss')
            axes[2, 1].set_title('Convergence Analysis')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].set_yscale('log')
    
    # 9. Learning rate (if available)
    if 'learning_rate' in metrics:
        lr_data = metrics['learning_rate']
        if lr_data:
            axes[2, 2].plot(epochs[:len(lr_data)], lr_data, 'g-', linewidth=2)
            axes[2, 2].set_xlabel('Epoch')
            axes[2, 2].set_ylabel('Learning Rate')
            axes[2, 2].set_title('Learning Rate Schedule')
            axes[2, 2].grid(True, alpha=0.3)
            axes[2, 2].set_yscale('log')
    else:
        axes[2, 2].text(0.5, 0.5, 'Learning rate\ndata not available', 
                       ha='center', va='center', transform=axes[2, 2].transAxes)
        axes[2, 2].set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{experiment_name}_detailed_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Detailed training curves saved: {save_path}")


def create_reconstruction_examples(model, representations, save_dir, experiment_name):
    """Create detailed reconstruction examples."""
    print("Creating reconstruction examples...")
    
    original_samples = representations['original_samples']
    device = next(model.parameters()).device
    
    fig, axes = plt.subplots(len(original_samples), 3, figsize=(15, 3*len(original_samples)))
    if len(original_samples) == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, sample in enumerate(original_samples):
            # Convert to tensor and reconstruct
            sample_tensor = torch.FloatTensor(sample).to(device)
            vq_loss, reconstruction, perplexity, quantized, _, _ = model(sample_tensor)
            
            # Convert back to numpy
            reconstruction = reconstruction.cpu().numpy()
            
            # Select most active neurons for visualization
            sample_batch = sample[0]  # First sample in batch
            recon_batch = reconstruction[0]
            
            neuron_activity = np.var(sample_batch, axis=1)
            top_neurons = np.argsort(neuron_activity)[-15:]  # Top 15 active neurons
            
            # Original
            im1 = axes[i, 0].imshow(sample_batch[top_neurons, :], 
                                   aspect='auto', cmap='viridis', interpolation='nearest')
            axes[i, 0].set_ylabel(f'Sample {i+1}\nNeurons')
            axes[i, 0].set_xlabel('Time')
            if i == 0:
                axes[i, 0].set_title('Original')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Reconstruction
            im2 = axes[i, 1].imshow(recon_batch[top_neurons, :], 
                                   aspect='auto', cmap='viridis', interpolation='nearest')
            axes[i, 1].set_xlabel('Time')
            if i == 0:
                axes[i, 1].set_title('Reconstruction')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # Difference
            diff = np.abs(sample_batch[top_neurons, :] - recon_batch[top_neurons, :])
            im3 = axes[i, 2].imshow(diff, aspect='auto', cmap='Reds', interpolation='nearest')
            axes[i, 2].set_xlabel('Time')
            if i == 0:
                axes[i, 2].set_title('|Difference|')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            # Add MSE as text
            mse = np.mean((sample_batch - recon_batch)**2)
            axes[i, 1].text(0.02, 0.98, f'MSE: {mse:.4f}', 
                           transform=axes[i, 1].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           verticalalignment='top')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{experiment_name}_detailed_reconstructions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Detailed reconstructions saved: {save_path}")


def create_behavior_correlation_analysis(representations, save_dir, experiment_name):
    """Analyze correlations between neural representations and behavior."""
    print("Analyzing behavior correlations...")
    
    latent = representations['latent']
    behavior = representations['behavior']
    
    feature_names = ['Mean Speed', 'Speed Std', 'Max Speed', 'Speed Change']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Correlation matrix heatmap
    corr_matrix = np.corrcoef(np.hstack([latent[:, :10], behavior]).T)  # First 10 latent dims + 4 behavior
    
    # Split correlation matrix
    latent_behavior_corr = corr_matrix[:10, 10:]  # Latent vs behavior correlations
    
    im = axes[0, 0].imshow(latent_behavior_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[0, 0].set_xticks(range(4))
    axes[0, 0].set_xticklabels(feature_names, rotation=45)
    axes[0, 0].set_yticks(range(10))
    axes[0, 0].set_yticklabels([f'Latent {i+1}' for i in range(10)])
    axes[0, 0].set_title('Latent-Behavior Correlations')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Add correlation values as text
    for i in range(10):
        for j in range(4):
            axes[0, 0].text(j, i, f'{latent_behavior_corr[i, j]:.2f}', 
                           ha='center', va='center', 
                           color='white' if abs(latent_behavior_corr[i, j]) > 0.5 else 'black')
    
    # 2. Scatter plots for highest correlations
    # Find strongest correlations
    max_corr_idx = np.unravel_index(np.argmax(np.abs(latent_behavior_corr)), latent_behavior_corr.shape)
    best_latent_dim = max_corr_idx[0]
    best_behavior_dim = max_corr_idx[1]
    
    axes[0, 1].scatter(latent[:, best_latent_dim], behavior[:, best_behavior_dim], 
                      alpha=0.6, s=20)
    axes[0, 1].set_xlabel(f'Latent Dimension {best_latent_dim+1}')
    axes[0, 1].set_ylabel(feature_names[best_behavior_dim])
    axes[0, 1].set_title(f'Best Correlation (r={latent_behavior_corr[best_latent_dim, best_behavior_dim]:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(latent[:, best_latent_dim], behavior[:, best_behavior_dim], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(latent[:, best_latent_dim], p(latent[:, best_latent_dim]), "r--", alpha=0.8)
    
    # 3. Distribution of correlations
    all_correlations = []
    for i in range(min(20, latent.shape[1])):
        for j in range(4):
            corr, _ = pearsonr(latent[:, i], behavior[:, j])
            all_correlations.append(corr)
    
    axes[0, 2].hist(all_correlations, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 2].set_xlabel('Correlation Coefficient')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Latent-Behavior Correlations')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4-6. Individual behavior feature analysis
    for i, feature_name in enumerate(feature_names[:3]):
        ax = axes[1, i]
        
        # Find best latent dimension for this behavior
        feature_correlations = [pearsonr(latent[:, j], behavior[:, i])[0] 
                               for j in range(min(20, latent.shape[1]))]
        best_dim = np.argmax(np.abs(feature_correlations))
        best_corr = feature_correlations[best_dim]
        
        # Scatter plot
        scatter = ax.scatter(latent[:, best_dim], behavior[:, i], 
                           c=behavior[:, i], cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel(f'Latent Dimension {best_dim+1}')
        ax.set_ylabel(feature_name)
        ax.set_title(f'{feature_name} vs Best Latent (r={best_corr:.3f})')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
        
        # Add trend line
        z = np.polyfit(latent[:, best_dim], behavior[:, i], 1)
        p = np.poly1d(z)
        ax.plot(latent[:, best_dim], p(latent[:, best_dim]), "r--", alpha=0.8)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{experiment_name}_behavior_correlations.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Behavior correlation analysis saved: {save_path}")


def create_comprehensive_summary(representations, latent_analysis, save_dir, experiment_name, results):
    """Create a comprehensive summary dashboard."""
    print("Creating comprehensive summary...")
    
    fig = plt.figure(figsize=(24, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.3)
    
    # 1. Model performance summary (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    if 'final_val_metrics' in results:
        metrics = results['final_val_metrics']
        metric_names = ['Recon MSE', 'VQ Loss', 'Perplexity', 'Behavior R²']
        metric_values = [
            metrics.get('recon_loss', 0),
            metrics.get('vq_loss', 0), 
            metrics.get('perplexity', 0),
            metrics.get('behavior_evaluation', {}).get('overall', {}).get('mean_r2', 0)
        ]
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.8)
        ax1.set_title('Model Performance Summary', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Latent space visualization (top middle)
    ax2 = fig.add_subplot(gs[0, 2:4])
    latent_pca = latent_analysis['latent_pca']
    behavior = representations['behavior']
    scatter = ax2.scatter(latent_pca[:, 0], latent_pca[:, 1], 
                         c=behavior[:, 0], cmap='viridis', alpha=0.6, s=15)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Latent Space (PCA)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Mean Speed')
    
    # 3. Codebook usage (top right)
    ax3 = fig.add_subplot(gs[0, 4:])
    usage_stats = results.get('codebook_usage', {})
    usage_pct = usage_stats.get('usage_percentage', 0)
    
    # Pie chart of codebook usage
    sizes = [usage_pct, 100 - usage_pct]
    labels = ['Used', 'Unused']
    colors = ['lightgreen', 'lightcoral']
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title(f'Codebook Usage\n({usage_pct:.1f}% used)', fontsize=14, fontweight='bold')
    
    # 4. Training progress (second row, left)
    ax4 = fig.add_subplot(gs[1, :3])
    if 'training_metrics' in results:
        metrics = results['training_metrics']
        epochs = range(1, len(metrics.get('val_total_loss', [])) + 1)
        
        if 'val_total_loss' in metrics:
            ax4.plot(epochs, metrics['val_total_loss'], 'b-', linewidth=2, label='Validation Loss')
        if 'val_behavior_r2' in metrics:
            # Use secondary y-axis for R²
            ax4_twin = ax4.twinx()
            r2_data = [x for x in metrics['val_behavior_r2'] if x is not None]
            if r2_data:
                ax4_twin.plot(epochs[:len(r2_data)], r2_data, 'r-', linewidth=2, label='Behavior R²')
                ax4_twin.set_ylabel('Behavior R²', color='red')
                ax4_twin.tick_params(axis='y', labelcolor='red')
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Loss', color='blue')
        ax4.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    
    # 5. Behavior prediction accuracy (second row, right)
    ax5 = fig.add_subplot(gs[1, 3:])
    if 'final_val_metrics' in results and 'behavior_evaluation' in results['final_val_metrics']:
        behavior_eval = results['final_val_metrics']['behavior_evaluation']
        feature_names = ['Mean Speed', 'Speed Std', 'Max Speed', 'Speed Change']
        r2_scores = []
        
        for feature in feature_names:
            if feature in behavior_eval:
                r2_scores.append(behavior_eval[feature]['r2'])
            else:
                r2_scores.append(0)
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        bars = ax5.barh(feature_names, r2_scores, color=colors, alpha=0.8)
        ax5.set_xlabel('R² Score')
        ax5.set_title('Behavior Prediction Accuracy', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([min(0, min(r2_scores)-0.1), max(1, max(r2_scores)+0.1)])
        
        # Add value labels
        for bar, value in zip(bars, r2_scores):
            width = bar.get_width()
            ax5.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2.,
                    f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center', fontweight='bold')
    
    # 6. Reconstruction examples (third row)
    original_samples = representations['original_samples']
    if original_samples:
        # Show first reconstruction example
        sample = original_samples[0][0]  # First sample, first in batch
        
        # Original
        ax6 = fig.add_subplot(gs[2, :2])
        neuron_activity = np.var(sample, axis=1)
        top_neurons = np.argsort(neuron_activity)[-10:]
        
        im = ax6.imshow(sample[top_neurons, :], aspect='auto', cmap='viridis', interpolation='nearest')
        ax6.set_title('Original Neural Activity', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Neurons (top 10)')
        plt.colorbar(im, ax=ax6)
        
        # Reconstruction (we'll need to compute this)
        ax7 = fig.add_subplot(gs[2, 2:4])
        ax7.text(0.5, 0.5, 'Reconstruction\n(computed during\nanalysis)', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title('Reconstructed Activity', fontsize=14, fontweight='bold')
        
        # Difference
        ax8 = fig.add_subplot(gs[2, 4:])
        ax8.text(0.5, 0.5, 'Reconstruction\nError\n(computed during\nanalysis)', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=12)
        ax8.set_title('Reconstruction Error', fontsize=14, fontweight='bold')
    
    # 7. Feature correlation summary (bottom row)
    ax9 = fig.add_subplot(gs[3, :])
    
    # Create a summary table of key results
    summary_data = []
    if 'final_val_metrics' in results:
        final_metrics = results['final_val_metrics']
        summary_data.append(['Reconstruction MSE', f"{final_metrics.get('recon_loss', 0):.4f}"])
        summary_data.append(['VQ Loss', f"{final_metrics.get('vq_loss', 0):.4f}"])
        summary_data.append(['Perplexity', f"{final_metrics.get('perplexity', 0):.1f}"])
        
        if 'behavior_evaluation' in final_metrics:
            behavior_eval = final_metrics['behavior_evaluation']
            if 'overall' in behavior_eval:
                summary_data.append(['Mean Behavior R²', f"{behavior_eval['overall']['mean_r2']:.3f}"])
                summary_data.append(['Best Behavior R²', f"{behavior_eval['overall']['best_r2']:.3f}"])
    
    summary_data.append(['Codebook Usage', f"{usage_stats.get('usage_percentage', 0):.1f}%"])
    summary_data.append(['Best Epoch', f"{results.get('best_epoch', 0)}"])
    summary_data.append(['Dataset Size', f"{len(representations['latent'])} samples"])
    
    # Create table
    table_data = [['Metric', 'Value']] + summary_data
    
    ax9.axis('tight')
    ax9.axis('off')
    table = ax9.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0.1, 0.2, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax9.set_title('Experiment Summary', fontsize=16, fontweight='bold', pad=20)
    
    # Add overall title
    fig.suptitle(f'VQ-VAE Calcium Imaging Analysis: {experiment_name}', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Save plot
    save_path = os.path.join(save_dir, f'{experiment_name}_comprehensive_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comprehensive summary saved: {save_path}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze VQ-VAE calcium imaging results")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model checkpoint')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for output files')
    parser.add_argument('--save_dir', type=str, default='./analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--max_samples', type=int, default=2000,
                       help='Maximum samples for analysis (for speed)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get experiment name from model path if not provided
    if args.experiment_name is None:
        args.experiment_name = Path(args.model_path).stem
    
    print(f"🔬 VQ-VAE Calcium Imaging Analysis")
    print(f"Model: {args.model_path}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 60)
    
    try:
        # Load model and data
        model, dataloaders, dataset_info, checkpoint = load_model_and_data(
            args.model_path, args.experiment_name
        )
        train_loader, val_loader, test_loader = dataloaders
        
        # Extract representations
        representations = extract_latent_representations(
            model, val_loader, max_samples=args.max_samples
        )
        
        # Run analyses
        print("\n📊 Running comprehensive analysis...")
        
        # 1. Latent space analysis
        latent_analysis = analyze_latent_space(
            representations, args.save_dir, args.experiment_name
        )
        
        # 2. Codebook analysis
        analyze_codebook(
            model, representations, args.save_dir, args.experiment_name
        )
        
        # 3. Detailed training curves
        create_detailed_training_curves(
            checkpoint, args.save_dir, args.experiment_name
        )
        
        # 4. Reconstruction examples
        create_reconstruction_examples(
            model, representations, args.save_dir, args.experiment_name
        )
        
        # 5. Behavior correlation analysis
        create_behavior_correlation_analysis(
            representations, args.save_dir, args.experiment_name
        )
        
        # 6. Comprehensive summary
        results = checkpoint.get('results', {})
        create_comprehensive_summary(
            representations, latent_analysis, args.save_dir, args.experiment_name, results
        )
        
        print("\n🎉 Analysis completed successfully!")
        print(f"\nGenerated files in {args.save_dir}:")
        print(f"  • {args.experiment_name}_latent_space_analysis.png")
        print(f"  • {args.experiment_name}_codebook_analysis.png") 
        print(f"  • {args.experiment_name}_detailed_training_curves.png")
        print(f"  • {args.experiment_name}_detailed_reconstructions.png")
        print(f"  • {args.experiment_name}_behavior_correlations.png")
        print(f"  • {args.experiment_name}_comprehensive_summary.png")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)