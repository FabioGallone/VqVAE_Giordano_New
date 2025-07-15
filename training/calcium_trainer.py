import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
import os
import time
from collections import defaultdict
import matplotlib.pyplot as plt

from models.behavior import BehaviorLoss, evaluate_behavior_predictions


class CalciumTrainer:
    """
    Specialized trainer for calcium imaging VQ-VAE models.
    
    Features:
    - Multi-task training (reconstruction + behavior)
    - Advanced learning rate scheduling
    - Early stopping with patience
    - Comprehensive metrics tracking
    - Automatic model checkpointing
    - TensorBoard logging
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader=None,
                 device='cuda', save_dir='./results', experiment_name=None):
        """
        Args:
            model: CalciumVQVAE model
            train_loader: training dataloader
            val_loader: validation dataloader  
            test_loader: test dataloader (optional)
            device: device for training
            save_dir: directory to save results
            experiment_name: name for the experiment
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Setup directories
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"calcium_vqvae_{int(time.time())}"
        self.experiment_name = experiment_name
        
        # Initialize tracking
        self.metrics = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Setup logging
        self.writer = SummaryWriter(f'{save_dir}/logs/{experiment_name}')
        
        print(f"Trainer initialized for experiment: {experiment_name}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Device: {device}")
    
    def train(self, num_epochs=100, learning_rate=3e-4, behavior_weight=0.5,
              patience=20, gradient_clip=1.0, save_best=True, eval_interval=5):
        """
        Main training loop.
        
        Args:
            num_epochs: maximum number of epochs
            learning_rate: initial learning rate
            behavior_weight: weight for behavior prediction loss
            patience: early stopping patience
            gradient_clip: gradient clipping value
            save_best: whether to save best model
            eval_interval: interval for detailed evaluation
        
        Returns:
            dict: training results and metrics
        """
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Create scheduler with version compatibility
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
            )
        except TypeError:
            # Fallback for older PyTorch versions without verbose parameter
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=patience//2
            )
            print("Note: Using ReduceLROnPlateau without verbose (older PyTorch version)")
        
        # Setup loss functions
        reconstruction_loss = nn.MSELoss()
        if self.model.enable_behavior_prediction:
            behavior_loss_fn = BehaviorLoss()
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Behavior weight: {behavior_weight}")
        print(f"Learning rate: {learning_rate}")
        print(f"Patience: {patience}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(
                self.optimizer, reconstruction_loss, behavior_loss_fn if self.model.enable_behavior_prediction else None,
                behavior_weight, gradient_clip
            )
            
            # Validation phase
            val_metrics = self._validate_epoch(
                reconstruction_loss, behavior_loss_fn if self.model.enable_behavior_prediction else None,
                behavior_weight
            )
            
            # Learning rate scheduling
            scheduler.step(val_metrics['total_loss'])
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Early stopping check
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                if save_best:
                    self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Detailed evaluation
            if (epoch + 1) % eval_interval == 0:
                self._detailed_evaluation(epoch)
            
            # Progress reporting
            if (epoch + 1) % 10 == 0 or epoch == 0:
                epoch_time = time.time() - epoch_start
                total_time = time.time() - start_time
                self._print_progress(epoch, train_metrics, val_metrics, epoch_time, total_time)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch + 1}")
                break
        
        # Final evaluation
        print("\nTraining completed. Running final evaluation...")
        final_results = self._final_evaluation()
        
        # Save final results
        self._save_results(final_results)
        
        return final_results
    
    def _train_epoch(self, optimizer, recon_loss_fn, behavior_loss_fn, behavior_weight, gradient_clip):
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = {
            'recon_loss': 0.0,
            'vq_loss': 0.0,
            'behavior_loss': 0.0,
            'total_loss': 0.0,
            'perplexity': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, (neural_data, behavior_data) in enumerate(self.train_loader):
            neural_data = neural_data.to(self.device)
            behavior_data = behavior_data.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            vq_loss, neural_recon, perplexity, quantized, _, behavior_pred = self.model(neural_data)
            
            # Reconstruction loss
            recon_loss = recon_loss_fn(neural_recon, neural_data)
            
            # Behavior loss
            behavior_loss = 0.0
            if self.model.enable_behavior_prediction and behavior_loss_fn is not None:
                behavior_loss = behavior_loss_fn(behavior_pred, behavior_data)
            
            # Total loss
            total_loss = recon_loss + 0.25 * vq_loss + behavior_weight * behavior_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            optimizer.step()
            
            # Update metrics
            epoch_metrics['recon_loss'] += recon_loss.item()
            epoch_metrics['vq_loss'] += vq_loss.item()
            epoch_metrics['behavior_loss'] += behavior_loss.item() if isinstance(behavior_loss, torch.Tensor) else behavior_loss
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['perplexity'] += perplexity.item()
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _validate_epoch(self, recon_loss_fn, behavior_loss_fn, behavior_weight):
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_metrics = {
            'recon_loss': 0.0,
            'vq_loss': 0.0,
            'behavior_loss': 0.0,
            'total_loss': 0.0,
            'perplexity': 0.0
        }
        
        num_batches = len(self.val_loader)
        all_behavior_preds = []
        all_behavior_targets = []
        
        with torch.no_grad():
            for neural_data, behavior_data in self.val_loader:
                neural_data = neural_data.to(self.device)
                behavior_data = behavior_data.to(self.device)
                
                # Forward pass
                vq_loss, neural_recon, perplexity, _, _, behavior_pred = self.model(neural_data)
                
                # Losses
                recon_loss = recon_loss_fn(neural_recon, neural_data)
                
                behavior_loss = 0.0
                if self.model.enable_behavior_prediction and behavior_loss_fn is not None:
                    behavior_loss = behavior_loss_fn(behavior_pred, behavior_data)
                    all_behavior_preds.append(behavior_pred.cpu())
                    all_behavior_targets.append(behavior_data.cpu())
                
                total_loss = recon_loss + 0.25 * vq_loss + behavior_weight * behavior_loss
                
                # Update metrics
                epoch_metrics['recon_loss'] += recon_loss.item()
                epoch_metrics['vq_loss'] += vq_loss.item()
                epoch_metrics['behavior_loss'] += behavior_loss.item() if isinstance(behavior_loss, torch.Tensor) else behavior_loss
                epoch_metrics['total_loss'] += total_loss.item()
                epoch_metrics['perplexity'] += perplexity.item()
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Behavior R² score
        if all_behavior_preds:
            all_preds = torch.cat(all_behavior_preds).numpy()
            all_targets = torch.cat(all_behavior_targets).numpy()
            
            valid_mask = np.isfinite(all_targets[:, 0]) & np.isfinite(all_preds[:, 0])
            if valid_mask.sum() > 0:
                epoch_metrics['behavior_r2'] = r2_score(
                    all_targets[valid_mask, 0], all_preds[valid_mask, 0]
                )
            else:
                epoch_metrics['behavior_r2'] = 0.0
        
        return epoch_metrics
    
    def _detailed_evaluation(self, epoch):
        """Run detailed evaluation including behavior prediction analysis."""
        if not self.model.enable_behavior_prediction:
            return
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for neural_data, behavior_data in self.val_loader:
                neural_data = neural_data.to(self.device)
                _, _, _, _, _, behavior_pred = self.model(neural_data)
                all_predictions.append(behavior_pred.cpu().numpy())
                all_targets.append(behavior_data.numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Evaluate behavior predictions
        behavior_results = evaluate_behavior_predictions(predictions, targets)
        
        # Log to tensorboard
        for feature, metrics in behavior_results.items():
            if feature != 'overall':
                self.writer.add_scalar(f'Behavior/{feature}_R2', metrics['r2'], epoch)
                self.writer.add_scalar(f'Behavior/{feature}_MSE', metrics['mse'], epoch)
        
        # Log overall metrics
        if 'overall' in behavior_results:
            overall = behavior_results['overall']
            self.writer.add_scalar('Behavior/Overall_R2', overall['mean_r2'], epoch)
    
    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """Log metrics to tensorboard and internal tracking."""
        
        # Log to tensorboard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Log learning rate
        for param_group in self.optimizer.param_groups:
            self.writer.add_scalar('Training/LearningRate', param_group['lr'], epoch)
            break  # Only log the first group
        
        # Store in internal tracking
        for key, value in train_metrics.items():
            self.metrics[f'train_{key}'].append(value)
        
        for key, value in val_metrics.items():
            self.metrics[f'val_{key}'].append(value)
    
    def _print_progress(self, epoch, train_metrics, val_metrics, epoch_time, total_time):
        """Print training progress."""
        print(f"\nEpoch [{epoch+1:3d}] ({epoch_time:.1f}s, total: {total_time/60:.1f}min)")
        print(f"  Train | Recon: {train_metrics['recon_loss']:.4f}, "
              f"VQ: {train_metrics['vq_loss']:.4f}, "
              f"Behavior: {train_metrics['behavior_loss']:.4f}, "
              f"Perplexity: {train_metrics['perplexity']:.1f}")
        print(f"  Val   | Recon: {val_metrics['recon_loss']:.4f}, "
              f"VQ: {val_metrics['vq_loss']:.4f}, "
              f"Behavior: {val_metrics['behavior_loss']:.4f}, "
              f"Total: {val_metrics['total_loss']:.4f}")
        
        if 'behavior_r2' in val_metrics:
            print(f"        | Behavior R²: {val_metrics['behavior_r2']:.3f}")
        
        print(f"  Best Val Loss: {self.best_val_loss:.4f} (epoch {self.best_epoch+1}), "
              f"Patience: {self.patience_counter}")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': dict(self.metrics),
            'model_config': {
                'quantizer_type': self.model.quantizer_type,
                'enable_behavior_prediction': self.model.enable_behavior_prediction
            }
        }
        
        if is_best:
            save_path = os.path.join(self.save_dir, f'{self.experiment_name}_best.pth')
            torch.save(checkpoint, save_path)
        
        # Always save latest
        latest_path = os.path.join(self.save_dir, f'{self.experiment_name}_latest.pth')
        torch.save(checkpoint, latest_path)
    
    def _final_evaluation(self):
        """Run comprehensive final evaluation."""
        results = {
            'training_metrics': dict(self.metrics),
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }
        
        # Load best model for evaluation
        best_path = os.path.join(self.save_dir, f'{self.experiment_name}_best.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set if available
        if self.test_loader is not None:
            test_metrics = self._evaluate_on_loader(self.test_loader, "Test")
            results['test_metrics'] = test_metrics
        
        # Evaluate on validation set
        val_metrics = self._evaluate_on_loader(self.val_loader, "Validation")
        results['final_val_metrics'] = val_metrics
        
        # Codebook usage analysis
        codebook_stats = self.model.get_codebook_usage()
        results['codebook_usage'] = codebook_stats
        
        print(f"\nFinal Results:")
        print(f"  Best epoch: {self.best_epoch + 1}")
        print(f"  Best validation loss: {self.best_val_loss:.6f}")
        print(f"  Codebook usage: {codebook_stats.get('usage_percentage', 'N/A'):.1f}%")
        
        return results
    
    def _evaluate_on_loader(self, loader, dataset_name):
        """Evaluate model on a specific dataloader."""
        self.model.eval()
        
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        total_behavior_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        all_reconstructions = []
        all_originals = []
        
        with torch.no_grad():
            for neural_data, behavior_data in loader:
                neural_data = neural_data.to(self.device)
                behavior_data = behavior_data.to(self.device)
                
                vq_loss, neural_recon, perplexity, _, _, behavior_pred = self.model(neural_data)
                
                recon_loss = F.mse_loss(neural_recon, neural_data)
                behavior_loss = 0.0
                
                if self.model.enable_behavior_prediction and behavior_pred is not None:
                    behavior_loss = F.mse_loss(behavior_pred, behavior_data)
                    all_predictions.append(behavior_pred.cpu().numpy())
                    all_targets.append(behavior_data.cpu().numpy())
                
                total_recon_loss += recon_loss.item()
                total_vq_loss += vq_loss.item()
                total_behavior_loss += behavior_loss.item() if isinstance(behavior_loss, torch.Tensor) else behavior_loss
                total_perplexity += perplexity.item()
                num_batches += 1
                
                # Store some examples for visualization
                if len(all_reconstructions) < 5:  # Store first 5 batches
                    all_reconstructions.append(neural_recon.cpu().numpy())
                    all_originals.append(neural_data.cpu().numpy())
        
        metrics = {
            'recon_loss': total_recon_loss / num_batches,
            'vq_loss': total_vq_loss / num_batches,
            'behavior_loss': total_behavior_loss / num_batches,
            'perplexity': total_perplexity / num_batches
        }
        
        # Behavior evaluation
        if all_predictions:
            predictions = np.vstack(all_predictions)
            targets = np.vstack(all_targets)
            behavior_results = evaluate_behavior_predictions(predictions, targets)
            metrics['behavior_evaluation'] = behavior_results
            
            print(f"\n{dataset_name} Behavior Prediction Results:")
            from models.behavior import print_behavior_evaluation
            print_behavior_evaluation(behavior_results)
        
        # Store examples for visualization
        if all_reconstructions:
            metrics['reconstruction_examples'] = {
                'originals': np.vstack(all_originals[:2]),  # First 2 batches
                'reconstructions': np.vstack(all_reconstructions[:2])
            }
        
        print(f"\n{dataset_name} Metrics:")
        print(f"  Reconstruction MSE: {metrics['recon_loss']:.6f}")
        print(f"  VQ Loss: {metrics['vq_loss']:.6f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        if metrics['behavior_loss'] > 0:
            print(f"  Behavior Loss: {metrics['behavior_loss']:.6f}")
        
        return metrics
    
    def _save_results(self, results):
        """Save final results and create visualizations."""
        import json
        import pickle
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Save results as JSON (for easy reading)
        json_results = convert_numpy_types(results)
        
        json_path = os.path.join(self.save_dir, f'{self.experiment_name}_results.json')
        try:
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not save JSON results: {e}")
            # Fallback: save only basic metrics
            basic_results = {
                'best_epoch': int(results.get('best_epoch', 0)),
                'best_val_loss': float(results.get('best_val_loss', 0.0)),
                'final_metrics_available': True
            }
            with open(json_path, 'w') as f:
                json.dump(basic_results, f, indent=2)
        
        # Save full results as pickle
        pickle_path = os.path.join(self.save_dir, f'{self.experiment_name}_results.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Create training curves visualization
        if 'training_metrics' in results:
            self._plot_training_curves(results['training_metrics'])
        
        # Create reconstruction examples if available
        if 'final_val_metrics' in results and 'reconstruction_examples' in results['final_val_metrics']:
            self._plot_reconstruction_examples(results['final_val_metrics']['reconstruction_examples'])
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pickle_path}")
    
    def _plot_training_curves(self, metrics):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(metrics['train_total_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, metrics['train_recon_loss'], label='Train', alpha=0.7)
        axes[0, 0].plot(epochs, metrics['val_recon_loss'], label='Validation', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Reconstruction Loss')
        axes[0, 0].set_title('Reconstruction Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # VQ Loss
        axes[0, 1].plot(epochs, metrics['train_vq_loss'], label='Train', alpha=0.7)
        axes[0, 1].plot(epochs, metrics['val_vq_loss'], label='Validation', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('VQ Loss')
        axes[0, 1].set_title('Vector Quantization Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Perplexity
        axes[1, 0].plot(epochs, metrics['train_perplexity'], label='Train', alpha=0.7)
        axes[1, 0].plot(epochs, metrics['val_perplexity'], label='Validation', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].set_title('Codebook Perplexity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Behavior loss (if available)
        if 'train_behavior_loss' in metrics and any(x > 0 for x in metrics['train_behavior_loss']):
            axes[1, 1].plot(epochs, metrics['train_behavior_loss'], label='Train', alpha=0.7)
            axes[1, 1].plot(epochs, metrics['val_behavior_loss'], label='Validation', alpha=0.7)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Behavior Loss')
            axes[1, 1].set_title('Behavior Prediction Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Behavior\nPrediction', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Behavior Prediction Loss')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'{self.experiment_name}_training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Training curves: {plot_path}")
    
    def _plot_reconstruction_examples(self, examples):
        """Plot reconstruction examples."""
        originals = examples['originals']
        reconstructions = examples['reconstructions']
        
        n_examples = min(3, originals.shape[0])
        n_neurons_to_show = min(15, originals.shape[1])
        
        fig, axes = plt.subplots(n_examples, 2, figsize=(12, 3 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_examples):
            # Select most active neurons
            neuron_vars = np.var(originals[i], axis=1)
            top_neurons = np.argsort(neuron_vars)[-n_neurons_to_show:]
            
            # Original
            im1 = axes[i, 0].imshow(originals[i, top_neurons, :], 
                                   aspect='auto', cmap='viridis', interpolation='nearest')
            axes[i, 0].set_ylabel(f'Example {i+1}\nNeurons')
            axes[i, 0].set_xlabel('Time')
            if i == 0:
                axes[i, 0].set_title('Original')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Reconstruction
            im2 = axes[i, 1].imshow(reconstructions[i, top_neurons, :], 
                                   aspect='auto', cmap='viridis', interpolation='nearest')
            axes[i, 1].set_xlabel('Time')
            if i == 0:
                axes[i, 1].set_title('Reconstruction')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'{self.experiment_name}_reconstructions.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Reconstruction examples: {plot_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.metrics = defaultdict(list, checkpoint.get('metrics', {}))
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return checkpoint
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'writer'):
            self.writer.close()


# Utility functions for easy training setup
def create_trainer_from_config(model, dataloaders, config):
    """
    Create trainer from configuration dictionary.
    
    Args:
        model: CalciumVQVAE model
        dataloaders: tuple of (train_loader, val_loader, test_loader)
        config: configuration dictionary
    
    Returns:
        CalciumTrainer instance
    """
    train_loader, val_loader, test_loader = dataloaders
    
    trainer_config = config.get('trainer', {})
    
    trainer = CalciumTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config.get('device', 'cuda'),
        save_dir=config.get('save_dir', './results'),
        experiment_name=config.get('experiment_name', None)
    )
    
    return trainer


def train_calcium_vqvae(model, dataloaders, training_config=None):
    """
    High-level function to train a calcium VQ-VAE model.
    
    Args:
        model: CalciumVQVAE model
        dataloaders: tuple of (train_loader, val_loader, test_loader)
        training_config: training configuration dictionary
    
    Returns:
        tuple: (trainer, results)
    """
    # Default training configuration
    default_config = {
        'num_epochs': 100,
        'learning_rate': 3e-4,
        'behavior_weight': 0.5,
        'patience': 20,
        'gradient_clip': 1.0,
        'save_best': True,
        'eval_interval': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './results'
    }
    
    if training_config:
        default_config.update(training_config)
    
    # Create trainer
    trainer = CalciumTrainer(
        model=model,
        train_loader=dataloaders[0],
        val_loader=dataloaders[1],
        test_loader=dataloaders[2] if len(dataloaders) > 2 else None,
        device=default_config['device'],
        save_dir=default_config['save_dir'],
        experiment_name=default_config.get('experiment_name')
    )
    
    # Train model
    results = trainer.train(**{k: v for k, v in default_config.items() 
                             if k not in ['device', 'save_dir', 'experiment_name']})
    
    return trainer, results


if __name__ == "__main__":
    # Test trainer setup
    print("Testing CalciumTrainer...")
    
    # Create dummy model and data
    from models.vqvae import CalciumVQVAE
    from torch.utils.data import DataLoader, TensorDataset
    
    # Dummy data
    neural_data = torch.randn(200, 50, 100)  # 200 samples, 50 neurons, 100 timepoints
    behavior_data = torch.randn(200, 4)      # 4 behavioral features
    
    dataset = TensorDataset(neural_data, behavior_data)
    train_loader = DataLoader(dataset[:160], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[160:180], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[180:], batch_size=32, shuffle=False)
    
    # Create model
    model = CalciumVQVAE(
        num_neurons=50,
        num_hiddens=64,
        num_residual_layers=2,
        num_residual_hiddens=32,
        num_embeddings=256,
        embedding_dim=32,
        commitment_cost=0.25,
        quantizer_type='improved_vq',
        enable_behavior_prediction=True
    )
    
    # Create trainer
    trainer = CalciumTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device='cpu',  # Use CPU for testing
        save_dir='./test_results'
    )
    
    print("Trainer created successfully!")
    print("Running short training test...")
    
    # Run short training
    results = trainer.train(num_epochs=3, learning_rate=1e-3)
    
    print("Training test completed!")
    print(f"Final validation loss: {results['best_val_loss']:.6f}")
    
    trainer.close()