#!/usr/bin/env python3
"""
Training script for Calcium VQ-VAE models.

This script provides a clean interface for training VQ-VAE models on calcium imaging data
from the Allen Brain Observatory or other sources.

Usage:
    python scripts/train_calcium.py --config configs/default.yaml
    python scripts/train_calcium.py --quantizer grouped_rvq --epochs 150 --batch_size 64
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Try to import yaml, with fallback
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not available. Configuration files will be disabled.")
    print("Install with: pip install PyYAML")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try importing with fallback for missing modules
try:
    from models.vqvae import CalciumVQVAE, create_calcium_vqvae
    from datasets.calcium import create_calcium_dataloaders  
    from training.calcium_trainer import CalciumTrainer, train_calcium_vqvae
    from utils.allen_utils import find_best_sessions
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure you have created the following directory structure:")
    print("- utils/__init__.py")
    print("- training/__init__.py") 
    print("- scripts/__init__.py")
    print("- utils/allen_utils.py")
    print("- training/calcium_trainer.py")
    print("- datasets/calcium.py")
    print("\nAlternatively, run from project root: python -m scripts.train_calcium")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Calcium VQ-VAE")
    
    # Data arguments
    parser.add_argument('--dataset_type', type=str, default='single',
                       choices=['single', 'multi'], 
                       help='Type of dataset to use')
    parser.add_argument('--session_id', type=int, default=None,
                       help='Specific Allen Brain session ID')
    parser.add_argument('--window_size', type=int, default=50,
                       help='Temporal window size')
    parser.add_argument('--stride', type=int, default=10,
                       help='Stride for sliding windows')
    parser.add_argument('--min_neurons', type=int, default=30,
                       help='Minimum number of neurons to keep')
    
    # Model arguments
    parser.add_argument('--quantizer', type=str, default='improved_vq',
                       choices=['improved_vq', 'grouped_rvq'],
                       help='Type of vector quantizer')
    parser.add_argument('--num_hiddens', type=int, default=128,
                       help='Number of hidden dimensions')
    parser.add_argument('--num_residual_layers', type=int, default=3,
                       help='Number of residual layers')
    parser.add_argument('--num_residual_hiddens', type=int, default=64,
                       help='Hidden dimensions in residual blocks')
    parser.add_argument('--num_embeddings', type=int, default=512,
                       help='Number of codebook embeddings')
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--commitment_cost', type=float, default=0.25,
                       help='Commitment cost for VQ loss')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--behavior_dim', type=int, default=4,
                       help='Number of behavioral features')
    parser.add_argument('--disable_behavior', action='store_true',
                       help='Disable behavior prediction')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--behavior_weight', type=float, default=0.5,
                       help='Weight for behavior prediction loss')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='Evaluation interval')
    
    # System arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of dataloader workers')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for the experiment')
    parser.add_argument('--config', type=str, default=None,
                       help='YAML config file path')
    
    # Utility arguments
    parser.add_argument('--find_sessions', action='store_true',
                       help='Find and list good Allen Brain sessions')
    parser.add_argument('--dry_run', action='store_true',
                       help='Setup everything but don\'t train')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with smaller dataset')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML not available. Install with: pip install PyYAML")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_args(config, args):
    """Merge command line arguments with config file."""
    # Command line arguments override config file
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


def setup_experiment_name(config):
    """Generate experiment name if not provided."""
    if config.get('experiment_name') is None:
        import time
        timestamp = int(time.time())
        quantizer = config.get('quantizer', 'vq')
        behavior = 'behav' if not config.get('disable_behavior', False) else 'nobehav'
        config['experiment_name'] = f'calcium_{quantizer}_{behavior}_{timestamp}'
    
    print(f"Experiment name: {config['experiment_name']}")
    return config


def create_model_from_config(config, num_neurons):
    """Create model from configuration."""
    model_config = {
        'num_neurons': num_neurons,
        'num_hiddens': config.get('num_hiddens', 128),
        'num_residual_layers': config.get('num_residual_layers', 3),
        'num_residual_hiddens': config.get('num_residual_hiddens', 64),
        'num_embeddings': config.get('num_embeddings', 512),
        'embedding_dim': config.get('embedding_dim', 64),
        'commitment_cost': config.get('commitment_cost', 0.25),
        'quantizer_type': config.get('quantizer', 'improved_vq'),
        'dropout_rate': config.get('dropout_rate', 0.1),
        'behavior_dim': config.get('behavior_dim', 4),
        'enable_behavior_prediction': not config.get('disable_behavior', False)
    }
    
    model = CalciumVQVAE(**model_config)
    
    print(f"Created {model_config['quantizer_type']} model:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Behavior prediction: {model_config['enable_behavior_prediction']}")
    
    return model


def create_dataloaders_from_config(config):
    """Create dataloaders from configuration."""
    dataloader_config = {
        'dataset_type': config.get('dataset_type', 'single'),
        'batch_size': config.get('batch_size', 32),
        'num_workers': config.get('num_workers', 0),
        'session_id': config.get('session_id'),
        'window_size': config.get('window_size', 50),
        'stride': config.get('stride', 10),
        'min_neurons': config.get('min_neurons', 30),
        'augment': not config.get('debug', False)  # No augmentation in debug mode
    }
    
    # Debug mode: smaller dataset
    if config.get('debug', False):
        dataloader_config.update({
            'validation_split': 0.3,
            'test_split': 0.3,
            'min_neurons': min(20, dataloader_config['min_neurons'])
        })
        print("Debug mode: using smaller dataset splits")
    
    print(f"Creating dataloaders with config: {dataloader_config}")
    
    try:
        train_loader, val_loader, test_loader, dataset_info = create_calcium_dataloaders(**dataloader_config)
        
        print(f"Dataset info:")
        print(f"  Total samples: {dataset_info['total_samples']}")
        print(f"  Train/Val/Test: {dataset_info['train_samples']}/{dataset_info['val_samples']}/{dataset_info['test_samples']}")
        print(f"  Neural shape: {dataset_info['neural_shape']}")
        print(f"  Behavior shape: {dataset_info['behavior_shape']}")
        print(f"  Session ID: {dataset_info.get('session_id', 'Multiple/Unknown')}")
        
        return (train_loader, val_loader, test_loader), dataset_info
    
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("This might be due to Allen SDK not being installed or network issues.")
        raise


def main():
    """Main training function."""
    args = parse_args()
    
    # Handle utility commands
    if args.find_sessions:
        print("Finding good Allen Brain Observatory sessions...")
        sessions = find_best_sessions(min_neurons=args.min_neurons, max_sessions=10)
        print(f"Found {len(sessions)} good sessions:")
        for session in sessions:
            print(f"  Session {session['id']}: {session['neurons']} neurons, {session.get('cre_line', 'unknown')}")
        return
    
    # Setup configuration
    config = {}
    if args.config:
        if not YAML_AVAILABLE:
            print("Error: PyYAML not available for config files. Install with: pip install PyYAML")
            return 1
        config = load_config(args.config)
    config = merge_config_args(config, args)
    config = setup_experiment_name(config)
    
    # Setup device
    if config.get('device') is None:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {config['device']}")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(config['save_dir'], f"{config['experiment_name']}_config.yaml")
    if YAML_AVAILABLE:
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Configuration saved to: {config_save_path}")
    else:
        # Save as text file instead
        config_save_path = config_save_path.replace('.yaml', '.txt')
        with open(config_save_path, 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        print(f"Configuration saved to: {config_save_path} (text format)")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        dataloaders, dataset_info = create_dataloaders_from_config(config)
        train_loader, val_loader, test_loader = dataloaders
    except Exception as e:
        print(f"Failed to create dataloaders: {e}")
        return 1
    
    # Create model
    print("\nCreating model...")
    num_neurons = dataset_info['neural_shape'][0]
    model = create_model_from_config(config, num_neurons)
    
    if config.get('dry_run', False):
        print("\nDry run completed successfully!")
        print("Model and dataloaders created without training.")
        return 0
    
    # Setup training configuration
    training_config = {
        'num_epochs': config.get('epochs', 100),
        'learning_rate': config.get('learning_rate', 3e-4),
        'behavior_weight': config.get('behavior_weight', 0.5),
        'patience': config.get('patience', 20),
        'gradient_clip': config.get('gradient_clip', 1.0),
        'save_best': True,
        'eval_interval': config.get('eval_interval', 5),
        'device': config['device'],
        'save_dir': config['save_dir'],
        'experiment_name': config['experiment_name']
    }
    
    # Debug mode: shorter training
    if config.get('debug', False):
        training_config.update({
            'num_epochs': 5,
            'eval_interval': 2,
            'patience': 3
        })
        print("Debug mode: using shorter training")
    
    print("\nStarting training...")
    print(f"Training configuration: {training_config}")
    
    try:
        # Train model
        trainer, results = train_calcium_vqvae(model, dataloaders, training_config)
        
        print(f"\nTraining completed successfully!")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Results saved in: {config['save_dir']}")
        
        # Print final summary
        if 'test_metrics' in results:
            test_metrics = results['test_metrics']
            print(f"\nFinal Test Results:")
            print(f"  Reconstruction MSE: {test_metrics['recon_loss']:.6f}")
            print(f"  Perplexity: {test_metrics['perplexity']:.2f}")
            
            if 'behavior_evaluation' in test_metrics:
                behavior_eval = test_metrics['behavior_evaluation']
                if 'overall' in behavior_eval:
                    print(f"  Mean Behavior R²: {behavior_eval['overall']['mean_r2']:.3f}")
        
        if 'codebook_usage' in results:
            usage = results['codebook_usage']
            print(f"  Codebook Usage: {usage.get('usage_percentage', 'N/A'):.1f}%")
        
        trainer.close()
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def create_default_config():
    """Create a default configuration file."""
    default_config = {
        # Data configuration
        'dataset_type': 'single',
        'session_id': None,
        'window_size': 50,
        'stride': 10,
        'min_neurons': 30,
        
        # Model configuration
        'quantizer': 'improved_vq',
        'num_hiddens': 128,
        'num_residual_layers': 3,
        'num_residual_hiddens': 64,
        'num_embeddings': 512,
        'embedding_dim': 64,
        'commitment_cost': 0.25,
        'dropout_rate': 0.1,
        'behavior_dim': 4,
        'disable_behavior': False,
        
        # Training configuration
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'behavior_weight': 0.5,
        'patience': 20,
        'gradient_clip': 1.0,
        'eval_interval': 5,
        
        # System configuration
        'device': None,  # Auto-detect
        'num_workers': 0,
        'save_dir': './results'
    }
    
    return default_config


if __name__ == "__main__":
    # Create default config if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--create-config':
        config = create_default_config()
        config_path = 'configs/default.yaml'
        os.makedirs('configs', exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Default configuration created at: {config_path}")
        print("Edit this file and use with: python scripts/train_calcium.py --config configs/default.yaml")
        sys.exit(0)
    
    exit_code = main()
    sys.exit(exit_code)