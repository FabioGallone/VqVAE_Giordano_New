import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score


class BehaviorDecoder(nn.Module):
    """
    Neural decoder for predicting behavior from latent representations.
    
    This replaces simple linear models with a more powerful neural network
    that can capture non-linear relationships between neural activity and behavior.
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=4, dropout=0.2, 
                 num_layers=3):
        super(BehaviorDecoder, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Create multiple hidden layers
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
            hidden_dim = hidden_dim // 2  # Progressively smaller layers
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)


class BehaviorHead(nn.Module):
    """
    Integrated behavior prediction head for VQ-VAE models.
    
    This head can be attached to any VQ-VAE to enable multi-task learning
    with both reconstruction and behavior prediction.
    """
    
    def __init__(self, embedding_dim, behavior_dim=4, hidden_dim=128, dropout=0.2):
        super(BehaviorHead, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, behavior_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, quantized_features):
        """
        Args:
            quantized_features: (B, embedding_dim, time_steps) or (B, embedding_dim, H, W)
        Returns:
            behavior_pred: (B, behavior_dim)
        """
        # Global average pooling to get fixed-size representation
        if len(quantized_features.shape) == 3:  # 1D case
            pooled = self.global_pool(quantized_features).squeeze(-1)  # (B, embedding_dim)
        else:  # 2D case
            pooled = F.adaptive_avg_pool2d(quantized_features, (1, 1)).squeeze(-1).squeeze(-1)
        
        return self.head(pooled)


class MultiTaskBehaviorDecoder(nn.Module):
    """
    Multi-task decoder that can predict multiple behavioral variables
    with task-specific heads and shared features.
    """
    
    def __init__(self, input_dim, shared_hidden_dim=256, task_configs=None, dropout=0.2):
        super(MultiTaskBehaviorDecoder, self).__init__()
        
        # Default task configuration
        if task_configs is None:
            task_configs = {
                'speed': {'dim': 1, 'activation': 'relu'},  # Non-negative speed
                'direction': {'dim': 2, 'activation': 'tanh'},  # Direction vector
                'acceleration': {'dim': 1, 'activation': 'linear'},  # Can be negative
                'state': {'dim': 3, 'activation': 'softmax'}  # Categorical state
            }
        
        self.task_configs = task_configs
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden_dim, shared_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        shared_out_dim = shared_hidden_dim // 2
        
        for task_name, config in task_configs.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(shared_out_dim, 64),
                nn.ReLU(),
                nn.Linear(64, config['dim'])
            )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        
        outputs = {}
        for task_name, config in self.task_configs.items():
            raw_output = self.task_heads[task_name](shared_features)
            
            # Apply task-specific activation
            if config['activation'] == 'relu':
                outputs[task_name] = F.relu(raw_output)
            elif config['activation'] == 'tanh':
                outputs[task_name] = torch.tanh(raw_output)
            elif config['activation'] == 'softmax':
                outputs[task_name] = F.softmax(raw_output, dim=-1)
            else:  # linear
                outputs[task_name] = raw_output
        
        return outputs


class BehaviorLoss(nn.Module):
    """
    Specialized loss function for behavior prediction that can handle
    multiple behavioral features with different scales and importance.
    """
    
    def __init__(self, feature_weights=None, loss_types=None):
        super(BehaviorLoss, self).__init__()
        
        # Default weights for different behavioral features
        if feature_weights is None:
            feature_weights = [1.0, 0.5, 0.8, 0.3]  # [speed, variance, max, change]
        
        # Default loss types for each feature
        if loss_types is None:
            loss_types = ['mse', 'mse', 'mse', 'mae']  # Different loss for each feature
        
        self.feature_weights = torch.tensor(feature_weights)
        self.loss_types = loss_types
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, num_features)
            targets: (B, num_features)
        """
        device = predictions.device
        self.feature_weights = self.feature_weights.to(device)
        
        total_loss = 0.0
        
        for i, (loss_type, weight) in enumerate(zip(self.loss_types, self.feature_weights)):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            # Handle NaN values
            valid_mask = torch.isfinite(target_i) & torch.isfinite(pred_i)
            if valid_mask.sum() == 0:
                continue
            
            pred_valid = pred_i[valid_mask]
            target_valid = target_i[valid_mask]
            
            if loss_type == 'mse':
                loss_i = F.mse_loss(pred_valid, target_valid)
            elif loss_type == 'mae':
                loss_i = F.l1_loss(pred_valid, target_valid)
            elif loss_type == 'huber':
                loss_i = F.smooth_l1_loss(pred_valid, target_valid)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            total_loss += weight * loss_i
        
        return total_loss / len(self.loss_types)


class TemporalBehaviorDecoder(nn.Module):
    """
    Decoder that can predict behavior at multiple time scales.
    Useful for predicting both instant behavior and future behavior.
    """
    
    def __init__(self, input_dim, output_dim=4, prediction_horizons=[1, 5, 10],
                 hidden_dim=128, dropout=0.2):
        super(TemporalBehaviorDecoder, self).__init__()
        
        self.prediction_horizons = prediction_horizons
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate heads for different time horizons
        self.temporal_heads = nn.ModuleDict()
        for horizon in prediction_horizons:
            self.temporal_heads[f'horizon_{horizon}'] = nn.Sequential(
                nn.Linear(hidden_dim // 2, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        
        predictions = {}
        for horizon in self.prediction_horizons:
            predictions[f'horizon_{horizon}'] = self.temporal_heads[f'horizon_{horizon}'](shared_features)
        
        return predictions


class AdaptiveBehaviorHead(nn.Module):
    """
    Behavior head that adapts its predictions based on neural activity patterns.
    Uses attention mechanism to focus on relevant time periods.
    """
    
    def __init__(self, embedding_dim, behavior_dim=4, hidden_dim=128, 
                 num_attention_heads=4, dropout=0.2):
        super(AdaptiveBehaviorHead, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        
        # Multi-head attention for temporal pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, behavior_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, quantized_features):
        """
        Args:
            quantized_features: (B, embedding_dim, time_steps)
        Returns:
            behavior_pred: (B, behavior_dim)
        """
        # Transpose for attention: (B, time_steps, embedding_dim)
        x = quantized_features.transpose(1, 2)
        
        # Self-attention to find important time periods
        attended, attention_weights = self.attention(x, x, x)
        
        # Global average pooling of attended features
        pooled = attended.mean(dim=1)  # (B, embedding_dim)
        
        # Predict behavior
        behavior_pred = self.prediction_head(pooled)
        
        return behavior_pred


def evaluate_behavior_predictions(predictions, targets, feature_names=None):
    """
    Comprehensive evaluation of behavior predictions.
    
    Args:
        predictions: numpy array (N, num_features)
        targets: numpy array (N, num_features)
        feature_names: list of feature names
    
    Returns:
        dict with evaluation metrics
    """
    if feature_names is None:
        feature_names = ['Mean Speed', 'Speed Std', 'Max Speed', 'Speed Change']
    
    # Ensure numpy arrays
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    # Handle NaN values
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    
    results = {}
    
    for i, feature_name in enumerate(feature_names):
        pred_i = predictions[:, i]
        target_i = targets[:, i]
        
        # Skip if no variance in targets
        if np.std(target_i) < 1e-8:
            results[feature_name] = {
                'r2': 0.0,
                'mse': 0.0,
                'mae': 0.0,
                'correlation': 0.0
            }
            continue
        
        # Calculate metrics
        try:
            r2 = r2_score(target_i, pred_i)
            mse = np.mean((pred_i - target_i) ** 2)
            mae = np.mean(np.abs(pred_i - target_i))
            correlation = np.corrcoef(pred_i, target_i)[0, 1]
            
            if not np.isfinite(correlation):
                correlation = 0.0
        except:
            r2 = mse = mae = correlation = 0.0
        
        results[feature_name] = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'correlation': correlation
        }
    
    # Overall metrics
    all_r2 = [results[name]['r2'] for name in feature_names]
    all_correlations = [results[name]['correlation'] for name in feature_names]
    
    results['overall'] = {
        'mean_r2': np.mean(all_r2),
        'mean_correlation': np.mean(all_correlations),
        'best_r2': max(all_r2),
        'worst_r2': min(all_r2)
    }
    
    return results


def print_behavior_evaluation(results, title="Behavior Prediction Results"):
    """Pretty print behavior evaluation results."""
    print(f"\n{title}")
    print("=" * len(title))
    
    for feature_name, metrics in results.items():
        if feature_name == 'overall':
            continue
        
        print(f"{feature_name:15} | "
              f"R²: {metrics['r2']:6.3f} | "
              f"MSE: {metrics['mse']:8.4f} | "
              f"MAE: {metrics['mae']:8.4f} | "
              f"Corr: {metrics['correlation']:6.3f}")
    
    if 'overall' in results:
        print("-" * 60)
        overall = results['overall']
        print(f"{'Overall':15} | "
              f"Mean R²: {overall['mean_r2']:6.3f} | "
              f"Best: {overall['best_r2']:6.3f} | "
              f"Worst: {overall['worst_r2']:6.3f}")


def create_behavior_targets(running_speed, window_size=50, feature_type='standard'):
    """
    Create behavior targets from running speed data.
    
    Args:
        running_speed: running speed timeseries
        window_size: window size for feature computation
        feature_type: 'standard', 'enhanced', or 'temporal'
    
    Returns:
        numpy array: behavior targets
    """
    if feature_type == 'standard':
        # Standard 4 features
        features = np.array([
            np.mean(running_speed),
            np.std(running_speed),
            np.max(running_speed),
            running_speed[-1] - running_speed[0]
        ])
    elif feature_type == 'enhanced':
        # Enhanced feature set
        features = np.array([
            np.mean(running_speed),           # Mean speed
            np.std(running_speed),            # Speed variability
            np.max(running_speed),            # Peak speed
            np.min(running_speed),            # Minimum speed
            np.percentile(running_speed, 75), # 75th percentile
            np.sum(running_speed > 1.0),      # Time above threshold
            len(np.where(np.diff(running_speed) > 0.5)[0])  # Number of accelerations
        ])
    elif feature_type == 'temporal':
        # Temporal sequence features
        num_windows = len(running_speed) - window_size + 1
        features = np.zeros((num_windows, 4))
        
        for i in range(num_windows):
            window = running_speed[i:i + window_size]
            features[i] = [
                np.mean(window),
                np.std(window),
                np.max(window),
                window[-1] - window[0]
            ]
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    return features


def compute_behavior_baseline(targets, predictions=None):
    """
    Compute baseline performance for behavior prediction.
    
    Args:
        targets: ground truth behavior targets
        predictions: model predictions (optional)
    
    Returns:
        dict: baseline metrics
    """
    baselines = {}
    
    # Mean baseline
    mean_pred = np.mean(targets, axis=0, keepdims=True)
    mean_pred = np.repeat(mean_pred, targets.shape[0], axis=0)
    
    baselines['mean'] = evaluate_behavior_predictions(mean_pred, targets)
    
    # Last value baseline (for temporal data)
    if targets.shape[0] > 1:
        last_value_pred = np.repeat(targets[:-1], 1, axis=0)
        last_value_pred = np.vstack([targets[0:1], last_value_pred])
        
        baselines['last_value'] = evaluate_behavior_predictions(last_value_pred, targets)
    
    # Linear trend baseline
    if targets.shape[0] > 2:
        trend_pred = np.zeros_like(targets)
        for i in range(targets.shape[1]):
            # Fit linear trend
            x = np.arange(targets.shape[0])
            coeffs = np.polyfit(x, targets[:, i], 1)
            trend_pred[:, i] = np.polyval(coeffs, x)
        
        baselines['trend'] = evaluate_behavior_predictions(trend_pred, targets)
    
    # Model comparison (if predictions provided)
    if predictions is not None:
        baselines['model'] = evaluate_behavior_predictions(predictions, targets)
        
        # Improvement over baselines
        model_r2 = baselines['model']['overall']['mean_r2']
        mean_r2 = baselines['mean']['overall']['mean_r2']
        
        baselines['improvement'] = {
            'over_mean': model_r2 - mean_r2,
            'relative_improvement': (model_r2 - mean_r2) / (1 - mean_r2) if mean_r2 < 1 else 0
        }
    
    return baselines


if __name__ == "__main__":
    # Test behavior decoder
    print("Testing BehaviorDecoder:")
    decoder = BehaviorDecoder(input_dim=256, output_dim=4)
    x = torch.randn(32, 256)
    output = decoder(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    
    # Test behavior head
    print("\nTesting BehaviorHead:")
    head = BehaviorHead(embedding_dim=64, behavior_dim=4)
    quantized = torch.randn(32, 64, 50)  # (batch, embedding_dim, time)
    behavior_pred = head(quantized)
    print(f"Quantized: {quantized.shape}, Prediction: {behavior_pred.shape}")
    
    # Test multi-task decoder
    print("\nTesting MultiTaskBehaviorDecoder:")
    multi_decoder = MultiTaskBehaviorDecoder(input_dim=256)
    outputs = multi_decoder(x)
    for task, output in outputs.items():
        print(f"Task '{task}': {output.shape}")
    
    # Test temporal decoder
    print("\nTesting TemporalBehaviorDecoder:")
    temporal_decoder = TemporalBehaviorDecoder(input_dim=256, prediction_horizons=[1, 5, 10])
    temporal_outputs = temporal_decoder(x)
    for horizon, output in temporal_outputs.items():
        print(f"Horizon '{horizon}': {output.shape}")
    
    # Test adaptive head
    print("\nTesting AdaptiveBehaviorHead:")
    adaptive_head = AdaptiveBehaviorHead(embedding_dim=64, behavior_dim=4)
    adaptive_pred = adaptive_head(quantized)
    print(f"Adaptive prediction: {adaptive_pred.shape}")
    
    # Test evaluation
    print("\nTesting evaluation:")
    fake_predictions = np.random.randn(100, 4)
    fake_targets = fake_predictions + 0.1 * np.random.randn(100, 4)  # Add noise
    results = evaluate_behavior_predictions(fake_predictions, fake_targets)
    print_behavior_evaluation(results)
    
    # Test baselines
    print("\nTesting baselines:")
    baselines = compute_behavior_baseline(fake_targets, fake_predictions)
    print("Baseline R² scores:")
    for baseline_name, baseline_results in baselines.items():
        if isinstance(baseline_results, dict) and 'overall' in baseline_results:
            print(f"  {baseline_name}: {baseline_results['overall']['mean_r2']:.3f}")
    
    print("\nAll tests completed successfully!")