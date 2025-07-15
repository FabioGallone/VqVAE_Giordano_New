import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, zscore
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class CalciumDataset(Dataset):
    """
    Dataset for calcium imaging data with optional data augmentation.
    
    Supports various preprocessing options and augmentation techniques
    specific to neural time series data.
    """
    
    def __init__(self, neural_data, behavior_data, augment=False, 
                 augment_prob=0.5, noise_std=0.1, time_shift_max=5):
        """
        Args:
            neural_data: numpy array (n_samples, n_neurons, time_steps)
            behavior_data: numpy array (n_samples, behavior_features)
            augment: whether to apply data augmentation
            augment_prob: probability of applying augmentation
            noise_std: standard deviation for gaussian noise augmentation
            time_shift_max: maximum time shift for temporal augmentation
        """
        self.neural_data = torch.FloatTensor(neural_data)
        self.behavior_data = torch.FloatTensor(behavior_data)
        self.augment = augment
        self.augment_prob = augment_prob
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        
        print(f"Dataset created with {len(self)} samples")
        print(f"Neural data shape: {self.neural_data.shape}")
        print(f"Behavior data shape: {self.behavior_data.shape}")
        
    def __len__(self):
        return len(self.neural_data)
    
    def __getitem__(self, idx):
        neural = self.neural_data[idx].clone()
        behavior = self.behavior_data[idx].clone()
        
        # Apply augmentation during training
        if self.augment and np.random.rand() < self.augment_prob:
            neural = self._augment_neural_data(neural)
            
        return neural, behavior
    
    def _augment_neural_data(self, neural_data):
        """Apply various augmentation techniques to neural data."""
        
        # 1. Gaussian noise
        if np.random.rand() < 0.5:
            noise = torch.randn_like(neural_data) * self.noise_std
            neural_data = neural_data + noise
        
        # 2. Temporal shift
        if np.random.rand() < 0.3 and self.time_shift_max > 0:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
            if shift != 0:
                neural_data = torch.roll(neural_data, shift, dims=-1)
        
        # 3. Temporal scaling (slight speed changes)
        if np.random.rand() < 0.2:
            scale_factor = np.random.uniform(0.9, 1.1)
            original_length = neural_data.shape[-1]
            new_length = int(original_length * scale_factor)
            
            if new_length != original_length:
                # Interpolate to new length, then crop/pad to original
                neural_np = neural_data.numpy()
                time_old = np.linspace(0, 1, original_length)
                time_new = np.linspace(0, 1, new_length)
                
                neural_scaled = np.zeros_like(neural_np)
                for i in range(neural_np.shape[0]):
                    interp_func = interp1d(time_old, neural_np[i], 
                                         kind='linear', fill_value='extrapolate')
                    neural_interp = interp_func(time_new)
                    
                    # Crop or pad to original length
                    if new_length > original_length:
                        neural_scaled[i] = neural_interp[:original_length]
                    else:
                        padded = np.pad(neural_interp, (0, original_length - new_length), 
                                      mode='edge')
                        neural_scaled[i] = padded
                
                neural_data = torch.FloatTensor(neural_scaled)
        
        # 4. Neuron dropout (randomly zero out some neurons)
        if np.random.rand() < 0.1:
            dropout_prob = np.random.uniform(0.05, 0.15)
            dropout_mask = torch.rand(neural_data.shape[0]) > dropout_prob
            neural_data = neural_data * dropout_mask.unsqueeze(-1)
        
        return neural_data


class AllenBrainDataset(Dataset):
    """
    Specialized dataset for Allen Brain Observatory data.
    
    Handles the specific data format and preprocessing pipeline
    for Allen Brain calcium imaging experiments.
    """
    
    def __init__(self, session_id=None, window_size=50, stride=10, 
                 min_neurons=30, transform=None, augment=False):
        """
        Args:
            session_id: Allen Brain session ID (if None, will find a good one)
            window_size: size of temporal windows
            stride: stride for sliding windows
            min_neurons: minimum number of neurons to keep
            transform: optional transform function
            augment: whether to apply augmentation
        """
        self.session_id = session_id
        self.window_size = window_size
        self.stride = stride
        self.min_neurons = min_neurons
        self.transform = transform
        self.augment = augment
        
        # Load and preprocess data
        self._load_allen_data()
        
    def _load_allen_data(self):
        """Load data from Allen Brain Observatory."""
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        
        print("Loading Allen Brain Observatory data...")
        boc = BrainObservatoryCache()
        
        # Find good session if not provided
        if self.session_id is None:
            self.session_id = self._find_good_session(boc)
        
        # Load session data
        data_set = boc.get_ophys_experiment_data(self.session_id)
        
        # Extract neural and behavioral data
        timestamps, dff_traces = data_set.get_dff_traces()
        run_ts, running_speed = data_set.get_running_speed()
        
        print(f"Session {self.session_id}: {dff_traces.shape[0]} neurons, "
              f"{dff_traces.shape[1]} timepoints")
        
        # Preprocess data
        neural_data, behavior_data = self._preprocess_data(
            timestamps, dff_traces, run_ts, running_speed
        )
        
        # Convert to tensors
        self.neural_data = torch.FloatTensor(neural_data)
        self.behavior_data = torch.FloatTensor(behavior_data)
        
        print(f"Final dataset: {len(self)} windows, "
              f"neural shape: {self.neural_data.shape}, "
              f"behavior shape: {self.behavior_data.shape}")
    
    def _find_good_session(self, boc):
        """Find a session with sufficient neurons and data."""
        experiments = boc.get_ophys_experiments()
        
        print("Searching for suitable sessions...")
        for exp in experiments[:500]:  # Check first 500
            if exp.get('cell_count', 0) >= self.min_neurons:
                try:
                    session_id = exp['id']
                    data_set = boc.get_ophys_experiment_data(session_id)
                    _, running_speed = data_set.get_running_speed()
                    
                    if len(running_speed) > 10000:  # Sufficient data
                        print(f"Selected session {session_id} with "
                              f"{exp['cell_count']} neurons")
                        return session_id
                except:
                    continue
        
        # Fallback to known good session
        print("Using fallback session")
        return 501940850
    
    def _preprocess_data(self, timestamps, dff_traces, run_ts, running_speed):
        """Comprehensive preprocessing pipeline."""
        
        # 1. Align behavioral data to neural timestamps
        speed_interp = interp1d(run_ts, running_speed, 
                               bounds_error=False, fill_value=0)
        speed_aligned = speed_interp(timestamps)
        speed_smooth = gaussian_filter1d(speed_aligned, sigma=5)
        
        # 2. Select active neurons
        active_neurons = self._select_active_neurons(dff_traces, speed_smooth)
        dff_active = dff_traces[active_neurons, :]
        
        print(f"Selected {np.sum(active_neurons)} active neurons")
        
        # 3. Normalize neural data
        dff_normalized = zscore(dff_active, axis=1)
        dff_normalized = np.nan_to_num(dff_normalized, nan=0.0)
        
        # 4. Create temporal windows
        neural_windows, behavior_windows = self._create_windows(
            dff_normalized, speed_smooth
        )
        
        return neural_windows, behavior_windows
    
    def _select_active_neurons(self, dff_traces, speed_smooth):
        """Select neurons based on activity and correlation with behavior."""
        
        correlations = []
        variances = np.var(dff_traces, axis=1)
        
        for i, neuron_trace in enumerate(dff_traces):
            trace_clean = np.nan_to_num(neuron_trace, nan=0.0)
            speed_clean = np.nan_to_num(speed_smooth, nan=0.0)
            
            if np.std(trace_clean) > 1e-8 and np.std(speed_clean) > 1e-8:
                try:
                    corr, _ = pearsonr(trace_clean, speed_clean)
                    correlations.append(abs(corr) if np.isfinite(corr) else 0)
                except:
                    correlations.append(0)
            else:
                correlations.append(0)
        
        correlations = np.array(correlations)
        
        # Combined score: variance + correlation
        variance_rank = np.argsort(variances)
        correlation_rank = np.argsort(correlations)
        
        combined_score = np.zeros(len(dff_traces))
        for i in range(len(dff_traces)):
            var_percentile = np.where(variance_rank == i)[0][0] / len(variance_rank)
            corr_percentile = np.where(correlation_rank == i)[0][0] / len(correlation_rank)
            combined_score[i] = 0.6 * var_percentile + 0.4 * corr_percentile
        
        # Select top neurons
        top_neurons = np.argsort(combined_score)[-self.min_neurons:]
        active_neurons = np.zeros(len(dff_traces), dtype=bool)
        active_neurons[top_neurons] = True
        
        return active_neurons
    
    def _create_windows(self, neural_data, behavior_data):
        """Create sliding temporal windows."""
        neural_windows = []
        behavior_windows = []
        
        min_length = min(neural_data.shape[1], len(behavior_data))
        
        for start in range(0, min_length - self.window_size + 1, self.stride):
            # Neural window
            neural_window = neural_data[:, start:start + self.window_size]
            neural_windows.append(neural_window)
            
            # Behavior features
            behavior_window = behavior_data[start:start + self.window_size]
            behavior_features = [
                np.mean(behavior_window),  # Mean speed
                np.std(behavior_window),   # Speed variability
                np.max(behavior_window),   # Peak speed
                behavior_window[-1] - behavior_window[0]  # Speed change
            ]
            behavior_windows.append(behavior_features)
        
        neural_windows = np.array(neural_windows)
        behavior_windows = np.array(behavior_windows)
        behavior_windows = np.nan_to_num(behavior_windows, nan=0.0)
        
        return neural_windows, behavior_windows
    
    def __len__(self):
        return len(self.neural_data)
    
    def __getitem__(self, idx):
        neural = self.neural_data[idx]
        behavior = self.behavior_data[idx]
        
        if self.transform:
            neural = self.transform(neural)
        
        # Apply augmentation if enabled
        if self.augment:
            neural = self._augment_neural_data(neural)
        
        return neural, behavior
    
    def _augment_neural_data(self, neural_data):
        """Apply augmentation specific to Allen Brain data."""
        if np.random.rand() < 0.5:
            # Add realistic neural noise
            noise_std = 0.05 * torch.std(neural_data)
            noise = torch.randn_like(neural_data) * noise_std
            neural_data = neural_data + noise
        
        return neural_data


class MultiSessionDataset(Dataset):
    """
    Dataset that combines data from multiple Allen Brain sessions.
    
    Useful for training on diverse neural data patterns and improving
    generalization across different recording conditions.
    """
    
    def __init__(self, session_ids=None, max_sessions=5, **kwargs):
        """
        Args:
            session_ids: list of session IDs to use
            max_sessions: maximum number of sessions to load
            **kwargs: passed to AllenBrainDataset
        """
        self.session_ids = session_ids
        self.max_sessions = max_sessions
        self.kwargs = kwargs
        
        # Load data from multiple sessions
        self._load_multi_session_data()
    
    def _load_multi_session_data(self):
        """Load and combine data from multiple sessions."""
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        
        if self.session_ids is None:
            self.session_ids = self._find_multiple_sessions()
        
        all_neural_data = []
        all_behavior_data = []
        
        for session_id in self.session_ids[:self.max_sessions]:
            try:
                print(f"Loading session {session_id}...")
                dataset = AllenBrainDataset(session_id=session_id, **self.kwargs)
                all_neural_data.append(dataset.neural_data)
                all_behavior_data.append(dataset.behavior_data)
            except Exception as e:
                print(f"Failed to load session {session_id}: {e}")
                continue
        
        if not all_neural_data:
            raise ValueError("No sessions loaded successfully")
        
        # Combine data
        self.neural_data = torch.cat(all_neural_data, dim=0)
        self.behavior_data = torch.cat(all_behavior_data, dim=0)
        
        print(f"Multi-session dataset: {len(self)} total windows from "
              f"{len(all_neural_data)} sessions")
    
    def _find_multiple_sessions(self):
        """Find multiple good sessions."""
        boc = BrainObservatoryCache()
        experiments = boc.get_ophys_experiments()
        
        good_sessions = []
        min_neurons = self.kwargs.get('min_neurons', 30)
        
        for exp in experiments[:1000]:  # Check more experiments
            if exp.get('cell_count', 0) >= min_neurons:
                try:
                    session_id = exp['id']
                    data_set = boc.get_ophys_experiment_data(session_id)
                    _, running_speed = data_set.get_running_speed()
                    
                    if len(running_speed) > 10000:
                        good_sessions.append(session_id)
                        print(f"Found session {session_id} with "
                              f"{exp['cell_count']} neurons")
                        
                        if len(good_sessions) >= self.max_sessions * 2:
                            break
                except:
                    continue
        
        return good_sessions
    
    def __len__(self):
        return len(self.neural_data)
    
    def __getitem__(self, idx):
        return self.neural_data[idx], self.behavior_data[idx]


def create_calcium_dataloaders(dataset_type='single', batch_size=32, 
                              validation_split=0.2, test_split=0.2,
                              num_workers=0, **dataset_kwargs):
    """
    Factory function to create calcium imaging dataloaders.
    
    Args:
        dataset_type: 'single', 'multi', or 'custom'
        batch_size: batch size for dataloaders
        validation_split: fraction for validation set
        test_split: fraction for test set
        num_workers: number of workers for dataloaders
        **dataset_kwargs: passed to dataset constructor
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset_info)
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create dataset
    if dataset_type == 'single':
        dataset = AllenBrainDataset(**dataset_kwargs)
    elif dataset_type == 'multi':
        dataset = MultiSessionDataset(**dataset_kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size - test_size
    
    # Random split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    dataset_info = {
        'total_samples': total_size,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'neural_shape': dataset.neural_data.shape[1:],
        'behavior_shape': dataset.behavior_data.shape[1:],
        'session_id': getattr(dataset, 'session_id', None)
    }
    
    return train_loader, val_loader, test_loader, dataset_info


if __name__ == "__main__":
    # Test basic calcium dataset
    print("Testing CalciumDataset...")
    neural_data = np.random.randn(100, 50, 100)  # 100 samples, 50 neurons, 100 timepoints
    behavior_data = np.random.randn(100, 4)  # 4 behavioral features
    
    dataset = CalciumDataset(neural_data, behavior_data, augment=True)
    print(f"Dataset length: {len(dataset)}")
    
    sample_neural, sample_behavior = dataset[0]
    print(f"Sample neural shape: {sample_neural.shape}")
    print(f"Sample behavior shape: {sample_behavior.shape}")
    
    # Test Allen Brain dataset
    print("\nTesting AllenBrainDataset...")
    try:
        allen_dataset = AllenBrainDataset(window_size=50, stride=25, min_neurons=20)
        print(f"Allen dataset length: {len(allen_dataset)}")
        
        sample_neural, sample_behavior = allen_dataset[0]
        print(f"Allen neural shape: {sample_neural.shape}")
        print(f"Allen behavior shape: {sample_behavior.shape}")
    except Exception as e:
        print(f"Allen dataset test failed: {e}")
        print("This is expected if Allen SDK is not installed or no internet connection")
    
    # Test dataloader creation
    print("\nTesting dataloader creation...")
    train_loader, val_loader, test_loader, info = create_calcium_dataloaders(
        dataset_type='single',
        session_id=501940850,  # Known session
        batch_size=16,
        window_size=50,
        min_neurons=20
    )
    
    print(f"Dataset info: {info}")
    print("Dataloader creation successful!")