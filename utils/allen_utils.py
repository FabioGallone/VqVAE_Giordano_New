"""
Utility functions for working with Allen Brain Observatory data.

This module provides functions for:
- Finding good recording sessions
- Preprocessing calcium imaging data
- Selecting active neurons
- Creating temporal windows
"""

import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, zscore
warnings.filterwarnings('ignore')


def find_best_sessions(min_neurons=30, max_sessions=10, min_timepoints=10000):
    """
    Find Allen Brain Observatory sessions with sufficient data quality.
    
    Args:
        min_neurons: minimum number of neurons required
        max_sessions: maximum number of sessions to return
        min_timepoints: minimum number of behavioral timepoints required
    
    Returns:
        list: session information dictionaries
    """
    try:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    except ImportError:
        print("Allen SDK not available. Install with: pip install allensdk")
        return []
    
    print(f"Searching for sessions with {min_neurons}+ neurons...")
    
    boc = BrainObservatoryCache()
    experiments = boc.get_ophys_experiments()
    
    good_sessions = []
    checked_count = 0
    
    for exp in experiments:
        cell_count = exp.get('cell_count', 0)
        
        if cell_count >= min_neurons:
            try:
                session_id = exp['id']
                checked_count += 1
                
                if checked_count % 50 == 0:
                    print(f"  Checked {checked_count} sessions, found {len(good_sessions)} good ones...")
                
                # Quick validation
                data_set = boc.get_ophys_experiment_data(session_id)
                _, running_speed = data_set.get_running_speed()
                
                if len(running_speed) >= min_timepoints:
                    session_info = {
                        'id': session_id,
                        'neurons': cell_count,
                        'cre_line': exp.get('cre_line', 'unknown'),
                        'imaging_depth': exp.get('imaging_depth', 0),
                        'area': exp.get('targeted_structure', 'unknown'),
                        'timepoints': len(running_speed)
                    }
                    good_sessions.append(session_info)
                    
                    print(f"  ✓ Session {session_id}: {cell_count} neurons, "
                          f"{len(running_speed)} timepoints, {session_info['cre_line']}")
                    
                    if len(good_sessions) >= max_sessions:
                        break
                        
            except Exception as e:
                if checked_count <= 10:  # Only show first few errors
                    print(f"  ✗ Session {exp.get('id', 'unknown')}: {str(e)[:50]}...")
                continue
        
        # Stop after checking many sessions
        if checked_count >= 1000:
            break
    
    print(f"\nFound {len(good_sessions)} suitable sessions")
    
    # Sort by number of neurons (descending)
    good_sessions.sort(key=lambda x: x['neurons'], reverse=True)
    
    return good_sessions


def load_session_data(session_id, cache_dir=None):
    """
    Load data from a specific Allen Brain Observatory session.
    
    Args:
        session_id: Allen Brain session ID
        cache_dir: directory for caching downloaded data
    
    Returns:
        tuple: (timestamps, dff_traces, run_ts, running_speed, metadata)
    """
    try:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    except ImportError:
        raise ImportError("Allen SDK not available. Install with: pip install allensdk")
    
    print(f"Loading session {session_id}...")
    
    # Setup cache
    if cache_dir:
        boc = BrainObservatoryCache(manifest_file=f'{cache_dir}/manifest.json')
    else:
        boc = BrainObservatoryCache()
    
    # Load session data
    data_set = boc.get_ophys_experiment_data(session_id)
    
    # Extract neural data
    timestamps, dff_traces = data_set.get_dff_traces()
    
    # Extract behavioral data
    run_ts, running_speed = data_set.get_running_speed()
    
    # Get metadata
    metadata = {
        'session_id': session_id,
        'num_neurons': dff_traces.shape[0],
        'num_timepoints': dff_traces.shape[1],
        'duration_seconds': timestamps[-1] - timestamps[0],
        'sampling_rate': 1.0 / np.median(np.diff(timestamps))
    }
    
    # Try to get additional metadata
    try:
        metadata['cre_line'] = data_set.get_metadata()['cre_line']
        metadata['imaging_depth'] = data_set.get_metadata()['imaging_depth']
        metadata['targeted_structure'] = data_set.get_metadata()['targeted_structure']
    except:
        pass
    
    print(f"  Loaded {metadata['num_neurons']} neurons, {metadata['num_timepoints']} timepoints")
    print(f"  Duration: {metadata['duration_seconds']:.1f} seconds")
    print(f"  Sampling rate: {metadata['sampling_rate']:.1f} Hz")
    
    return timestamps, dff_traces, run_ts, running_speed, metadata


def preprocess_calcium_data(timestamps, dff_traces, run_ts, running_speed,
                           min_neurons=30, smooth_sigma=5):
    """
    Comprehensive preprocessing of calcium imaging data.
    
    Args:
        timestamps: neural data timestamps
        dff_traces: neural ΔF/F traces (neurons x time)
        run_ts: running speed timestamps
        running_speed: running speed data
        min_neurons: minimum number of neurons to keep
        smooth_sigma: gaussian smoothing sigma for running speed
    
    Returns:
        tuple: (processed_neural_data, processed_behavior_data, selected_neurons_info)
    """
    print("Preprocessing calcium imaging data...")
    
    # 1. Align behavioral data to neural timestamps
    print("  Aligning behavioral data...")
    speed_interp = interp1d(run_ts, running_speed, 
                           bounds_error=False, fill_value=0)
    speed_aligned = speed_interp(timestamps)
    speed_smooth = gaussian_filter1d(speed_aligned, sigma=smooth_sigma)
    
    # 2. Select active neurons
    print("  Selecting active neurons...")
    active_neurons, neuron_stats = select_active_neurons(
        dff_traces, speed_smooth, min_neurons
    )
    
    dff_active = dff_traces[active_neurons, :]
    print(f"  Selected {np.sum(active_neurons)} neurons")
    
    # 3. Normalize neural data
    print("  Normalizing neural data...")
    dff_normalized = zscore(dff_active, axis=1)
    dff_normalized = np.nan_to_num(dff_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 4. Create behavior features
    print("  Creating behavior features...")
    behavior_features = create_behavior_features(speed_smooth)
    
    # 5. Quality control
    print("  Running quality control...")
    neural_qc = quality_control_neural(dff_normalized)
    behavior_qc = quality_control_behavior(behavior_features)
    
    results = {
        'neural_data': dff_normalized,
        'behavior_data': behavior_features,
        'selected_neurons': active_neurons,
        'neuron_stats': neuron_stats,
        'neural_qc': neural_qc,
        'behavior_qc': behavior_qc,
        'timestamps': timestamps
    }
    
    print("  Preprocessing completed!")
    print(f"  Final neural data shape: {dff_normalized.shape}")
    print(f"  Final behavior data shape: {behavior_features.shape}")
    
    return results


def select_active_neurons(dff_traces, behavior_data, min_neurons=30, 
                         variance_weight=0.6, correlation_weight=0.4):
    """
    Select active neurons based on variance and correlation with behavior.
    
    Args:
        dff_traces: neural traces (neurons x time)
        behavior_data: behavioral data (time,)
        min_neurons: minimum number of neurons to select
        variance_weight: weight for variance criterion
        correlation_weight: weight for correlation criterion
    
    Returns:
        tuple: (selected_neurons_mask, neuron_statistics)
    """
    num_neurons = dff_traces.shape[0]
    
    # Calculate variance for each neuron
    variances = np.var(dff_traces, axis=1)
    
    # Calculate correlation with behavior
    correlations = []
    for i, neuron_trace in enumerate(dff_traces):
        trace_clean = np.nan_to_num(neuron_trace, nan=0.0)
        behavior_clean = np.nan_to_num(behavior_data, nan=0.0)
        
        if np.std(trace_clean) > 1e-8 and np.std(behavior_clean) > 1e-8:
            try:
                corr, _ = pearsonr(trace_clean, behavior_clean)
                correlations.append(abs(corr) if np.isfinite(corr) else 0)
            except:
                correlations.append(0)
        else:
            correlations.append(0)
    
    correlations = np.array(correlations)
    
    # Combine criteria using percentile ranks
    variance_ranks = np.argsort(np.argsort(variances)) / (num_neurons - 1)
    correlation_ranks = np.argsort(np.argsort(correlations)) / (num_neurons - 1)
    
    combined_scores = (variance_weight * variance_ranks + 
                      correlation_weight * correlation_ranks)
    
    # Select top neurons
    min_neurons = min(min_neurons, num_neurons)
    top_neurons_idx = np.argsort(combined_scores)[-min_neurons:]
    
    selected_neurons = np.zeros(num_neurons, dtype=bool)
    selected_neurons[top_neurons_idx] = True
    
    # Compile statistics
    neuron_stats = {
        'total_neurons': num_neurons,
        'selected_neurons': min_neurons,
        'variance_range': (np.min(variances), np.max(variances)),
        'correlation_range': (np.min(correlations), np.max(correlations)),
        'selected_variance_mean': np.mean(variances[selected_neurons]),
        'selected_correlation_mean': np.mean(correlations[selected_neurons]),
        'variance_threshold': np.min(variances[selected_neurons]),
        'correlation_threshold': np.min(correlations[selected_neurons])
    }
    
    return selected_neurons, neuron_stats


def create_behavior_features(running_speed, window_size=None):
    """
    Create multiple behavioral features from running speed.
    
    Args:
        running_speed: running speed timeseries
        window_size: window size for local features (if None, uses global features)
    
    Returns:
        numpy array: behavioral features (time x features) or (features,) if global
    """
    if window_size is None:
        # Global features
        features = np.array([
            np.mean(running_speed),           # Mean speed
            np.std(running_speed),            # Speed variability
            np.max(running_speed),            # Peak speed
            np.sum(running_speed > 1.0)       # Time moving (arbitrary threshold)
        ])
        return features
    else:
        # Local windowed features
        num_windows = len(running_speed) - window_size + 1
        features = np.zeros((num_windows, 4))
        
        for i in range(num_windows):
            window = running_speed[i:i + window_size]
            features[i] = [
                np.mean(window),              # Mean speed
                np.std(window),               # Speed variability  
                np.max(window),               # Peak speed
                window[-1] - window[0]        # Speed change
            ]
        
        return features


def create_temporal_windows(neural_data, behavior_data, window_size=50, stride=10):
    """
    Create sliding temporal windows from neural and behavioral data.
    
    Args:
        neural_data: neural data (neurons x time)
        behavior_data: behavioral data (time,) or (time x features)
        window_size: size of temporal windows
        stride: stride for sliding windows
    
    Returns:
        tuple: (neural_windows, behavior_windows)
    """
    if behavior_data.ndim == 1:
        # Single behavioral variable - create features
        behavior_features = create_behavior_features(behavior_data, window_size)
        min_length = min(neural_data.shape[1], len(behavior_features))
        
        neural_windows = []
        behavior_windows = []
        
        for start in range(0, min_length - window_size + 1, stride):
            neural_window = neural_data[:, start:start + window_size]
            neural_windows.append(neural_window)
            
            # Behavior features for this window
            if start < len(behavior_features):
                behavior_windows.append(behavior_features[start])
            else:
                # Fallback: compute features for this specific window
                speed_window = behavior_data[start:start + window_size]
                features = create_behavior_features(speed_window)
                behavior_windows.append(features)
    else:
        # Pre-computed behavioral features
        min_length = min(neural_data.shape[1], behavior_data.shape[0])
        
        neural_windows = []
        behavior_windows = []
        
        for start in range(0, min_length - window_size + 1, stride):
            neural_window = neural_data[:, start:start + window_size]
            behavior_window = behavior_data[start:start + window_size]
            
            neural_windows.append(neural_window)
            # Use mean of behavioral features in window
            behavior_windows.append(np.mean(behavior_window, axis=0))
    
    neural_windows = np.array(neural_windows)
    behavior_windows = np.array(behavior_windows)
    
    # Clean up any NaN values
    behavior_windows = np.nan_to_num(behavior_windows, nan=0.0, posinf=0.0, neginf=0.0)
    
    return neural_windows, behavior_windows


def quality_control_neural(neural_data, max_zeros_fraction=0.8, min_variance=1e-6):
    """
    Perform quality control on neural data.
    
    Args:
        neural_data: neural data (neurons x time)
        max_zeros_fraction: maximum fraction of zeros allowed per neuron
        min_variance: minimum variance required per neuron
    
    Returns:
        dict: quality control results
    """
    num_neurons, num_timepoints = neural_data.shape
    
    # Check for neurons with too many zeros
    zeros_fraction = np.mean(neural_data == 0, axis=1)
    high_zeros = np.sum(zeros_fraction > max_zeros_fraction)
    
    # Check for low-variance neurons
    variances = np.var(neural_data, axis=1)
    low_variance = np.sum(variances < min_variance)
    
    # Check for NaN/inf values
    nan_count = np.sum(~np.isfinite(neural_data))
    
    # Dynamic range
    data_range = np.max(neural_data) - np.min(neural_data)
    
    qc_results = {
        'total_neurons': num_neurons,
        'total_timepoints': num_timepoints,
        'high_zeros_neurons': high_zeros,
        'low_variance_neurons': low_variance,
        'nan_inf_count': nan_count,
        'data_range': data_range,
        'mean_variance': np.mean(variances),
        'median_zeros_fraction': np.median(zeros_fraction),
        'passed_qc': (high_zeros == 0 and low_variance == 0 and nan_count == 0)
    }
    
    return qc_results


def quality_control_behavior(behavior_data, min_variance=1e-6):
    """
    Perform quality control on behavioral data.
    
    Args:
        behavior_data: behavioral data (time x features) or (features,)
        min_variance: minimum variance required per feature
    
    Returns:
        dict: quality control results
    """
    if behavior_data.ndim == 1:
        behavior_data = behavior_data.reshape(1, -1)
    
    num_samples, num_features = behavior_data.shape
    
    # Check variance for each feature
    variances = np.var(behavior_data, axis=0)
    low_variance_features = np.sum(variances < min_variance)
    
    # Check for NaN/inf values
    nan_count = np.sum(~np.isfinite(behavior_data))
    
    # Check feature ranges
    feature_ranges = np.max(behavior_data, axis=0) - np.min(behavior_data, axis=0)
    
    qc_results = {
        'num_samples': num_samples,
        'num_features': num_features,
        'low_variance_features': low_variance_features,
        'nan_inf_count': nan_count,
        'feature_ranges': feature_ranges,
        'mean_variance': np.mean(variances),
        'passed_qc': (low_variance_features == 0 and nan_count == 0)
    }
    
    return qc_results


def save_session_id(session_id, filename='best_session_id.txt'):
    """Save session ID to file for future use."""
    with open(filename, 'w') as f:
        f.write(str(session_id))
    print(f"Session ID {session_id} saved to {filename}")


def load_session_id(filename='best_session_id.txt'):
    """Load session ID from file."""
    try:
        with open(filename, 'r') as f:
            session_id = int(f.read().strip())
        print(f"Loaded session ID {session_id} from {filename}")
        return session_id
    except:
        print(f"Could not load session ID from {filename}")
        return None


if __name__ == "__main__":
    # Test the utility functions
    print("Testing Allen Brain utilities...")
    
    # Test session finding
    try:
        sessions = find_best_sessions(min_neurons=20, max_sessions=3)
        if sessions:
            print(f"Found {len(sessions)} sessions")
            
            # Test loading data from first session
            session_id = sessions[0]['id']
            timestamps, dff_traces, run_ts, running_speed, metadata = load_session_data(session_id)
            
            # Test preprocessing
            results = preprocess_calcium_data(
                timestamps, dff_traces, run_ts, running_speed, min_neurons=20
            )
            
            print("Preprocessing results:")
            print(f"  Neural data: {results['neural_data'].shape}")
            print(f"  Behavior data: {results['behavior_data'].shape}")
            print(f"  Neural QC passed: {results['neural_qc']['passed_qc']}")
            print(f"  Behavior QC passed: {results['behavior_qc']['passed_qc']}")
            
            # Test windowing
            neural_windows, behavior_windows = create_temporal_windows(
                results['neural_data'], running_speed, window_size=50, stride=25
            )
            
            print(f"Created {len(neural_windows)} windows")
            print(f"Neural window shape: {neural_windows.shape}")
            print(f"Behavior window shape: {behavior_windows.shape}")
            
        else:
            print("No suitable sessions found")
            
    except ImportError:
        print("Allen SDK not available - skipping tests")
    except Exception as e:
        print(f"Test failed: {e}")
    
    print("Utility test completed!")