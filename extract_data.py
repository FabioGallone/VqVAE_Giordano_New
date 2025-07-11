import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import warnings
warnings.filterwarnings('ignore')

print("=== VQ-VAE per Allen Brain Observatory ===\n")

# =============================================================================
# STEP 1: DOWNLOAD E PREPARAZIONE DATI
# =============================================================================

print("1. Scaricando dati dall'Allen Brain Observatory...")

# Crea cache per scaricare i dati
boc = BrainObservatoryCache()

# Scarica lista delle sessioni disponibili
experiments = boc.get_ophys_experiments()
print(f"   Sessioni disponibili: {len(experiments)}")

# Scegli UNA sessione
session_id = experiments[2]['id']
print(f"   Session ID selezionata: {session_id}")

# Scarica i dati di quella sessione
data_set = boc.get_ophys_experiment_data(session_id)
print("   Download completato!\n")

# =============================================================================
# STEP 2: ESTRAZIONE DATI
# =============================================================================

print("2. Estraendo tracce neurali e dati comportamentali...")

# Estrai le tracce ΔF/F
# data_set.get_dff_traces(),  restituisce già ΔF/F pre‑calcolato, pronto per l’analisi.
timestamps, dff_traces = data_set.get_dff_traces()
print(f"   Forma delle tracce: {dff_traces.shape}")  # (neuroni, tempo)
print(f"   Numero di neuroni: {dff_traces.shape[0]}")
print(f"   Numero di timesteps: {dff_traces.shape[1]}")

# Estrai running speed
run_ts, running_speed = data_set.get_running_speed()
print(f"   Running speed – timestamps: {run_ts.shape}, speeds: {running_speed.shape}")

# Estrai eye tracking (se disponibile)
try:
    eye_tracking = data_set.get_eye_tracking()
    print(f"   Eye tracking shape: {eye_tracking.shape}")
    has_eye_tracking = True
except:
    print("   Eye tracking non disponibile")
    eye_tracking = None
    has_eye_tracking = False

print()

# =============================================================================
# STEP 3: PREPROCESSING
# =============================================================================

print("3. Preprocessing dei dati...")

# Normalizza le tracce neurali
dff_normalized = (dff_traces - np.mean(dff_traces, axis=1, keepdims=True)) / (np.std(dff_traces, axis=1, keepdims=True) + 1e-8)

# Parametri per le finestre temporali
window_size = 100  # 100 timesteps per finestra
stride = 50       # overlap del 50%

def create_windows(data, window_size, stride):
    """Crea finestre temporali dai dati"""
    windows = []
    for start in range(0, data.shape[1] - window_size + 1, stride):
        window = data[:, start:start + window_size]
        windows.append(window)
    return np.array(windows)

# Crea finestre dalle tracce neurali
neural_windows = create_windows(dff_normalized, window_size, stride)
print(f"   Numero di finestre create: {neural_windows.shape[0]}")
print(f"   Forma di ogni finestra: {neural_windows.shape[1:]}")

# Prepara i dati comportamentali corrispondenti
behavior_windows = []
for start in range(0, len(running_speed) - window_size + 1, stride):
    behavior_mean = np.mean(running_speed[start:start + window_size])
    behavior_windows.append(behavior_mean)
behavior_windows = np.array(behavior_windows)

# Split train/test
split_idx = int(0.8 * len(neural_windows))
train_neural = neural_windows[:split_idx]
test_neural = neural_windows[split_idx:]
train_behavior = behavior_windows[:split_idx]
test_behavior = behavior_windows[split_idx:]

print(f"   Train set: {train_neural.shape[0]} finestre")
print(f"   Test set: {test_neural.shape[0]} finestre")
print()