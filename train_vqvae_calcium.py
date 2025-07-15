import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import warnings
import os

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

print("=== VQ-VAE per Allen Brain Observatory ===\n")

# =============================================================================
# STEP 1: SELEZIONE SESSIONE CON PIÙ NEURONI
# =============================================================================

print("1. Cercando sessioni con molti neuroni...")

# Crea cache per scaricare i dati
boc = BrainObservatoryCache()

# Scarica lista delle sessioni disponibili
experiments = boc.get_ophys_experiments()

# Prova a caricare sessione salvata o cercane una nuova
try:
    with open('best_session_id.txt', 'r') as f:
        session_id = int(f.read().strip())
    print(f"   Usando sessione salvata: {session_id}")
except:
    print("   Cercando sessioni con dati adeguati...")
    
    # Abbassa soglia neuroni a 30 (più realistico)
    min_neurons = 30
    good_sessions = []
    
    # Cerca tra le prime 200 sessioni
    for i, exp in enumerate(experiments[:10000]):
        if exp.get('cell_count', 0) >= min_neurons:
            try:
                session_id = exp['id']
                print(f"   Verificando sessione {session_id} ({exp['cell_count']} neuroni)...")
                
                # Verifica velocemente se ha dati
                data_set = boc.get_ophys_experiment_data(session_id)
                _, running_speed = data_set.get_running_speed()
                
                if len(running_speed) > 10000:  # Almeno 10k timesteps
                    good_sessions.append({
                        'id': session_id,
                        'neurons': exp['cell_count'],
                        'cre_line': exp.get('cre_line', 'unknown')
                    })
                    print(f"     ✓ OK! {exp['cell_count']} neuroni, {len(running_speed)} timesteps")
                    
                    if len(good_sessions) >= 3:  # Trova almeno 3 sessioni
                        break
            except Exception as e:
                print(f"     ✗ Errore: {str(e)[:40]}...")
                continue
    
    if not good_sessions:
        # Se ancora niente, usa una sessione nota che funziona
        print("   Usando sessione di default...")
        session_id = 501940850 
    else:
        # Scegli la sessione con più neuroni
        best_session = max(good_sessions, key=lambda x: x['neurons'])
        session_id = best_session['id']
        print(f"\n   Sessione selezionata: {session_id} con {best_session['neurons']} neuroni")

# Scarica i dati
data_set = boc.get_ophys_experiment_data(session_id)
print("   Download completato!\n")

# =============================================================================
# STEP 2: ESTRAZIONE DATI MIGLIORATA
# =============================================================================

print("2. Estraendo tracce neurali e dati comportamentali...")

# Estrai le tracce ΔF/F
timestamps, dff_traces = data_set.get_dff_traces()
print(f"   Forma delle tracce: {dff_traces.shape}")
print(f"   Numero di neuroni: {dff_traces.shape[0]}")
print(f"   Numero di timesteps: {dff_traces.shape[1]}")

# Estrai running speed
run_ts, running_speed = data_set.get_running_speed()
print(f"   Running speed: {len(running_speed)} timesteps")

# Opzionale: prova a ottenere informazioni stimoli
try:
    stim_epochs = data_set.get_stimulus_epoch_table()
    print(f"   Epoche stimoli disponibili: {len(stim_epochs)}")
except:
    print("   Info stimoli non disponibili per questa sessione")

# =============================================================================
# STEP 3: PREPROCESSING AVANZATO
# =============================================================================

print("\n3. Preprocessing avanzato dei dati...")

speed_interp = interp1d(run_ts, running_speed, bounds_error=False, fill_value=0)
speed_aligned = speed_interp(timestamps)
speed_smooth = gaussian_filter1d(speed_aligned, sigma=5)


# Calcola correlazioni per ogni neurone
print("   Calcolando correlazioni neuroni-movimento...")
correlations = []
for i, neuron_trace in enumerate(dff_traces):
    trace_clean = np.nan_to_num(neuron_trace, nan=0.0, posinf=0.0, neginf=0.0)
    speed_clean = np.nan_to_num(speed_smooth, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.std(trace_clean) > 1e-8 and np.std(speed_clean) > 1e-8:
        try:
            corr, _ = pearsonr(trace_clean, speed_clean)
            if np.isfinite(corr):
                correlations.append(abs(corr))
            else:
                correlations.append(0)
        except:
            correlations.append(0)
    else:
        correlations.append(0)

correlations = np.array(correlations)

# Seleziona neuroni basandosi su correlazione + varianza
min_neurons_to_keep = 30

# Combina criteri: alta varianza E correlazione con movimento
neuron_variance = np.var(dff_traces, axis=1)
variance_rank = np.argsort(neuron_variance)
correlation_rank = np.argsort(correlations)

# Score combinato (50% varianza, 50% correlazione)
combined_score = np.zeros(len(dff_traces))
for i in range(len(dff_traces)):
    variance_percentile = np.where(variance_rank == i)[0][0] / len(variance_rank)
    correlation_percentile = np.where(correlation_rank == i)[0][0] / len(correlation_rank)
    combined_score[i] = 0.5 * variance_percentile + 0.5 * correlation_percentile

# Seleziona top neuroni
top_neurons = np.argsort(combined_score)[-min_neurons_to_keep:]
active_neurons = np.zeros(len(dff_traces), dtype=bool)
active_neurons[top_neurons] = True

dff_active = dff_traces[active_neurons, :]
selected_correlations = correlations[active_neurons]

print(f"   Neuroni selezionati: {np.sum(active_neurons)} su {len(active_neurons)}")
print(f"   Correlazioni motorie: min={selected_correlations.min():.3f}, "
      f"max={selected_correlations.max():.3f}, mean={selected_correlations.mean():.3f}")

# Normalizza con z-score robusto
from scipy import stats
dff_normalized = stats.zscore(dff_active, axis=1)

# Parametri ottimizzati per finestre temporali
window_size = 50  # Ridotto per catturare dinamiche più rapide
stride = 10       # Più overlap per più training data

def create_windows_aligned(neural_data, behavior_data, window_size, stride):
    """Crea finestre allineate tra dati neurali e comportamentali"""
    windows_neural = []
    windows_behavior = []
    
    # Assicura che i dati siano allineati temporalmente
    min_length = min(neural_data.shape[1], len(behavior_data))
    
    for start in range(0, min_length - window_size + 1, stride):
        # Finestra neurale
        neural_window = neural_data[:, start:start + window_size]
        windows_neural.append(neural_window)
        
        # Comportamento: usa sia media che varianza per catturare più informazione
        behavior_window = behavior_data[start:start + window_size]
        behavior_features = [
            np.mean(behavior_window),  # Media velocità
            np.std(behavior_window),   # Variabilità
            np.max(behavior_window),   # Picco velocità
            behavior_window[-1] - behavior_window[0]  # Cambio velocità
        ]
        windows_behavior.append(behavior_features)
    
    return np.array(windows_neural), np.array(windows_behavior)

# Crea finestre con allineamento migliorato - USA speed_smooth!
neural_windows, behavior_windows = create_windows_aligned(
    dff_normalized, speed_smooth, window_size, stride
)
behavior_windows = np.nan_to_num(behavior_windows, nan=0.0, posinf=0.0, neginf=0.0)
print(f"   Behavior windows stats: min={behavior_windows.min():.3f}, max={behavior_windows.max():.3f}")
print(f"   NaN count: {np.isnan(behavior_windows).sum()}")

print(f"   Numero di finestre create: {neural_windows.shape[0]}")
print(f"   Forma di ogni finestra: {neural_windows.shape[1:]}")
print(f"   Features comportamentali per finestra: {behavior_windows.shape[1]}")

# Split train/validation/test (60/20/20)
n_samples = len(neural_windows)
train_idx = int(0.6 * n_samples)
val_idx = int(0.8 * n_samples)

train_neural = neural_windows[:train_idx]
val_neural = neural_windows[train_idx:val_idx]
test_neural = neural_windows[val_idx:]

train_behavior = behavior_windows[:train_idx]
val_behavior = behavior_windows[train_idx:val_idx]
test_behavior = behavior_windows[val_idx:]

print(f"   Train set: {train_neural.shape[0]} finestre")
print(f"   Validation set: {val_neural.shape[0]} finestre")
print(f"   Test set: {test_neural.shape[0]} finestre")

# =============================================================================
# DATASET CLASS MIGLIORATA
# =============================================================================

class CalciumDataset(Dataset):
    def __init__(self, neural_data, behavior_data, augment=False):
        self.neural_data = torch.FloatTensor(neural_data)
        self.behavior_data = torch.FloatTensor(behavior_data)
        self.augment = augment
        
    def __len__(self):
        return len(self.neural_data)
    
    def __getitem__(self, idx):
        neural = self.neural_data[idx]
        behavior = self.behavior_data[idx]
        
        # Data augmentation durante training
        if self.augment and np.random.rand() > 0.5:
            # Aggiungi rumore gaussiano leggero
            noise = torch.randn_like(neural) * 0.1
            neural = neural + noise
            
        return neural, behavior

# =============================================================================
# ARCHITETTURA MIGLIORATA
# =============================================================================

class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ImprovedResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, num_residual_hiddens),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ImprovedEncoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ImprovedEncoder, self).__init__()
        
        # Architettura più profonda e graduale
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                  out_channels=num_hiddens//4,
                                  kernel_size=7,
                                  stride=1, padding=3)
        self._conv_2 = nn.Conv1d(in_channels=num_hiddens//4,
                                  out_channels=num_hiddens//2,
                                  kernel_size=5,
                                  stride=2, padding=2)
        self._conv_3 = nn.Conv1d(in_channels=num_hiddens//2,
                                  out_channels=num_hiddens,
                                  kernel_size=3,
                                  stride=2, padding=1)
        
        self._residual_stack = nn.ModuleList([
            ImprovedResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])

    def forward(self, inputs):
        x = F.relu(self._conv_1(inputs))
        x = F.relu(self._conv_2(x))
        x = self._conv_3(x)
        
        for block in self._residual_stack:
            x = block(x)
            
        return x

# =============================================================================
# VECTOR QUANTIZER MIGLIORATO
# =============================================================================

class ImprovedVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super(ImprovedVectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        
        # EMA per aggiornamento codebook
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._codebook_usage = torch.zeros(num_embeddings)

    def forward(self, inputs):
        # Convert inputs from BCL -> BLC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Update codebook usage
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
        
        # Convert quantized from BLC -> BCL
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings

# =============================================================================
# GROUPED RESIDUAL VQ (dal paper)
# =============================================================================

class GroupedResidualVQ(nn.Module):
    """Implementa Grouped Residual VQ per migliore utilizzo del codebook"""
    def __init__(self, num_embeddings, embedding_dim, num_groups=4, num_residual=2, commitment_cost=0.25):
        super(GroupedResidualVQ, self).__init__()
        
        self.num_groups = num_groups
        self.num_residual = num_residual
        self.group_dim = embedding_dim // num_groups
        
        # Crea quantizzatori per ogni gruppo e livello residuale
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
        # BCL -> BLC
        inputs = inputs.permute(0, 2, 1).contiguous()
        B, L, D = inputs.shape
        
        # Split into groups
        x_groups = inputs.view(B, L, self.num_groups, self.group_dim)
        
        all_losses = []
        all_perplexities = []
        quantized_groups = []
        all_encodings = []
        
        for g in range(self.num_groups):
            x_g = x_groups[:, :, g, :]
            quantized_g = torch.zeros_like(x_g)
            
            # Residual quantization
            residual = x_g
            for r in range(self.num_residual):
                loss_r, quantized_r, perplexity_r, encodings_r = self.quantizers[r][g](
                    residual.permute(0, 2, 1)  # BLC -> BCL for quantizer
                )
                quantized_r = quantized_r.permute(0, 2, 1)  # BCL -> BLC
                
                quantized_g = quantized_g + quantized_r
                residual = residual - quantized_r.detach()
                
                all_losses.append(loss_r)
                all_perplexities.append(perplexity_r)
                
            quantized_groups.append(quantized_g)
        
        # Concatenate groups
        quantized = torch.cat(quantized_groups, dim=-1)
        
        # Average metrics
        loss = sum(all_losses) / len(all_losses)
        perplexity = sum(all_perplexities) / len(all_perplexities)
        
        # Fake encodings for compatibility
        encodings = torch.zeros(B * L, self._num_embeddings, device=inputs.device)
        
        # BLC -> BCL
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings

# =============================================================================
# MODELLO VQ-VAE COMPLETO 
# =============================================================================

class ImprovedCalciumVQVAE(nn.Module):
    def __init__(self, num_neurons, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, quantizer_type='improved_vq',
                 dropout_rate=0.1, behavior_dim=4):
        super(ImprovedCalciumVQVAE, self).__init__()
        
        self._encoder = ImprovedEncoder(num_neurons, num_hiddens,
                                        num_residual_layers, num_residual_hiddens)
        
        self._pre_vq_conv = nn.Sequential(
            nn.Conv1d(in_channels=num_hiddens, 
                      out_channels=embedding_dim,
                      kernel_size=1, 
                      stride=1),
            nn.Dropout1d(dropout_rate)
        )
        
        if quantizer_type == 'improved_vq':
            self._vq_vae = ImprovedVectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        elif quantizer_type == 'grouped_rvq':
            self._vq_vae = GroupedResidualVQ(num_embeddings, embedding_dim, commitment_cost=commitment_cost)
        else:
            raise ValueError(f"Unknown quantizer type: {quantizer_type}")
        
        self.quantizer_type = quantizer_type
            
        # Decoder simmetrico con skip connections
        self._decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, num_hiddens,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(num_hiddens, num_hiddens//2,
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(num_hiddens//2, num_neurons,
                      kernel_size=7, stride=1, padding=3)
        )
        
        # NUOVO: Behavior prediction head
        self.behavior_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, behavior_dim)
        )

    def forward(self, x, return_behavior_pred=True):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, encodings = self._vq_vae(z)
        
        # Aggiusta dimensioni per decoder
        if quantized.shape[2] != x.shape[2] // 4:
            quantized = F.interpolate(quantized, size=x.shape[2] // 4, mode='linear', align_corners=False)
        
        x_recon = self._decoder(quantized)
        
        # Aggiusta dimensione finale se necessario
        if x_recon.shape[2] != x.shape[2]:
            x_recon = F.interpolate(x_recon, size=x.shape[2], mode='linear', align_corners=False)

        # NUOVO: Behavior prediction
        behavior_pred = None
        if return_behavior_pred:
            z_pooled = quantized.mean(dim=2)
            behavior_pred = self.behavior_head(z_pooled)

        return loss, x_recon, perplexity, quantized, encodings, behavior_pred
    
# =============================================================================
# DECODIFICATORE COMPORTAMENTALE NON LINEARE
# =============================================================================

class BehaviorDecoder(nn.Module):
    """Decodificatore neurale per predire il comportamento"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=4, dropout=0.2):
        super(BehaviorDecoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# =============================================================================
# TRAINING MIGLIORATO
# =============================================================================
def train_model_improved(model, train_loader, val_loader, test_loader, 
                        num_epochs=150, learning_rate=3e-4, device='cuda',
                        behavior_weight=0.5):  # AGGIUNGI behavior_weight
    
    # Optimizer con learning rate scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    train_res_recon_error = []
    train_res_perplexity = []
    val_res_recon_error = []
    val_res_perplexity = []
    val_behavior_r2 = []  # NUOVO
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_recon_error = 0
        train_perplexity = 0
        
        for batch_idx, (data, behavior) in enumerate(train_loader):
            data = data.to(device)
            behavior = behavior.to(device)  # NUOVO
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity, quantized, _, behavior_pred = model(data)
            recon_error = F.mse_loss(data_recon, data)
            
            # NUOVO: Behavior loss
            behavior_loss = F.mse_loss(behavior_pred, behavior)
            
            # Loss totale multi-task
            loss = recon_error + 0.25 * vq_loss + behavior_weight * behavior_loss
            
            train_recon_error += recon_error.item()
            train_perplexity += perplexity.item()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
        
        # Validation
        model.eval()
        val_recon_error = 0
        val_perplexity = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, behavior in val_loader:
                data = data.to(device)
                behavior = behavior.to(device)
                vq_loss, data_recon, perplexity, _, _, behavior_pred = model(data)
                val_recon_error += F.mse_loss(data_recon, data).item()
                val_perplexity += perplexity.item()
                
                # Salva per R2
                val_predictions.append(behavior_pred.cpu().numpy())
                val_targets.append(behavior.cpu().numpy())
        
        # Calcola R2 per validazione
        val_predictions = np.vstack(val_predictions)
        val_targets = np.vstack(val_targets)
        from sklearn.metrics import r2_score
        valid_mask = np.isfinite(val_targets[:, 0]) & np.isfinite(val_predictions[:, 0])
        if valid_mask.sum() > 0:
            r2_val = r2_score(val_targets[valid_mask, 0], val_predictions[valid_mask, 0])
        else:
            r2_val = 0.0
        
        # Calculate averages
        avg_train_recon = train_recon_error / len(train_loader)
        avg_train_perplexity = train_perplexity / len(train_loader)
        avg_val_recon = val_recon_error / len(val_loader)
        avg_val_perplexity = val_perplexity / len(val_loader)
        
        train_res_recon_error.append(avg_train_recon)
        train_res_perplexity.append(avg_train_perplexity)
        val_res_recon_error.append(avg_val_recon)
        val_res_perplexity.append(avg_val_perplexity)
        val_behavior_r2.append(r2_val)
        
        # Learning rate scheduling
        scheduler.step(avg_val_recon)
        
        # Early stopping
        if avg_val_recon < best_val_loss:
            best_val_loss = avg_val_recon
            patience_counter = 0
            # Salva il miglior modello
            torch.save(model.state_dict(), f'best_model_{model.quantizer_type}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] '
                  f'Train Recon: {avg_train_recon:.4f}, '
                  f'Train Perplexity: {avg_train_perplexity:.2f}, '
                  f'Val Recon: {avg_val_recon:.4f}, '
                  f'Val R²: {r2_val:.3f}')  # MOSTRA R2
    
    # Carica il miglior modello
    model.load_state_dict(torch.load(f'best_model_{model.quantizer_type}.pt'))
    
    # Test final
    test_recon_error = []
    test_perplexity = []
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            vq_loss, data_recon, perplexity, _, _, _ = model(data)
            test_recon_error.append(F.mse_loss(data_recon, data).item())
            test_perplexity.append(perplexity.item())
    
    return (train_res_recon_error, train_res_perplexity, 
            val_res_recon_error, val_res_perplexity,
            test_recon_error, test_perplexity)
# =============================================================================
# EVALUATION MIGLIORATA
# =============================================================================

def evaluate_behavior_prediction(model, test_loader, device='cuda'):
    """Valuta direttamente le predizioni del behavior head integrato"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, behavior in test_loader:
            data = data.to(device)
            _, _, _, _, _, behavior_pred = model(data)
            
            # Controllo NaN
            if torch.isnan(behavior_pred).any():
                print("Warning: NaN detected in behavior predictions")
                behavior_pred = torch.nan_to_num(behavior_pred, nan=0.0)
            
            all_predictions.append(behavior_pred.cpu().numpy())
            all_targets.append(behavior.numpy())
    
    all_predictions = np.nan_to_num(all_predictions, nan=0.0, posinf=0.0, neginf=0.0)
    all_targets = np.nan_to_num(all_targets, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calcola R² per ogni feature
    feature_names = ['Mean Speed', 'Speed Std', 'Max Speed', 'Speed Change']
    r2_scores = []
    
    for i in range(all_targets.shape[1]):
        r2 = r2_score(all_targets[:, i], all_predictions[:, i])
        r2_scores.append(r2)
        print(f"  {feature_names[i]}: R² = {r2:.3f}")
    
    return r2_scores
# =============================================================================
# MAIN EXPERIMENT MIGLIORATO
# =============================================================================

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Parametri adattivi basati sul numero di neuroni
    num_neurons = train_neural.shape[1]
    
    # Scala i parametri in base al numero di neuroni
    if num_neurons < 50:
        num_hiddens = 128  # Rete più piccola per pochi neuroni
        num_embeddings = 512
        embedding_dim = 64
        batch_size = 32
    else:
        num_hiddens = 256
        num_embeddings = 1024
        embedding_dim = 128
        batch_size = 64
    
    num_residual_layers = 3
    num_residual_hiddens = min(64, num_hiddens // 2)
    commitment_cost = 0.25
    
    # Training parameters
    num_epochs = 150
    learning_rate = 3e-4
    
    print(f"\nParametri adattati per {num_neurons} neuroni:")
    print(f"  Hidden dimensions: {num_hiddens}")
    print(f"  Embeddings: {num_embeddings}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    # Crea datasets con augmentation
    train_dataset = CalciumDataset(train_neural, train_behavior, augment=True)
    val_dataset = CalciumDataset(val_neural, val_behavior, augment=False)
    test_dataset = CalciumDataset(test_neural, test_behavior, augment=False)
    
    # Su Windows, usa num_workers=0 per evitare problemi di multiprocessing
    num_workers = 0 if os.name == 'nt' else 2
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    results = {}
    
    # =========================
    # 1. TRAIN IMPROVED VQ-VAE
    # =========================
    print("\n" + "="*60)
    print("TRAINING VQ-VAE")
    print("="*60)
    
    vqvae_model = ImprovedCalciumVQVAE(
        num_neurons, num_hiddens, num_residual_layers, num_residual_hiddens,
        num_embeddings, embedding_dim, commitment_cost, quantizer_type='improved_vq', behavior_dim=4
    ).to(device)
    
    print(f"Parametri modello: {sum(p.numel() for p in vqvae_model.parameters() if p.requires_grad):,}")
    
    (vq_train_recon, vq_train_perp, vq_val_recon, vq_val_perp, 
     vq_test_recon, vq_test_perp) = train_model_improved(
        vqvae_model, train_loader, val_loader, test_loader, 
        num_epochs, learning_rate, device, behavior_weight=0.5
    )
    
    # Train behavior decoder
    print("   Valutando predizioni comportamentali...")
    vq_r2_scores = evaluate_behavior_prediction(vqvae_model, test_loader, device)

    
    # Calculate codebook usage
    if hasattr(vqvae_model._vq_vae, '_codebook_usage'):
        vq_usage = (vqvae_model._vq_vae._codebook_usage > 0.01).sum().item()
        vq_percent_used = (vq_usage / num_embeddings) * 100
    else:
        vq_percent_used = 100.0  # Assume full usage for GRQ
    
    results['improved_vq'] = {
        'final_recon_mse': np.mean(vq_test_recon),
        'final_perplexity': np.mean(vq_test_perp),
        'codebook_usage': vq_percent_used,
        'behavior_r2_scores': vq_r2_scores,
        'mean_behavior_r2': np.mean(vq_r2_scores)
    }
    
    print(f"\nImproved VQ-VAE Results:")
    print(f"  Test Reconstruction MSE: {np.mean(vq_test_recon):.4f}")
    print(f"  Codebook Usage: {vq_percent_used:.1f}%")
    print(f"  Behavior R² scores: {[f'{r2:.3f}' for r2 in vq_r2_scores]}")
    print(f"  Mean Behavior R²: {np.mean(vq_r2_scores):.3f}")
    
    # =========================
    # 2. TRAIN GROUPED RVQ
    # =========================
    print("\n" + "="*60)
    print("TRAINING GROUPED RESIDUAL VQ")
    print("="*60)
    
    grvq_model = ImprovedCalciumVQVAE(
        num_neurons, num_hiddens, num_residual_layers, num_residual_hiddens,
        num_embeddings, embedding_dim, commitment_cost, quantizer_type='grouped_rvq', behavior_dim=4
    ).to(device)
    
    (grvq_train_recon, grvq_train_perp, grvq_val_recon, grvq_val_perp,
     grvq_test_recon, grvq_test_perp) = train_model_improved(
        grvq_model, train_loader, val_loader, test_loader,
        num_epochs, learning_rate, device, behavior_weight=0.5
    )
    
    # Train behavior decoder
    print("   Valutando predizioni comportamentali...")
    grvq_r2_scores = evaluate_behavior_prediction(grvq_model, test_loader, device)

    
    results['grouped_rvq'] = {
        'final_recon_mse': np.mean(grvq_test_recon),
        'final_perplexity': np.mean(grvq_test_perp),
        'behavior_r2_scores': grvq_r2_scores,
        'mean_behavior_r2': np.mean(grvq_r2_scores)
    }
    
    print(f"\nGrouped RVQ Results:")
    print(f"  Test Reconstruction MSE: {np.mean(grvq_test_recon):.4f}")
    print(f"  Behavior R² scores: {[f'{r2:.3f}' for r2 in grvq_r2_scores]}")
    print(f"  Mean Behavior R²: {np.mean(grvq_r2_scores):.3f}")
    
    # =========================
    # 3. VISUALIZZAZIONI
    # =========================
    print("\n" + "="*60)
    print("CREANDO VISUALIZZAZIONI")
    print("="*60)
    
    # Figura 1: Training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Reconstruction error - Training
    axes[0, 0].plot(vq_train_recon, label='Improved VQ', alpha=0.7)
    axes[0, 0].plot(grvq_train_recon, label='Grouped RVQ', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Reconstruction MSE')
    axes[0, 0].set_title('Training Reconstruction Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction error - Validation
    axes[0, 1].plot(vq_val_recon, label='Improved VQ', alpha=0.7)
    axes[0, 1].plot(grvq_val_recon, label='Grouped RVQ', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction MSE')
    axes[0, 1].set_title('Validation Reconstruction Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Perplexity
    axes[0, 2].plot(vq_val_perp, label='Improved VQ', alpha=0.7)
    axes[0, 2].plot(grvq_val_perp, label='Grouped RVQ', alpha=0.7)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Perplexity')
    axes[0, 2].set_title('Codebook Utilization')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Behavior decoding comparison
    behavior_names = ['Mean Speed', 'Speed Variance', 'Max Speed', 'Speed Change']
    x = np.arange(len(behavior_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, vq_r2_scores, width, label='Improved VQ', alpha=0.7)
    axes[1, 0].bar(x + width/2, grvq_r2_scores, width, label='Grouped RVQ', alpha=0.7)
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Behavior Decoding Performance')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(behavior_names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Overall comparison
    metrics_data = [
        ['Test Recon MSE', f"{results['improved_vq']['final_recon_mse']:.4f}", 
         f"{results['grouped_rvq']['final_recon_mse']:.4f}"],
        ['Mean Behavior R²', f"{results['improved_vq']['mean_behavior_r2']:.3f}", 
         f"{results['grouped_rvq']['mean_behavior_r2']:.3f}"],
        ['Codebook Usage %', f"{results['improved_vq']['codebook_usage']:.1f}", 'N/A'],
        ['Final Perplexity', f"{results['improved_vq']['final_perplexity']:.1f}", 
         f"{results['grouped_rvq']['final_perplexity']:.1f}"]
    ]
    
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=metrics_data,
                            colLabels=['Metric', 'Improved VQ', 'Grouped RVQ'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Final Metrics Comparison', pad=20)
    
    # Test set distribution
    axes[1, 2].hist(vq_test_recon, bins=30, alpha=0.5, label='Improved VQ', density=True)
    axes[1, 2].hist(grvq_test_recon, bins=30, alpha=0.5, label='Grouped RVQ', density=True)
    axes[1, 2].set_xlabel('Reconstruction MSE')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Test Error Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_vqvae_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # =========================
    # 4. RECONSTRUCTION EXAMPLES
    # =========================
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    # Get sample batch
    sample_data, sample_behavior = next(iter(test_loader))
    sample_data = sample_data.to(device)
    
    with torch.no_grad():
        _, vq_recon, _, _, _ = vqvae_model(sample_data)
        _, grvq_recon, _, _, _ = grvq_model(sample_data)
    
    # Convert to numpy
    sample_data = sample_data.cpu().numpy()
    vq_recon = vq_recon.cpu().numpy()
    grvq_recon = grvq_recon.cpu().numpy()
    
    # Plot 3 examples
    for i in range(3):
        # Select neurons with high variance for visualization
        neuron_vars = np.var(sample_data[i], axis=1)
        top_neurons = np.argsort(neuron_vars)[-15:]  # Top 15 most active
        
        # Original
        im0 = axes[i, 0].imshow(sample_data[i, top_neurons, :], 
                                aspect='auto', cmap='viridis', interpolation='nearest')
        axes[i, 0].set_ylabel(f'Sample {i+1}\nNeurons')
        if i == 0:
            axes[i, 0].set_title('Original')
        
        # Improved VQ reconstruction
        im1 = axes[i, 1].imshow(vq_recon[i, top_neurons, :], 
                                aspect='auto', cmap='viridis', interpolation='nearest')
        if i == 0:
            axes[i, 1].set_title('Improved VQ-VAE')
        
        # Grouped RVQ reconstruction
        im2 = axes[i, 2].imshow(grvq_recon[i, top_neurons, :], 
                                aspect='auto', cmap='viridis', interpolation='nearest')
        if i == 0:
            axes[i, 2].set_title('Grouped RVQ')
        
        # Add colorbars
        if i == 0:
            for j, im in enumerate([im0, im1, im2]):
                plt.colorbar(im, ax=axes[i, j], pad=0.02)
    
    for ax in axes.flat:
        ax.set_xlabel('Time (frames)')
    
    plt.tight_layout()
    plt.savefig('improved_reconstruction_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # =========================
    # 5. LATENT SPACE ANALYSIS
    # =========================
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    print("\nAnalyzing latent spaces...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract latent representations
    vq_latents = []
    grvq_latents = []
    behaviors = []
    
    with torch.no_grad():
        for data, behavior in test_loader:
            data = data.to(device)
            
            # VQ latents
            z_vq = vqvae_model._encoder(data)
            z_vq = vqvae_model._pre_vq_conv(z_vq)
            vq_latents.append(z_vq.mean(dim=2).cpu().numpy())  # Average over time
            
            # GRVQ latents
            z_grvq = grvq_model._encoder(data)
            z_grvq = grvq_model._pre_vq_conv(z_grvq)
            grvq_latents.append(z_grvq.mean(dim=2).cpu().numpy())
            
            behaviors.append(behavior[:, 0].numpy())  # Use mean speed for coloring
    
    vq_latents = np.vstack(vq_latents)
    grvq_latents = np.vstack(grvq_latents)
    behaviors = np.hstack(behaviors)
    
    # PCA visualization
    pca = PCA(n_components=2)
    vq_pca = pca.fit_transform(vq_latents)
    grvq_pca = pca.fit_transform(grvq_latents)
    
    # Plot PCA - VQ
    scatter1 = axes[0, 0].scatter(vq_pca[:, 0], vq_pca[:, 1], 
                                  c=behaviors, cmap='viridis', s=20, alpha=0.6)
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title('Improved VQ Latent Space (PCA)')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Mean Speed')
    
    # Plot PCA - GRVQ
    scatter2 = axes[0, 1].scatter(grvq_pca[:, 0], grvq_pca[:, 1], 
                                  c=behaviors, cmap='viridis', s=20, alpha=0.6)
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].set_title('Grouped RVQ Latent Space (PCA)')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Mean Speed')
    
    # t-SNE visualization (subset for speed)
    subset_idx = np.random.choice(len(vq_latents), min(1000, len(vq_latents)), replace=False)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    vq_tsne = tsne.fit_transform(vq_latents[subset_idx])
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    grvq_tsne = tsne.fit_transform(grvq_latents[subset_idx])
    
    # Plot t-SNE - VQ
    scatter3 = axes[1, 0].scatter(vq_tsne[:, 0], vq_tsne[:, 1], 
                                  c=behaviors[subset_idx], cmap='viridis', s=20, alpha=0.6)
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    axes[1, 0].set_title('Improved VQ Latent Space (t-SNE)')
    plt.colorbar(scatter3, ax=axes[1, 0], label='Mean Speed')
    
    # Plot t-SNE - GRVQ
    scatter4 = axes[1, 1].scatter(grvq_tsne[:, 0], grvq_tsne[:, 1], 
                                  c=behaviors[subset_idx], cmap='viridis', s=20, alpha=0.6)
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    axes[1, 1].set_title('Grouped RVQ Latent Space (t-SNE)')
    plt.colorbar(scatter4, ax=axes[1, 1], label='Mean Speed')
    
    plt.tight_layout()
    plt.savefig('latent_space_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nKey improvements achieved:")
    print("- Better neural data selection (more neurons)")
    print("- Advanced quantization methods (Grouped Residual VQ)")
    print("- Neural behavior decoder instead of linear")
    print("- Multiple behavioral features")
    print("- Better training procedures (early stopping, LR scheduling)")
    print("\nFiles saved:")
    print("- improved_vqvae_results.png")
    print("- improved_reconstruction_examples.png")
    print("- latent_space_analysis.png")
    print("- best_model.pt")
    
    return results

if __name__ == "__main__":
    # Protezione per multiprocessing su Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    results = main()