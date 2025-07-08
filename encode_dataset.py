import torch
import numpy as np
import argparse
import os
import sys

# Aggiungi la directory corrente al path
sys.path.append(os.getcwd())

# Prova diversi import
try:
    from vqvae import VQVAE
    print("✓ Importato da vqvae.py")
except ImportError:
    try:
        from models.vqvae import VQVAE
        print("✓ Importato da models/vqvae.py")
    except ImportError:
        try:
            from model import VQVAE
            print("✓ Importato da model.py")
        except ImportError:
            print("✗ Impossibile importare VQVAE")
            exit(1)

from utils import load_data_and_data_loaders

def encode_dataset(model_path, dataset_name, save_path):
    # Verifica che il file del modello esista
    if not os.path.exists(model_path):
        print(f"Errore: Il file del modello {model_path} non esiste!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Carica il checkpoint per vedere i parametri
    checkpoint = torch.load(model_path, map_location=device)
    
    print("Contenuto del checkpoint:")
    for key in checkpoint.keys():
        print(f"  {key}: {type(checkpoint[key])}")
    
    # Proviamo a ottenere i parametri dal checkpoint
    if 'hyperparameters' in checkpoint:
        hyperparams = checkpoint['hyperparameters']
        print(f"Iperparametri trovati: {hyperparams}")
        
        # Mappa i parametri ai nomi corretti del costruttore
        model_params = {
            'h_dim': hyperparams['n_hiddens'],
            'res_h_dim': hyperparams['n_residual_hiddens'],
            'n_res_layers': hyperparams['n_residual_layers'],
            'n_embeddings': hyperparams['n_embeddings'],
            'embedding_dim': hyperparams['embedding_dim'],
            'beta': hyperparams['beta']
        }
        
        print(f"Parametri del modello: {model_params}")
        
        # Crea il modello con i parametri corretti
        try:
            model = VQVAE(**model_params)
            print("✓ Modello creato con iperparametri del checkpoint")
        except Exception as e:
            print(f"Errore nella creazione del modello: {e}")
            return
    else:
        # Parametri di default se non sono nel checkpoint
        try:
            model = VQVAE(h_dim=128, res_h_dim=32, n_res_layers=2, 
                         n_embeddings=512, embedding_dim=64, beta=0.25)
            print("✓ Modello creato con parametri di default")
        except Exception as e:
            print(f"Errore con parametri di default: {e}")
            return
    
    # Carica i pesi del modello
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✓ Modello caricato e impostato in modalità eval")
    
    # Carica i dati
    print(f"Caricando il dataset: {dataset_name}")
    training_data, validation_data, training_loader, validation_loader, _ = load_data_and_data_loaders(
        dataset_name, batch_size=32
    )
    
    # Test con un singolo batch per capire la struttura
    print("Testando con un batch...")
    with torch.no_grad():
        for data, _ in training_loader:
            data = data.to(device)
            print(f"Forma input: {data.shape}")
            
            # Forward pass base
            output = model(data)
            print(f"Output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"Output contiene {len(output)} elementi")
                for i, item in enumerate(output):
                    if hasattr(item, 'shape'):
                        print(f"  Output[{i}] shape: {item.shape}")
                    else:
                        print(f"  Output[{i}]: {item}")
            
            # Estrazione indici discreti dal quantizer
            print("\nTentando di estrarre gli indici discreti...")
            encoded = model.encoder(data)
            pre_quantized = model.pre_quantization_conv(encoded)
            quantizer = model.vector_quantization
            quantized_output = quantizer(pre_quantized)
            
            # Gli indici sono nel quinto elemento (indice 4)
            if isinstance(quantized_output, tuple) and len(quantized_output) >= 5:
                raw_indices = quantized_output[4]  # shape: [batch*H*W, 1]
                B = data.size(0)
                _, _, H, W = pre_quantized.shape
                # Rimodella in [B, H, W]
                min_encoding_indices = raw_indices.view(B, H, W)
                print(f"Min encoding indices shape: {min_encoding_indices.shape}")
                print("✓ Trovati gli indici discreti nel batch di test!")
                break
            else:
                print("Formato inaspettato del quantizer output durante il test.")
                return
    
    # Processa tutto il dataset
    print("\nProcessando tutto il dataset...")
    latent_indices = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(training_loader):
            if batch_idx % 100 == 0:
                print(f"Processato batch {batch_idx}/{len(training_loader)}")
            data = data.to(device)
            
            # Encoding completo
            encoded = model.encoder(data)
            pre_quantized = model.pre_quantization_conv(encoded)
            quantized_output = model.vector_quantization(pre_quantized)
            
            # Estrai e rimodella indici
            if isinstance(quantized_output, tuple) and len(quantized_output) >= 5:
                raw_indices = quantized_output[4]
                B = data.size(0)
                _, _, H, W = pre_quantized.shape
                min_encoding_indices = raw_indices.view(B, H, W)
                latent_indices.append(min_encoding_indices.cpu().numpy())
            else:
                print(f"Formato inaspettato del quantizer output al batch {batch_idx}.")
                return
    
    # Concatena tutti i batch
    all_latent_indices = np.concatenate(latent_indices, axis=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, all_latent_indices)
    print(f"✓ Salvate {all_latent_indices.shape[0]} rappresentazioni latenti in {save_path}")
    print(f"✓ Forma degli indici latenti: {all_latent_indices.shape}")
    print(f"✓ Range valori: {all_latent_indices.min()} - {all_latent_indices.max()}")
    print("✓ Ora puoi eseguire: python pixelcnn/gated_pixelcnn.py --dataset LATENT_BLOCK")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--save_path', type=str, default='data/latent_e_indices.npy')
    
    args = parser.parse_args()
    encode_dataset(args.model_path, args.dataset, args.save_path)