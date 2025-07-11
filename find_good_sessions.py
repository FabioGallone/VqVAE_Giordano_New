import numpy as np
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import warnings
warnings.filterwarnings('ignore')

print("=== RICERCA SESSIONI ALLEN BRAIN OBSERVATORY ===\n")

# Crea cache
boc = BrainObservatoryCache()

print("1. Scaricando lista esperimenti...")
experiments = boc.get_ophys_experiments()
print(f"   Totale esperimenti disponibili: {len(experiments)}")

# Analizza distribuzione neuroni
neuron_counts = [exp.get('cell_count', 0) for exp in experiments]
neuron_counts = [n for n in neuron_counts if n > 0]

print(f"\n2. Statistiche neuroni per sessione:")
print(f"   Min neuroni: {min(neuron_counts)}")
print(f"   Max neuroni: {max(neuron_counts)}")
print(f"   Media neuroni: {np.mean(neuron_counts):.1f}")
print(f"   Mediana neuroni: {np.median(neuron_counts):.1f}")

# Trova sessioni con diversi criteri
print("\n3. Cercando sessioni ottimali...")

good_sessions = []
min_neurons = 50  # Abbassa la soglia

print(f"   Cercando sessioni con almeno {min_neurons} neuroni...")

for i, exp in enumerate(experiments[:100]):  # Controlla prime 100
    cell_count = exp.get('cell_count', 0)
    if cell_count >= min_neurons:
        print(f"   - Session {exp['id']}: {cell_count} neuroni, {exp.get('cre_line', 'unknown')}")
        good_sessions.append({
            'id': exp['id'],
            'neurons': cell_count,
            'cre_line': exp.get('cre_line', 'unknown'),
            'imaging_depth': exp.get('imaging_depth', 0)
        })
        
        if len(good_sessions) >= 10:
            break

print(f"\n4. Trovate {len(good_sessions)} sessioni con {min_neurons}+ neuroni")

# Verifica quali hanno running speed
print("\n5. Verificando disponibilità dati comportamentali...")

verified_sessions = []
for session in good_sessions[:5]:  # Testa solo le prime 5
    try:
        print(f"   Testando sessione {session['id']}...")
        data_set = boc.get_ophys_experiment_data(session['id'])
        _, running_speed = data_set.get_running_speed()
        
        if len(running_speed) > 10000:
            verified_sessions.append(session)
            print(f"     ✓ OK: {len(running_speed)} timesteps di running speed")
        else:
            print(f"     ✗ Pochi dati: solo {len(running_speed)} timesteps")
            
    except Exception as e:
        print(f"     ✗ Errore: {str(e)[:50]}...")

print(f"\n6. Sessioni verificate con dati completi: {len(verified_sessions)}")

if verified_sessions:
    print("\nSESSIONI CONSIGLIATE:")
    for s in verified_sessions:
        print(f"   ID: {s['id']}, Neuroni: {s['neurons']}, Tipo: {s['cre_line']}")
    
    # Salva il migliore
    best_session_id = max(verified_sessions, key=lambda x: x['neurons'])['id']
    print(f"\nMIGLIOR SESSIONE: {best_session_id}")
    
    # Salva in file per uso futuro
    with open('best_session_id.txt', 'w') as f:
        f.write(str(best_session_id))
    print("\nID salvato in 'best_session_id.txt'")
else:
    print("\nNessuna sessione trovata con i criteri richiesti!")
    print("Prova ad abbassare min_neurons o controllare la connessione internet.")