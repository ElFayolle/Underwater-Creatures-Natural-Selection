import numpy as np
import random as rd

MIN_TICKS = 50
MAX_TICKS = 60
MIN_N_MOVEMENTS = 3
MAX_N_MOVEMENTS = 6
MIN_FORCE_MUSC = 0.1
MAX_FORCE_MUSC = 0.5

# --- Créature 1 ---
positions1 = np.array([[0, 0], [0, 20]])
distances1 = np.array([[0, 20], [20, 0]])
creature1 = (positions1, distances1)

# --- Créature 2 ---
positions2 = np.array([[0, 0], [0, 20], [20, 0]])
distances2 = np.array([
    [0, 20, 20],
    [20, 0, 0],
    [20, 0, 0]
])
creature2 = (positions2, distances2)

# --- Créature 3 ---
positions3 = np.array([[100, 100], [80, 100], [80, 80], [100, 120], [100, 80]])
distances3 = np.array([
    [0, 20, 0, 20, 20],
    [20, 0, 0, 0, 0],
    [0, 0, 0, 0, 20],
    [20, 0, 0, 0, 0],
    [20, 0, 20, 0, 0]
])
creature3 = (positions3, distances3)

# --- Dictionnaire total ---
creatures_tot = {
    1: creature1,
    2: creature2,
    3: creature3
}

# Nombre aléatoire de ticks par cycle, de mouvements par noeud dans un cycle, et de valeurs de force musculaire par noeud dans un cycle
for key, value in creatures_tot.items():
    n = len(value[0]) # Nombre de noeuds
    ticks = rd.randint(MIN_TICKS, MAX_TICKS) # Nombre de ticks pour un cycle
    force_musc = np.zeros((n,ticks,2))
    mask = np.zeros((n, ticks), dtype=bool) # On prépare un masque
    for i in range(n):
        n_movements = np.random.randint(MIN_N_MOVEMENTS, MAX_N_MOVEMENTS) # Nombre de mouvements dans un cycle pour le noeud i
        mask[i, np.random.choice(ticks, size=n_movements, replace=False)] = True # On choisit aléatoirement les ticks où le noeud i va bouger
    force_musc[mask] = MIN_FORCE_MUSC + (MAX_FORCE_MUSC - MIN_FORCE_MUSC) * np.random.random((mask.sum(),2)) # Valeurs de force musculaire
    # aléatoires pour les noeuds qui bougent
    print(force_musc)
    print("fin item")
    

