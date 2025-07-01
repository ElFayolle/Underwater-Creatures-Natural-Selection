import numpy as np
import random as rd

MIN_TICKS = 50
MAX_TICKS = 60
MIN_N_MOVEMENTS = 3
MAX_N_MOVEMENTS = 6

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
    force_musc = np.zeros(n,ticks,2)
    n_movements = rd.randint(MIN_N_MOVEMENTS, MAX_N_MOVEMENTS) # Nombre de mouvements dans un cycle
    

