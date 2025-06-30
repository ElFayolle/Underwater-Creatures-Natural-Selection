import numpy as np

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



creatures_tot = {}
creatures_tot[1] = creature1
creatures_tot[2] = creature2
creatures_tot[3] = creature3
print(creatures_tot)