import numpy as np
import matplotlib.pyplot as plt
import random
import math


LENGTH = 70
NOMBRE_DE_CREATURES = 25

def point_exists(new_pos, positions, tol=1e-6):
    """Fonction qui vérifie si le point new_pos recouvre un point déjà existant (vrai si recouvrement)"""
    for p in positions:
        if np.linalg.norm(np.array(new_pos) - np.array(p)) < tol:
            return True
    return False

def croisement(positions, connections, new_pos, base_index):
    """Vérifie si le dernier segment ajouté (entre new_pos et le point d'indice base_index) croise un segment existant
    Renvoie True si le segment est invalide (False si pas de croisement)"""
    point1 = (positions[base_index][0], positions[base_index][1])
    point2 = (new_pos[0], new_pos[1])

    new_segment = (point1, point2)

    #Etablir la liste des segments existants sous la forme (point1, point2)
    liste_segments = []
    for i in range(len(connections)):
        for j in range(i):
            if connections[i][j] != 0 :
                segment = ((positions[i][0], positions[i][1]), (positions[j][0], positions[j][1]))
                liste_segments.append(segment)

    #Vérification de croisement pour chaque segment
    for seg in liste_segments :
        if croisement_segments(new_segment, seg):
            return True

    return False

def croisement_segments(segment1, segment2):
    """Vérifie si deux segments se croisent.
    Renvoie True s'il y a croisement."""
    p1, p2 = segment1
    p3, p4 = segment2
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    #Test d'égalité de point - si les segments partent du même point ils ne se croisent pas
    if (x1 == x3 and y1 == y3) or (x1 == x4 and y1 == y4) :
        return False

    #Coefficients de la droite AB
    a1, b1 = y2 - y1, x1 - x2
    c1 = -(a1 * x1 + b1 * y1)

    #Coefficients de la droite CD
    a2, b2 = y4 - y3, x3 - x4
    c2 = -(a2 * x3 + b2 * y3)

    #Test des produits croisés
    d1 = a1 * x3 + b1 * y3 + c1
    d2 = a1 * x4 + b1 * y4 + c1
    d3 = a2 * x1 + b2 * y1 + c2
    d4 = a2 * x2 + b2 * y2 + c2

    if d1 * d2 <= 0 and d3 * d4 <= 0:
        return True  #Croisement détecté

    return False  #Pas de croisement détecté


def create_random_creature():
    num_points = random.randint(4, 6)
    positions = [[0, 0]]
    connections = [[0]]  #Matrice d'adjacence (de distances)
    i = 0
    
    while i < num_points - 1 :
        base_index = random.randint(0, len(positions) - 1)

        #Distribution gaussienne des longueurs des segments
        randomized_length = random.gauss(LENGTH, LENGTH/3)

        angle = random.randint(1,360)
        angle_rad = math.radians(angle)
        dy = randomized_length * math.sin(angle_rad)
        dx = randomized_length * math.cos(angle_rad)


        # Nouvelle position candidate
        new_pos = [positions[base_index][0] + dx, positions[base_index][1] + dy]
        
        #Eviter les croisements
        if croisement(positions, connections, new_pos, base_index):
            continue

        positions.append(new_pos)

        # Mettre à jour matrice de distances
        for row in connections:
            row.append(0)
        new_row = [0] * len(positions)
        new_row[base_index] = randomized_length
        connections[base_index][len(positions) - 1] = randomized_length
        connections.append(new_row)
        i += 1

    # Conversion en numpy
    pos_array = np.array(positions)
    dist_array = np.array(connections)
    return (pos_array, dist_array)

def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)

def distances_match(positions, distance_matrix):
    n = len(positions)
    for i in range(n):
        for j in range(n):
            if distance_matrix[i, j] != 0:
                actual_dist = round(np.linalg.norm(positions[i] - positions[j]), 5)
                if abs(actual_dist - distance_matrix[i, j]) > 0.01:
                    return False
    return True

def is_valid_creature(positions, distance_matrix):
    return is_symmetric(distance_matrix) and distances_match(positions, distance_matrix)

creatures_tot = {}
for i in range(NOMBRE_DE_CREATURES):
    pos, dist = create_random_creature()
    creatures_tot[i] = [pos, dist]

# fig, axes = plt.subplots(5, 5, figsize=(15, 6))
# axes = axes.flatten()

# for i, ax in enumerate(axes):
#     pos, dist = creatures[i]
#     for j in range(len(pos)):
#         x, y = pos[j]
#         ax.plot(x, y, 'ko')
#         ax.text(x + 1, y + 1, str(j), fontsize=8)
#         for k in range(j+1, len(pos)):
#             if dist[j][k] != 0:
#                 x2, y2 = pos[k]
#                 ax.plot([x, x2], [y, y2], 'b-')

#     ax.set_title(f"Créature {i}")
#     ax.axis('equal')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.grid(True)

# plt.tight_layout()
# plt.show()

MIN_TICKS = 50
MAX_TICKS = 60
MIN_N_MOVEMENTS = 3
MAX_N_MOVEMENTS = 6
MIN_FORCE_MUSC = 0.1
MAX_FORCE_MUSC = 0.5

# Nombre aléatoire de ticks par cycle, de mouvements par noeud dans un cycle, et de valeurs de force musculaire par noeud dans un cycle
for key, value in creatures_tot.items():
    n = len(value[0]) # Nombre de noeuds
    ticks = random.randint(MIN_TICKS, MAX_TICKS) # Nombre de ticks pour un cycle
    force_musc = np.zeros((n,ticks,2))
    mask = np.zeros((n, ticks), dtype=bool) # On prépare un masque
    for i in range(n):
        n_movements = np.random.randint(MIN_N_MOVEMENTS, MAX_N_MOVEMENTS) # Nombre de mouvements dans un cycle pour le noeud i
        mask[i, np.random.choice(ticks, size=n_movements, replace=False)] = True
    force_musc[mask] = MIN_FORCE_MUSC + (MAX_FORCE_MUSC - MIN_FORCE_MUSC) * np.random.random((mask.sum(),2))
    creatures_tot[key].append(force_musc)

print(creatures_tot[0])