import numpy as np
import matplotlib.pyplot as plt
import random
import math
#from main import check_line_cross

LENGTH = 20

def croisement(positions):
    """Vrai ou faux selon si le dernier segment ajouté à la créature croise un segment existant (vrai pour croisement)"""
    matrice_croisements = check_line_cross(positions)
    colonne = matrice_croisements[:,-1]
    for k in colonne[:-1] :
        if k != 0 :
            return True
    return False


def create_random_creature():
    num_points = random.randint(8, 9)
    positions = [[0, 0]]
    connections = [[0]]  # matrice de distances
    i = 0
    
    while i < num_points - 1 :
        base_index = random.randint(0, len(positions) - 1)

        angle = random.randint(0, 360)
        angle_rad = math.radians(angle)
        dy = LENGTH * math.sin(angle_rad)
        dx = LENGTH * math.cos(angle_rad)


        # Nouvelle position candidate
        new_pos = [positions[base_index][0] + dx, positions[base_index][1] + dy]

        # Éviter les doublons
        if new_pos in positions:
            continue

        # Ajouter position
        positions.append(new_pos)

#        if croisement(np.array(positions)):
#            positions.pop()
#            continue

        # Mettre à jour matrice de distances
        for row in connections:
            row.append(0)
        new_row = [0] * len(positions)
        new_row[base_index] = LENGTH
        connections[base_index][len(positions) - 1] = LENGTH
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

creatures = {}
for i in range(25):
    pos, dist = create_random_creature()
    creatures[i] = (pos, dist)

fig, axes = plt.subplots(5, 5, figsize=(15, 6))
axes = axes.flatten()

for i, ax in enumerate(axes):
    pos, dist = creatures[i]
    for j in range(len(pos)):
        x, y = pos[j]
        ax.plot(x, y, 'ko')
        ax.text(x + 1, y + 1, str(j), fontsize=8)
        for k in range(j+1, len(pos)):
            if dist[j][k] != 0:
                x2, y2 = pos[k]
                ax.plot([x, x2], [y, y2], 'b-')

    ax.set_title(f"Créature {i}")
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True)

plt.tight_layout()
plt.show()


