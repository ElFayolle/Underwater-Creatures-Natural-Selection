import numpy as np
import matplotlib.pyplot as plt
import random
import math
import json


LENGTH = 70
NOMBRE_DE_CREATURES = 100

def point_exists(new_pos, positions, tol=1e-6):
    """Fonction qui vérifie si le point new_pos recouvre un point déjà existant (vrai si recouvrement)"""
    for p in positions:
        if np.linalg.norm(np.array(new_pos) - np.array(p)) < tol:
            return True
    return False

def croisement(positions, connections, new_pos, base_index):
    """Vérifie si le segment [base_index - new_pos] croise un autre segment existant.
    Renvoie True s'il y a croisement, False sinon."""
    point1 = (positions[base_index][0], positions[base_index][1])
    point2 = (new_pos[0], new_pos[1])

    new_segment = (point1, point2)

    liste_segments = []
    for i in range(len(connections)):
        for j in range(i):
            if connections[i][j] != 0:
                # Éviter les segments liés à base_index
                if i == base_index or j == base_index:
                    continue
                segment = ((positions[i][0], positions[i][1]), (positions[j][0], positions[j][1]))
                liste_segments.append(segment)

    for seg in liste_segments:
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

fig, axes = plt.subplots(5, 5, figsize=(15, 6))
axes = axes.flatten()

for i, ax in enumerate(axes):
    pos, dist = creatures_tot[i]
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

MIN_TICKS = 50
MAX_TICKS = 60
MIN_N_MOVEMENTS = 10
MAX_N_MOVEMENTS = 20
MIN_FORCE_MUSC = -1000
MAX_FORCE_MUSC = 1000

def adn_longueur_segment(creature):
    """Modifie la longueur d'un segment (entre deux points connectés) de la créature.
    Ne modifie la créature que si la nouvelle configuration est valide (sans croisement), sinon la créature reste la même."""
    positions, connections, forces = creature[0], creature[1], creature[2]
    n = len(positions)

    # Choisir un sommet k, puis un voisin i (on suppose que k a toujours au moins un voisin)
    k = random.randint(0, n - 1)
    voisins = [j for j in range(n) if connections[k][j] != 0]
    i = random.choice(voisins)

    # Vecteur du segment ik
    vec = np.array(positions[k]) - np.array(positions[i])
    current_length = np.linalg.norm(vec)

    # Nouvelle longueur perturbée
    new_length = current_length + random.gauss(0, LENGTH / 2)

    direction = vec / current_length
    new_pos_k = np.array(positions[i]) + new_length * direction

    # Vérification de croisement avec les segments existants
    if croisement(positions, connections, new_pos_k, i):
        return creature

    # Appliquer la mutation
    positions[k] = new_pos_k.tolist()
    connections[k][i] = new_length
    connections[i][k] = new_length

    return (positions, connections, forces)


def adn_changement_amplitude_force(creature):
    """Modifie les forces appliquées au cours d'un cycle.
    Prend en argument une créature [positions, matrice, forces] et renvoie une créature avec des forces au même moment mais de valeur différente"""
    forces = np.zeros(np.shape(creature[2]))

    for noeud in range(len(creature[2])):
        for index, force in enumerate(creature[2][noeud]) :
            if force.any() != 0. :
                forces[noeud][index] = force * random.randint(-200, 200) / 100
    return([creature[0], creature[1], forces])



def adn_changement_ordre_force(creature):
    """Modifie l'ordre dans lequel les forces du cycle s'appliquent sur les noeuds.
    Prend en argument une créature [positions, matrice, forces] et renvoie une créature avec des forces de même amplitude mais pas au même moment"""
    forces = np.zeros(np.shape(creature[2]))

    for noeud in range(len(creature[2])):
        liste_forces = [force for force in creature[2][noeud] if force.any() != 0]
        indices = random.sample([k for k in range(len(creature[2][0]))], len(liste_forces))
        for i in indices :
            forces[noeud][i] = liste_forces.pop()
    return([creature[0], creature[1], forces])





def adn_ajout_segment(creature):
    """Ajoute un nouveau noeud (et donc un segment) à la créature si possible.
    Prend en argument une créature [positions, matrice, forces] et renvoie une créature (avec un noeud de plus)
    On tente 10 fois de placer un nouveau segment (avec 10 positions différentes), si on a 10 échecs on abandonne en renvoyant la créature telle quelle"""
    
    max_iterations = 10
    positions = creature[0]
    connections = creature[1]
    forces = creature[2]

    while max_iterations > 0 :
        randomized_length = random.gauss(LENGTH, LENGTH/3)
        sommet = random.randint(0, len(creature[0]) - 1)

        angle = random.randint(1,360)
        angle_rad = math.radians(angle)
        dy = randomized_length * math.sin(angle_rad)
        dx = randomized_length * math.cos(angle_rad)

        # Nouvelle position candidate
        new_pos = [creature[0][sommet][0] + dx, creature[0][sommet][1] + dy]
        
        #Eviter les croisements
        if croisement(creature[0], creature[1], new_pos, sommet):
            max_iterations = max_iterations - 1
            continue
        
        positions = np.concatenate([positions, [new_pos]])
        
        # Mettre à jour matrice de distances
        connections = np.array([np.append(connections[k], 0) for k in range(len(connections))])
        
            
        new_row = [0] * len(positions)
        new_row[sommet] = randomized_length
        
        connections = np.concatenate([connections, [new_row]])

        connections[sommet][-1] = randomized_length
        max_iterations = 0
    return ([positions, connections, forces])


def adn_suppression_segment(creature):
    """Retire le dernier noeud (et donc un segment) de la créature.
    Prend en argument une créature [positions, matrice, forces] et renvoie une créature (avec un noeud de moins)"""
    
    positions = creature[0]
    connections = creature[1]
    forces = creature[2]
    n = len(positions)

    print(connections, positions)

    positions = positions[:-1]
    
    # Mettre à jour matrice de distances
    connections = connections[:-1, :-1]
    
    print(connections, positions)
    return ([positions, connections, forces])



pos, dist = create_random_creature()
creature_test = [pos, dist]
n = len(pos) # Nombre de noeuds
ticks = random.randint(MIN_TICKS, MAX_TICKS) # Nombre de ticks pour un cycle
force_musc = np.zeros((n,ticks,2))
mask = np.zeros((n, ticks), dtype=bool) # On prépare un masque
for i in range(n):
    n_movements = np.random.randint(MIN_N_MOVEMENTS, MAX_N_MOVEMENTS) # Nombre de mouvements dans un cycle pour le noeud i
    mask[i, np.random.choice(ticks, size=n_movements, replace=False)] = True
force_musc[mask] = MIN_FORCE_MUSC + (MAX_FORCE_MUSC - MIN_FORCE_MUSC) * np.random.random((mask.sum(),2))
creature_test.append(force_musc)

creature_test_heritee = adn_suppression_segment(creature_test)


def afficher_creature(ax, positions, connections, color='b', title=""):
    """Affiche une créature sur un axe donné."""
    n = len(positions)
    for i in range(n):
        x, y = positions[i]
        ax.plot(x, y, f'{color}o')  # Noeud
        ax.text(x + 2, y + 2, str(i), fontsize=8, color=color)
        for j in range(i+1, n):
            if connections[i][j] != 0:
                x2, y2 = positions[j]
                ax.plot([x, x2], [y, y2], f'{color}-')
    ax.set_title(title)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True)




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

with open("creatures_text.txt", "w", encoding = 'utf-8') as fichier_texte :
  for key, creature in creatures_tot.items() :
      fichier_texte.write(f"Créature n° {key} :\n\n")
      fichier_texte.write(f"Positions des noeuds : \n{creature[0]}\n\n\n")
      fichier_texte.write(f"Matrice d'adjacence avec distances : \n{creature[1]}\n\n\n")
      fichier_texte.write(f"Forces par noeud en fonction du temps : \n{creature[2]}\n\n\n")

with open("creatures.json", "w", encoding="utf-8") as f:
    json_creatures = []
    for key, creature in creatures_tot.items():
        # Convertir en listes natives
        pos = creature[0].tolist() if hasattr(creature[0], "tolist") else creature[0]
        mat = creature[1].tolist() if hasattr(creature[1], "tolist") else creature[1]
        forc = creature[2].tolist() if hasattr(creature[2], "tolist") else creature[2]
        json_creatures.append([key, pos, mat, forc])
    json.dump(json_creatures, f, indent=2)

with open("creatures_text.txt") as fichier_texte:
  print(fichier_texte.read())
