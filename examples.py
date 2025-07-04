import numpy as np
import matplotlib.pyplot as plt
import random
import math
import json


LENGTH = 70
NOMBRE_DE_CREATURES = 1000
MIN_TICKS = 50
MAX_TICKS = 60
MIN_N_MOVEMENTS = 10
MAX_N_MOVEMENTS = 20
MIN_FORCE_MUSC = -10
MAX_FORCE_MUSC = 10



def calcul_distance(point1, point2):
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
    return (math.sqrt((y2 - y1)**2 + (x2 - x1)**2))

def calcul_distance(point1, point2):
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
    return (math.sqrt((y2 - y1)**2 + (x2 - x1)**2))

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

    # Liste des segments de la créature (en coordonnées) sauf ceux partant de base_index
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
    Renvoie True s'il y a croisement.
    Si on compare deux segments partageant une extrémité, on renvoie False."""
    p1, p2 = segment1
    p3, p4 = segment2
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    #Test d'égalité de point - si les segments partent du même point ils ne se croisent pas
    if (x1 == x3 and y1 == y3) or (x1 == x4 and y1 == y4) or (x2 == x3 and y2 == y3) or (x2 == x4 and y2 == y4) :
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
    num_points = random.randint(5, 6)
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
    """Vérifie si la matrice des distances est symétrique ainsi que la correspondance entre la matrice et les coordonnées"""
    return is_symmetric(distance_matrix) and distances_match(positions, distance_matrix)



# creatures_tot = {}
# for i in range(NOMBRE_DE_CREATURES):
#     pos, dist = create_random_creature()
#     creatures_tot[i] = [pos, dist]

# fig, axes = plt.subplots(5, 5, figsize=(15, 6))
# axes = axes.flatten()

# for i, ax in enumerate(axes):
#     pos, dist = creatures_tot[i]
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




def adn_longueur_segment(creature):
    """Modifie la longueur d'un segment (entre deux points connectés) de la créature.
    Ne modifie la créature que si la nouvelle configuration est valide (sans croisement), sinon la créature reste la même."""
    positions, connections, forces = np.copy(creature[0]), np.copy(creature[1]), np.copy(creature[2])
    n = len(positions)

    # Choisir un sommet k, puis un voisin i (on suppose que k a toujours au moins un voisin)
    #k = random.randint(0, n - 1)
    k = 0
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

    #Recalcul des distances avec les voisins
    for voisin, distance in enumerate(connections[k]) :
        if distance != 0 and voisin != k :
            nouvelle_distance = calcul_distance(positions[voisin], positions[k])
            connections[voisin][k] = nouvelle_distance
            connections[k][voisin] = nouvelle_distance


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
    positions = np.copy(creature[0])
    connections = np.copy(creature[1])
    forces = np.copy(creature[2])

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

        forces_point = force_musculaire_aleatoire_noeud(len(forces[0]))
        forces = np.concatenate([forces, [forces_point]])
    return ([positions, connections, forces])


def adn_suppression_segment(creature):
    """Retire un noeud (et donc un segment) à la créature.
    On retire un noeud qui est en bout de chaîne pour éviter d'avoir une créature coupée en deux.
    Prend en argument une créature [positions, matrice, forces] et renvoie une créature (avec un noeud de moins)"""
    
    positions = np.copy(creature[0])
    connections = np.copy(creature[1])
    forces = np.copy(creature[2])
    n = len(positions)

    if n == 3:
        return creature

    candidats = []

    for index, sommet in enumerate(connections) :
        if sum([1 for i in sommet if i != 0]) == 1 :
            candidats.append(index)

    noeud_suppr = random.choice(candidats)
    positions1 = positions[:noeud_suppr]
    positions2 = positions[noeud_suppr + 1:]
    positions = np.concatenate([positions1, positions2], axis = 0)
    
    
    connections1 = connections[:noeud_suppr]
    connections2 = connections[noeud_suppr + 1:]
    connections = np.concatenate([connections1, connections2], axis = 0)

    connections1 = connections[:,:noeud_suppr]
    connections2 = connections[:,noeud_suppr + 1:]
    connections = np.concatenate([connections1, connections2], axis = 1)

    forces1 = forces[:noeud_suppr]
    forces2 = forces[noeud_suppr + 1:]
    forces = np.concatenate([forces1, forces2], axis = 0)

    return ([positions, connections, forces])


def creature_est_valide(creature):
    """Vérifie si la créature contient des segments qui se croisent.
    Attention : beaucoup de tests car on regarde tous les couples de segments"""
    n = len(creature[0])

    # Liste des segments de la créature (en coordonnées) sauf ceux partant de base_index
    liste_segments = []
    for i in range(n):
        for j in range(i):
            if creature[1][i][j] != 0:
                segment = ((creature[0][i][0], creature[0][i][1]), (creature[0][j][0], creature[0][j][1]))
                liste_segments.append(segment)
    
    for seg1 in liste_segments :
        for seg2 in liste_segments :
            if croisement_segments(seg1, seg2) :
                return False
    return True



def adn_changement_position_noeud (creature):
    """Modifie la position d'un noeud de la créature, en gardant les mêmes relations entre les noeuds.
    Ne modifie la créature que si la nouvelle configuration est valide (sans croisement), sinon la créature reste la même."""

    positions = np.copy(creature[0])
    connections = np.copy(creature[1])
    forces = np.copy(creature[2])
    
    n = len(positions)

    # Choisir un sommet k, puis un voisin i (on suppose que k a toujours au moins un voisin)
    k = random.randint(0, n - 1)
    #print("Sommet choisi pour déplacer :", k)

    # Vecteur du segment ik
    dx = random.randint(-30, 30)
    dy = random.randint(-30, 30)
    vec = np.array([dx, dy])

    positions[k] = [positions[k][0] + dx, positions[k][1] + dy]

    #Recalcul des distances avec les voisins
    for voisin, distance in enumerate(connections[k]) :
        if distance != 0 and voisin != k :
            nouvelle_distance = calcul_distance(positions[voisin], positions[k])
            connections[voisin][k] = nouvelle_distance
            connections[k][voisin] = nouvelle_distance

    if not creature_est_valide([positions, connections, forces]) :
        print("Impossible de bouger le point ", k)
        return creature
    
    return (positions, connections, forces)


def adn_ajout_force(creature):
    """Ajoute une force supplémentaire à un noeud aléatoire à un moment aléatoire du cycle (un moment où il n'y a pas de force)
    Prend en argument une créature et renvoie une créature"""
    forces = np.copy(creature[2])
    positions, connections = np.copy(creature[0]), np.copy(creature[1])

    noeud = random.randint(0, len(positions) - 1)

    indices = [index for index, vecteur in enumerate(forces[noeud]) if vecteur.all() == 0]
    indice_changement = random.choice(indices)

    forces[noeud][indice_changement][0] = MIN_FORCE_MUSC + (MAX_FORCE_MUSC - MIN_FORCE_MUSC) * random.random()
    forces[noeud][indice_changement][1] = MIN_FORCE_MUSC + (MAX_FORCE_MUSC - MIN_FORCE_MUSC) * random.random()
    
    return [positions, connections, forces]

def adn_suppression_force(creature):
    """Enlève une force (non nulle) aléatoire du cycle d'un noeud aléatoire de la créature.
    Prend en argument une créature et renvoie une créature."""
    forces = np.copy(creature[2])
    positions, connections = np.copy(creature[0]), np.copy(creature[1])

    noeud = random.randint(0, len(positions) - 1)

    indices = [index for index, vecteur in enumerate(forces[noeud]) if vecteur.any() != 0]
    indice_changement = random.choice(indices)

    forces[noeud][indice_changement][0] = 0
    forces[noeud][indice_changement][1] = 0
    
    return [positions, connections, forces]


def adn_duree_cycle_forces(creature):
    """Modifie la durée du cycle des forces de la créature.
    V1 : On rajoute x 0 ou on supprime les x derniers temps avec x aléatoire"""

    forces = np.copy(creature[2])
    x = random.randint(1, 10)
    sens = random.choice([-1, 1])
    print(np.shape(forces))
    if sens == -1 :
        forces = forces[:,:-x]
    else :
        forces = np.concatenate([forces, np.zeros((len(forces), x, 2))], axis = 1)
    print(np.shape(forces), x, sens)

    return [np.copy(creature[0]), np.copy(creature[1]), forces]



def afficher_deux_creatures_sur_meme_graphe(ax, positions1, connections1, positions2, connections2):
    """Affiche deux créatures superposées avec couleurs différentes (avant/après mutation)."""
    n1 = len(positions1)
    for i in range(n1):
        x, y = positions1[i]
        ax.plot(x, y, 'bo')  # Avant mutation : bleu
        ax.text(x + 1, y + 1, str(i), fontsize=8, color='b')
        for j in range(i+1, n1):
            if connections1[i][j] != 0:
                x2, y2 = positions1[j]
                ax.plot([x, x2], [y, y2], 'b--', linewidth=1)

    n2 = len(positions2)
    for i in range(n2):
        x, y = positions2[i]
        ax.plot(x, y, 'ro')  # Après mutation : rouge
        ax.text(x + 1, y - 3, str(i), fontsize=8, color='r')
        for j in range(i+1, n2):
            if connections2[i][j] != 0:
                x2, y2 = positions2[j]
                ax.plot([x, x2], [y, y2], 'r-', linewidth=2)

    ax.set_title("Bleu : avant / Rouge : après mutation")
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True)

# Affichage matplotlib
# fig, ax = plt.subplots(figsize=(8, 8))
# afficher_deux_creatures_sur_meme_graphe(ax, creature_test[0], creature_test[1], creature_test_heritee[0], creature_test_heritee[1])
# plt.show()


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

def mutation_creature(creature):
    """Applique une mutation à la créature.
    Renvoie la créature modifiée."""
    liste_mutations = [
        adn_longueur_segment,
        adn_changement_amplitude_force,
        adn_changement_ordre_force,
        adn_ajout_segment,
        adn_suppression_segment,
        adn_changement_position_noeud,
        adn_duree_cycle_forces,
        adn_ajout_force,
        adn_suppression_force
    ]
    mutation = random.choice(liste_mutations)
    creature_modifiee = mutation(creature)
    return creature_modifiee

def creature_force_musculaire_aleatoire(creatures_tot):
    """Ajoute à chaque créature une force musculaire aléatoire."""
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
    return creatures_tot

def force_musculaire_aleatoire_noeud(ticks):
    """Génère des forces aléatoires sous forme vectorielle pendant un nombre donné de ticks
    Prend en argument le nombre de ticks et renvoie un tableau numpy des forces sur l'ensemble des ticks"""
    force_musc_noeud = np.zeros((ticks, 2))
    mask = np.zeros(ticks, dtype=bool)  # On prépare un masque
    n_movements = np.random.randint(MIN_N_MOVEMENTS, MAX_N_MOVEMENTS)  # Nombre de mouvements dans un cycle pour le noeud
    mask[np.random.choice(ticks, size=n_movements, replace=False)] = True
    force_musc_noeud[mask] = MIN_FORCE_MUSC + (MAX_FORCE_MUSC - MIN_FORCE_MUSC) * np.random.random((mask.sum(), 2))
    return force_musc_noeud

def generation_initiale():
    creatures_tot = {}
    for i in range(NOMBRE_DE_CREATURES):
        pos, dist = create_random_creature()
        creatures_tot[i] = [pos, dist]

    creatures_tot = creature_force_musculaire_aleatoire(creatures_tot)

    with open("generations/meilleures_creatures_0.txt", "w", encoding = 'utf-8') as fichier_texte :
        for key, creature in creatures_tot.items() :
            fichier_texte.write(f"Créature n° {key} :\n\n")
            fichier_texte.write(f"Positions des noeuds : \n{creature[0]}\n\n\n")
            fichier_texte.write(f"Matrice d'adjacence avec distances : \n{creature[1]}\n\n\n")
            fichier_texte.write(f"Forces par noeud en fonction du temps : \n{creature[2]}\n\n\n")

    with open("generations/meilleures_creatures_0.json", "w", encoding="utf-8") as f:
        json_creatures = []
        for key, creature in creatures_tot.items():
            # Convertir en listes natives
            pos = creature[0].tolist() if hasattr(creature[0], "tolist") else creature[0]
            mat = creature[1].tolist() if hasattr(creature[1], "tolist") else creature[1]
            forc = creature[2].tolist() if hasattr(creature[2], "tolist") else creature[2]
            json_creatures.append([key, pos, mat, forc])
        json.dump(json_creatures, f, indent=2)

generation_initiale()

# pos, dist = create_random_creature()
# creature_test = [pos, dist]
# n = len(pos) # Nombre de noeuds
# ticks = random.randint(MIN_TICKS, MAX_TICKS) # Nombre de ticks pour un cycle
# force_musc = np.zeros((n,ticks,2))
# mask = np.zeros((n, ticks), dtype=bool) # On prépare un masque
# for i in range(n):
#     n_movements = np.random.randint(MIN_N_MOVEMENTS, MAX_N_MOVEMENTS) # Nombre de mouvements dans un cycle pour le noeud i
#     mask[i, np.random.choice(ticks, size=n_movements, replace=False)] = True
# force_musc[mask] = MIN_FORCE_MUSC + (MAX_FORCE_MUSC - MIN_FORCE_MUSC) * np.random.random((mask.sum(),2))
# creature_test.append(force_musc)

# creature_test_heritee = adn_longueur_segment(creature_test)