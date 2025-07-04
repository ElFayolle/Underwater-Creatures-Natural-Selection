from utils import *
from params import *


# Fonction de frottement dans l'eau pour les créatures - dernière version
def frottement_eau_3(vitesse:np.ndarray,neighbours:np.ndarray,position:np.ndarray,t,alpha:float = 1):
    """
    vitesse: (n_nodes, n_t, 2)
    neighbours: matrice d'adjacence (n_nodes, n_nodes)
    position: (n_nodes, n_t, 2)
    t: temps instantané (int)
    alpha: coefficient de frottement
    return : force de frottement visqueux au temps t (n_nodes, 2)
    """
    l=len(position)
    F_visq = np.zeros((l,2))
    v_moy = vitesse_moyenne(vitesse,t-1)

    norm_locales = somme_normales_locales(position,neighbours,t-1)
    for node, normale in enumerate(norm_locales):
        # Fonction de frottement pas identique en fonction de la présence de voisins
        if np.count_nonzero(neighbours[node])<=1:
            #if np.linalg.norm(normale) > 1e-10:
            F_visq[node] = -alpha*(vitesse[node,t])*np.linalg.norm((vitesse[node,t]))
        else:
            F_visq[node] = -alpha*(vitesse[node,t])
    return F_visq  


# Fonction de rappel amorti pour les segments - obsolète, suite aux divergences, on garde les segments indéformables
def force_rappel_amortie(positions, vitesses, l0, t, k=10e-3, c=10):
    pos = positions[:, t]
    vel = vitesses[:, t]
    n = len(pos)
    pos_i = pos[:, np.newaxis, :]
    pos_j = pos[np.newaxis, :, :]
    r_ij = pos_j - pos_i
    l = np.linalg.norm(r_ij, axis=2)
    eps = 1e-12
    u_ij = r_ij / (l[..., np.newaxis] + eps)

    # Vitesse relative projetée sur l’axe du ressort
    vel_i = vel[:, np.newaxis, :]
    vel_j = vel[np.newaxis, :, :]
    vel_rel = vel_j - vel_i
    damping = c * np.sum(vel_rel * u_ij, axis=2, keepdims=True) * u_ij

    delta_l = (l - l0)[..., np.newaxis]
    F_spring = -k * delta_l * u_ij
    F_total = F_spring - damping
    F_total[l0 == 0] = 0.0
    return F_total.sum(axis=0)

def force_repulsion_noeuds(pos, matrice_adjacence, seuil=10.0, k_rep=100.0, t=0):
    """
    Applique une force répulsive entre noeuds non connectés trop proches.
    - pos: tableau des positions (n_nodes, n_t, 2)
    - matrice_adjacence: (n_nodes, n_nodes)
    - seuil: distance minimale tolérée
    - k_rep: intensité de la force répulsive
    - t: instant de temps
    Renvoie un tableau de forces répulsives (n_nodes, 2)
    """
    n = len(pos)
    force_rep = np.zeros((n, 2))
    is_rep = np.zeros((n,2))

    for i in range(n):
        for j in range(i + 1, n):
            if matrice_adjacence[i, j] == 0:  # Pas connectés
                
                delta = pos[j, t] - pos[i, t]
                dist = np.linalg.norm(delta)
                if dist < seuil and dist > 1e-6:
                    if not (is_rep[i][0] or is_rep[j][0]):
                        print("r&épulsion")
                        is_rep[i]=np.array([1,1])
                        is_rep[j]=np.array([1,1])
                    direction = delta / dist
                    magnitude = k_rep / ((dist/seuil+0.5)**4) 
                    f = -magnitude * direction
                    force_rep[i] += f
                    force_rep[j] -= f  # Action-réaction

    return is_rep,force_rep

#Ajuste xy_t (positions au temps t) pour que la distance entre chaque paire de noeuds connectés soit égale à l0.
def contrainte_longueurs(xy, l0, matrice_adjacence, t):
    """
    xy : positions des noeuds de la créature (n_nodes, n_interval_time, 2)
    l0 : matrice des longueurs à vide (n_nodes, n_nodes)
    matrice_adjacence : matrice d'adjacence de la créature (n_nodes, n_nodes)
    t : le temps actuel (int)
    retourne : xy_t ajusté pour respecter les contraintes de longueur
    """
    xy_t = xy[:, t]
    n = len(xy_t)
    for _ in range(5):  # nombre d'itérations de correction (à ajuster)
        for i in range(n):
            for j in range(i+1, n):
                if matrice_adjacence[i, j] != 0:
                    vec = xy_t[j] - xy_t[i]
                    dist = np.linalg.norm(vec)
                    if dist < 1e-10:
                        continue
                    diff = dist - l0[i, j]
                    correction = (diff / 2) * (vec / dist)
                    xy_t[i] += correction
                    xy_t[j] -= correction
                        
    return xy_t

# Fonction d'action-réaction pour les forces musculaires
def action_reaction(force_musc, pos, l0):
    """
    force_musc: (n_nodes, n_interval_time, 2)
    pos: (n_nodes, n_interval_time, 2)
    l0: (n_nodes, n_nodes)
    retourne : force de réaction pour chaque noeud (n_nodes, 2)
    """
    force_reaction = np.zeros((len(pos), 2))  # Initialisation des forces de réaction
    for i in range(len(pos)):
        for j in range(len(pos)):
            if l0[i, j] > 0:
                # Calcul de la force de réaction selon le principe d'action-réaction
                force_reaction[i] += -force_musc[j]
    return force_reaction




def orthogonalise_force(force_musc, pos, l0,t):
    """
    Force musculaire orthogonalisée pour chaque noeud d'une créature.
    force_musc: (n_nodes, n_interval_time, 2)
    pos: (n_nodes, n_interval_time, 2)
    l0: (n_nodes, n_nodes)
    retourne : force_musc orthogonalisée
    """
    n_nodes = len(pos)
    force_orthogonalisee = np.zeros((n_nodes,2), dtype=np.float64) 
    normales = somme_normales_locales(pos,l0,t-1)  #t-1 car la position d'avant définit la normale pour les projections de force à temps t
    for node, normale in enumerate(normales):
        force_orthogonalisee[node] = np.dot(force_musc[node,t], normale) * normale
    return force_orthogonalisee
 

# Fonction de calcul du PFD
def pfd(liste_force, t, mass=1):
    """
    liste_force: (n_nodes, n_interval_time, n_forces, 2)
    t: le temps actuel (int)
    mass: masse de la créature (float)
    retourne : accelerations de chaque noeud (n_nodes, n_interval_time, 2)
    """

    total_force = np.sum(liste_force[:,:,t,:], axis=0)  # shape: (n_nodes, n_interval_time, 2)
    accelerations = total_force / mass
    return accelerations

# Fonction itérative qui calcul la position de la créature ainsi que les forces appliquées sur chaque noeud
def calcul_position(creature, dt = 1/60, T = DUREE_SIM):
    """creature: liste contenant les positions initiales, la matrice d'adjacence et les forces musculaires périodiques
   dt: pas de temps pour la simulation (float)
   T: durée totale de la simulation (float)
   return : (vitesse, positions, liste_forces, score)"""


    pos_init, matrice_adjacence, f_musc_periode = creature[0], creature[1], creature[2]
    n_nodes = len(pos_init)  
    l0 = neighbors(pos_init, matrice_adjacence)

    delta_t_amort=0
    t_amort=10

    
    #Nombre d'itérations
    n_interval_time = int(T/dt)  

    # Forces qui boucle sur la période cyclique de force donnée
    f_musc = np.array([[f_musc_periode[i][j%len(f_musc_periode[i])] for j in range(n_interval_time)] for i in range(len(f_musc_periode))]) 

    # Initialisation des tableaux pour les forces, vitesses, positions et accélérations shape : (N_noeuds, N_t, 2)
    a = np.zeros((n_nodes, n_interval_time, 2))     
    v = np.zeros((n_nodes, n_interval_time, 2))    
    xy = np.zeros((n_nodes, n_interval_time, 2))    
    f_eau = np.zeros((n_nodes, n_interval_time, 2))
    force_reaction = np.zeros((n_nodes, n_interval_time, 2)) 
    f_repulsion = np.zeros((n_nodes, n_interval_time, 2))

    #Condition initiale de position
    xy[:,0] = pos_init
    
    #Calcul itératif des forces/vitesses et positions
    for t in range(1,int(n_interval_time)):

        #calcul de la force de frottement liée à l'eau
        f_eau[:,t] = frottement_eau_3(v,matrice_adjacence,xy,t-1,1) 

        f_musc[:,t] = orthogonalise_force(f_musc, xy, l0,t)
        force_reaction[:,t] = action_reaction(f_musc[:,t], xy[:,t], l0)
        force_reaction[:,t] = orthogonalise_force(force_reaction,xy,l0,t)
        is_rep_bool, force_rep = force_repulsion_noeuds(xy, matrice_adjacence, seuil=15.0, k_rep=300.0, t=t-1)
    
        
        f_repulsion[:, t] =   force_rep  +(-f_eau[:,t] -force_reaction[:,t] - f_musc[:,t])*is_rep_bool
        
        f_repulsion[:,t] = orthogonalise_force(f_repulsion,xy,l0,t)


        
        #Array rassemblant les différentes forces
        liste_forces = np.array([f_eau, force_reaction ,f_musc, f_repulsion])

        #Somme des forces et calcul du PFD au temps t
        a[:,t] = pfd(liste_forces, t)
        
        #Calcul de la vitesse et position au temps t
        v[:, t] = v[:, t-1] + dt * a[:, t-1]
        xy[:, t] = xy[:,t-1] + dt * v[:, t-1]
        xy[:, t] = contrainte_longueurs(xy, l0, matrice_adjacence, t)

        score = distance(xy,n_interval_time-1)
        
    return (v, xy, liste_forces, score)


