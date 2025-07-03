from utils import *
from params import *

def frottement_eau(vitesse:np.ndarray,neighbours:np.ndarray,position:np.ndarray,t,alpha:float = 1):  #UNE créature, UNE vitesse associée. Shapes = [N_noeuds,N_t,2]
    """Retourne les forces appliquées à chaque sommet i d'une créature dû à l'eau"""
    l=len(position)
    F_visq = np.zeros((l,2))
    v_moy = vitesse_moyenne(vitesse,t)
    for i in range(l-1):
        for index, voisin in enumerate(neighbours[i]):
            if voisin!=0:
                BA = -position[i+1,t-1]+position[i,t-1] # Vecteur BA avec A le premier sommet 
                cos_theta = np.dot(BA,np.array([1,0]))/np.linalg.norm(BA)
                sin_theta = np.dot(BA,np.array([0,1]))/np.linalg.norm(BA)
                u_theta = +cos_theta*np.array([0,1]) - sin_theta*np.array([1,0])
                v_orad_bout = (np.dot(vitesse[i,t-1]-v_moy,u_theta))*u_theta   # Vitesse ortho_radiale du bout du segment
                F_visq[i] =  - alpha*(np.linalg.norm(v_orad_bout)/4)*v_orad_bout                        # Force du point à r/2

    # Le dernier sommet correspond à inverser le calcul: on regarde l'angle de l'autre bout du bras donc on "déphase" de pi, cos= -cos, sin = -sin

    u_theta = -u_theta
    v_orad_bout = (np.dot(vitesse[l-1,t-1]-v_moy,u_theta))*u_theta   
    F_visq[l-1] =  - alpha*(np.linalg.norm(v_orad_bout)/4)*v_orad_bout
    
    return F_visq

def force_rappel_amortie(positions, vitesses, l0, t, k=10e-3, c=10):
    """
    Ajoute un amortissement proportionnel à la vitesse relative le long de l’axe du ressort
    """
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

def contrainte_longueurs(xy, l0, matrice_adjacence, t):
    """
    Ajuste xy_t (positions au temps t) pour que la distance entre chaque paire de noeuds connectés soit égale à l0.
    Utilise une correction simple itérative.
    """
    xy_t = xy[:, t]
    n = len(xy_t)
    centre_avant = centre_de_masse(xy, t)  # Centre de masse avant ajustement
    for _ in range(5):  # nombre d'itérations de correction (à ajuster)
        for i in range(n):
            for j in range(i+1, n):
                if matrice_adjacence[i, j] != 0:
                    vec = xy_t[j] - xy_t[i]
                    dist = np.linalg.norm(vec)
                    if dist == 0:
                        continue
                    diff = dist - l0[i, j]
                    correction = (diff / 2) * (vec / dist)
                    xy_t[i] += correction
                    xy_t[j] -= correction
                        
    # Réajustiment du centre de masse pour éviter les dérives
    # xy[:,t] = xy_t
    # centre_apres = centre_de_masse(xy, t)
    # xy_t += centre_avant - centre_apres  # Recentre les positions
    # if t>1:
    #     print("pipi",xy_t[1],xy[1,t-1],xy[1,t-1])
    return xy_t


def frottement_eau_globale(vitesse:np.ndarray,neighbours:np.ndarray,position:np.ndarray,t,alpha:float = 1):
    l=len(position)
    F_visq = np.zeros((l,2))
    v_moy = vitesse_moyenne(vitesse,t)

    for i,_ in enumerate(position[:,t]):
        voisins = [index for index, e in enumerate(neighbours[i]) if e != 0]
        section_efficace = np.zeros((l))
        for index in voisins:
            BA = -position[index,t]+position[i,t] # Vecteur BA avec A le premier sommet 
            norm = np.linalg.norm(BA)
            if norm>1e-6:
            # Coordonnées locales 
                cos_theta = np.dot(BA,np.array([1,0]))/np.linalg.norm(BA)
                sin_theta = np.dot(BA,np.array([0,1]))/np.linalg.norm(BA)
                normale_locale = +cos_theta*np.array([0,1]) - sin_theta*np.array([1,0])
                # print(f"{index},vun:{v_unitaire},vre:{v_reel,norm}")
                section_efficace[index] = (norm*np.abs(np.dot(vitesse[index,t],normale_locale)))
            
                if np.sum(section_efficace)>1e-6:
                    F_visq[index] += -alpha*vitesse[index,t]*np.linalg.norm(vitesse[index,t])*section_efficace[index]/np.sum(section_efficace)   
                    #print("prour")
                    #print(F_visq[index],np.sum(section_efficace),section_efficace[index],norm,cos_theta) 

    return F_visq

def frottement_eau_3(vitesse:np.ndarray,neighbours:np.ndarray,position:np.ndarray,t,alpha:float = 1):
    l=len(position)
    F_visq = np.zeros((l,2))
    v_moy = vitesse_moyenne(vitesse,t)

    norm_locales = somme_normales_locales(position,neighbours,t)
    for node, normale in enumerate(norm_locales):
        if np.linalg.norm(normale) > 1e-10:
            F_visq[node] = -alpha*(vitesse[node,t]-v_moy)*np.dot((vitesse[node,t]-v_moy),normale)
            #F_visq[node] = -alpha*(vitesse[node,t])*np.linalg.norm((vitesse[node,t]))
    return F_visq  

def force_eau_angle():
    return None

def action_reaction(force_musc, pos, l0):
    """

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
    force_orthogonalisee = np.zeros(force_musc.shape, dtype=np.float64) 
    for i in range(n_nodes):
        for j in range(n_nodes):
            if l0[i, j] > 0:
                # Calcul de la force orthogonale
                vec = pos[j] - pos[i]
                print(vec)
                vec_orth = np.array([-vec[1], vec[0]])  # Vecteur orthogonal
                norm_vec = np.linalg.norm(vec_orth) + 1e-12  # Éviter division par zéro
                unit_vec = vec_orth / norm_vec
                force_orthogonalisee[j] += np.dot(force_musc[j], unit_vec) * unit_vec

    return force_orthogonalisee

def orthogonalise_force2(force_musc, pos, l0,t):
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
 


def pfd(liste_force, t, mass=1):
    """
    forces: (n_nodes, n_interval_time, n_forces, 2)
    retourne : accelerations de chaque noeud (n_nodes, n_interval_time, 2)
    """
    total_force = np.sum(liste_force[:,:,t,:], axis=0)  # shape: (n_nodes, n_interval_time, 2)
    accelerations = total_force / mass
    return accelerations

def calcul_position(creature, dt = 1/60, T = DUREE_SIM):
   
    pos_init, matrice_adjacence, f_musc_periode = creature[0], creature[1], creature[2]
    n_nodes = len(pos_init)  # Nombre de noeuds dans la créature
    l0 = neighbors(pos_init, matrice_adjacence)
    #pos = [[100,100], [100,300]] #test pos initial pour 2 noeuds
    #neigh = [[0,200], [200,0]]   

    #Nombre d'itérations
    n_interval_time = int(T/dt)  
    # Forces qui boucle sur la période cyclique de force donnée
    f_musc = np.array([[f_musc_periode[i][j%len(f_musc_periode[i])] for j in range(n_interval_time)] for i in range(len(f_musc_periode))]) 
    #f_musc = np.zeros((n_nodes, n_interval_time,2))
    #accéleration en chaque noeud
    a = np.zeros((n_nodes, n_interval_time, 2))     #shape = (N_noeuds, N_t, 2)

    #vitesse en chaque noeud 
    v = np.zeros((n_nodes, n_interval_time, 2))     #shape = (N_noeuds, N_t, 2)

    #position en chaque noeud
    xy = np.zeros((n_nodes, n_interval_time, 2))    #shape = (N_noeuds, N_t, 2)

    #force de l'eau sur chaque sommet
    f_eau = np.zeros((n_nodes, n_interval_time, 2))  #shape = (N_noeuds, N_t, 2)

    #force de viscosité
    #f_visc = np.zeros((n_nodes, n_interval_time, 2)) #shape = (N_noeuds, N_t, 2)

    #force de rappel en chaque sommet
    f_rap = np.zeros((n_nodes, n_interval_time, 2)) #shape = (N_noeuds, N_t, 2)


    force_reaction = np.zeros((n_nodes, n_interval_time, 2)) #shape = (N_noeuds, N_t, 2)
    #Condition initiale de position
    xy[:,0] = pos_init
    gamma = 1200
    #Calcul itératif des forces/vitesses et positions
    for t in range(1,int(n_interval_time)):
        #calcul de la force de frottement liée à l'eau

        f_eau[:,t] = frottement_eau_globale(v,matrice_adjacence,xy,t-1,10) #np.array([[15,15] for i in range(n_nodes)])#
        #f_visc[:,t] = -gamma*v[:,t]
        #force de rappel en chacun des sommets
        f_rap[:,t] = 0 #force_rappel_amortie(xy, v, l0, t-1) 
        #force_reaction[:,t] = orthogonalise_force2(force_reaction,xy,l0,t)  
        force_reaction[:,t] = action_reaction(f_musc[:,t], xy[:,t], l0)
        force_reaction[:,t] = orthogonalise_force2(force_reaction, xy, l0,t)

        f_musc[:,t] = orthogonalise_force2(f_musc, xy, l0,t)
        
        #Array rassemblant les différentes forces
        liste_forces = np.array([f_eau, force_reaction ,f_musc])

        #Somme des forces et calcul du PFD au temps t
        a[:,t] = pfd(liste_forces, t)
        
        #Calcul de la vitesse et position au temps t
        v[:, t] = v[:, t-1] + dt * a[:, t-1]
        xy[:, t] = xy[:,t-1] + dt * v[:, t-1]
        xy[:, t] = contrainte_longueurs(xy, l0, matrice_adjacence, t)

    return (v, xy, liste_forces)