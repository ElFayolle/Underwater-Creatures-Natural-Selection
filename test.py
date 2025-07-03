import pygame
import numpy as np

pygame.init()
# Set up the display
WIDTH, HEIGHT = 800, 600
DUREE_SIM = 30  # Durée de la simulation en secondes
CURRENT_CREATURE = 0
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Natural Selection Simulation")
# Set up the clock for frame rate control
clock = pygame.time.Clock()
# Main loop
running = True

n_nodes = 10
forces = np.zeros((n_nodes, 4))
forces = [ [[[15,12],[0,0]],[[7,4],[1,3]],[[0,0],[0,0]]] , [[[25,22],[10,10]],[[17,14],[1,3]],[[0,0],[0,0]]] ]
accelerations = []





"""def force_musculaire(i, k, creatures, forces_creatures_points):
    #Calcule la force musculaire pour les points voisins du point k de la ième créature
    nb_vois = nombre_de_voisins(k, i, creatures)
    for index, voisin in enumerate(creatures[i][1][k]) :
        if voisin != 0 :
            forces[i][index][1] += forces_creatures_points[i][k] / nb_vois"""

def centres_de_masse(positions_tot:np.ndarray,t):

    C_tot = np.zeros((len(positions_tot,2)))
    for index,pos in enumerate(positions_tot): # Boucle for berk mais je ne trouve rien de pratique
        C_tot[index] = centre_de_masse(pos) 
    return C_tot

def centre_de_masse(position:np.ndarray,t):
    """Calcul du centre de masse de chaque créature à un instant t"""
    C = np.mean(position[:, t], axis=0) 
    return C

def nombre_de_voisins(k, i, creatures):
    """Calcule le nombre de voisins du point k de la ième créature"""
    nb = 0
    for voisin in creatures[i][1][k] :
        if voisin != 0 :
            nb += 1
    return nb

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
    xy_t = xy[:, t-1]
    n = len(xy_t)
    centre_avant = centre_de_masse(xy, t-1)  # Centre de masse avant ajustement
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
    xy[:,t] = xy_t
    centre_apres = centre_de_masse(xy, t)
    xy_t += centre_avant - centre_apres  # Recentre les positions
    return xy_t

def frottement_eau_3(vitesse:np.ndarray,neighbours:np.ndarray,position:np.ndarray,t,alpha:float = 1):
    l=len(position)
    F_visq = np.zeros((l,2))
    v_moy = vitesse_moyenne(vitesse,t)

    norm_locales = somme_normales_locales(position,neighbours,t)
    for node, normale in enumerate(norm_locales):
        if np.linalg.norm(normale) > 1e-10:
            F_visq[node] = -alpha*(vitesse[node,t]-v_moy)*np.dot((vitesse[node,t]-v_moy),normale)
            #print(f"à l'aide: {norm_locales},{vitesse[node,t]},{v_moy},{F_visq}")
    return F_visq  


def somme_normales_locales(position,neighbours,t):
    dico_normales = normales_locales(position, neighbours, t)
    normales_totales = np.zeros((len(position), 2))
    eps=1e-10
    for couple, normale in dico_normales.items():
        normales_totales[couple[0]] += normale
        normales_totales[couple[1]] += normale
    for i,normale in enumerate(normales_totales):
        if np.linalg.norm(normale)>eps:
            normales_totales[i] = normale/np.linalg.norm(normale)
        else:
            normales_totales[i] = np.array([0,0])
    return normales_totales

def normales_locales(position,neighbours,t)->dict:
    d = {}
    for i in range(len(position)):
        voisins = [index for index, e in enumerate(neighbours[i],start=0) if e != 0]
        for index in voisins:
            if ((index,i) in d) ^ ((i,index) not in d): 
                BA = -position[index,t]+position[i,t] # Vecteur BA avec A le premier sommet 
                norm = np.linalg.norm(BA)
                if norm>1e-6:
                # Coordonnées locales 
                    cos_theta = np.dot(BA,np.array([1,0]))/np.linalg.norm(BA)
                    sin_theta = np.dot(BA,np.array([0,1]))/np.linalg.norm(BA)
                    normale_locale = +cos_theta*np.array([0,1]) - sin_theta*np.array([1,0])
                    d[(index,i)] = normale_locale
    return d  

"""

def force_rappel(positions,l0,t):  #Renvoie la force de rappel totale qui s'applique sur chaque noeud d'une créature
    #positions: (n_nodes, t, 2) # Positions des noeuds
    #l0 : (n_nodes, n_nodes) # Longueurs de repos des liens entre les noeuds
    #retourne : forces de rappel totale qui s'applique sur chaque noeud de la créature, shape (n_nodes, 2)
    k = 10 # Constante de raideur du ressort
    pos = positions[:, t]  # On prend les positions au temps t
    # Étendre les positions pour faire des soustractions vectorisées
    pos_i = pos[:, np.newaxis, :]     # shape (n, 1, 2)
    pos_j = pos[np.newaxis, :, :]     # shape (1, n, 2)
    # Vecteurs de déplacement entre nœuds : r_ij = pos_j - pos_i
    vec = pos_j - pos_i             # shape (n, n, 2)
    # Distances actuelles
    l = np.linalg.norm(vec, axis=2)   # shape (n, n)
    # Éviter division par 0 (ajouter petite valeur ε)
    eps = 1e-12
    unit_vec = vec / (l[..., np.newaxis] + eps)  # shape (n, n, 2)
    # Calcul de la force de rappel selon Hooke : F = -k*(L - L0) * u
    # On met une condition masque pour les liens existants
    mask = (l0 > 0)
    # Delta L
    delta_L = l - l0                 # shape (n, n)

    # Forces totales
    F = -k * delta_L[..., np.newaxis] * unit_vec  # shape (n, n, 2)

    # Ne garder que les forces là où il y a un ressort (càd mettre à 0 les forces pour les nœuds sans lien entre eux)
    F[~mask] = 0.0

    # Résultat : F[i,j] est la force exercée sur le nœud j par le ressort entre i et j
    forces = F.sum(axis=0)
    
    return forces

"""



def pfd(liste_force, t, mass=1):
    """
    forces: (n_nodes, n_interval_time, n_forces, 2)
    retourne : accelerations de chaque noeud (n_nodes, n_interval_time, 2)
    """
    total_force = np.sum(liste_force[:,:,t,:], axis=0)  # shape: (n_nodes, n_interval_time, 2)
    accelerations = total_force / mass
    return accelerations

def vitesse_moyenne(vitesse, t):
    """
    vitesse: (n_nodes, n_interval_time, 2)
    t: float
    retourne : moyenne des vitesses sur le temps t
    """
    vitesse_moy = np.sum(vitesse[:, t], axis=0)  # liste de 2 éléments : v_moy_x, v_moy_y
    return vitesse_moy

vit = np.array([[[4,8],[2,3]],[[1,2],[3,4]],[[0,0],[1,1]]])  # Exemple de vitesses pour 3 noeuds et 2 temps
print("vitesse",vitesse_moyenne(vit, 1))  # Affiche la vitesse moyenne au temps t=1

def energie_cinetique(vitesse, t, masse = 1):
    """
    vitesse: (n_nodes, n_interval_time, 2)
    retourne : énergie cinétique de la créature
    """
    vitesse_norm = np.linalg.norm(vitesse[:, int(t)], axis=1)  # norme de la vitesse pour chaque noeud
    energie = 0.5 * masse * np.sum(vitesse_norm**2)  # somme des énergies cinétiques
    return energie

print("Energie cinétique", energie_cinetique(vit, 1))  # Affiche l'énergie cinétique pour les vitesses données


def distance(position,t):
    return round(np.linalg.norm(centre_de_masse(position,t)-centre_de_masse(position,0)),0)





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

print( "test = ", action_reaction(np.array([[-1,0],[0,0], [-1,0]]), np.array([[0,0], [2,2], [2, 0]]), np.array([[0,1,0],[1,0, 1],[0,1,0]])) )  # Exemple de force de réaction pour 2 noeuds


def orthogonalise_force(force_musc, pos, l0):
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

print(orthogonalise_force(np.array([[0,0],[1,2]]), np.array([[0,0], [0,2]]), np.array([[0,1],[1,0]])))  # Exemple de force musculaire orthogonalisée pour 2 noeuds
#calcul_position(np.Array()#cycle de forces de la créature, float #pas de temps, float #temps de simul, int #nombre de noeuds) -> vitesse et position 
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

        f_eau[:,t] = frottement_eau_globale(v,matrice_adjacence,xy,t)
        #f_visc[:,t] = -gamma*v[:,t]
        #force de rappel en chacun des sommets
        f_rap[:,t] = 0 #force_rappel_amortie(xy, v, l0, t-1) 
        force_reaction[:,t] = action_reaction(f_musc[:,t], xy[:,t], l0)   
        #Array rassemblant les différentes forces
        #f_musc[:,t] = orthogonalise_force(f_musc[:,t], xy[:,t], l0)
        liste_forces = np.array([f_rap, force_reaction ,f_musc])

        #Somme des forces et calcul du PFD au temps t
        a[:,t] = pfd(liste_forces, t)
        
        #Calcul de la vitesse et position au temps t
        v[:, t] = v[:, t-1] + dt * a[:, t-1]
        xy[:, t] = contrainte_longueurs(xy, l0, matrice_adjacence, t)
        xy[:, t] = xy[:, t-1] + dt * v[:, t-1]
        

    return (v, xy, liste_forces)


#Fonction qui calcule le "score" de chaque créature - amené à changer.
def score(energie, distance, taille):
    score = 2/3*distance/max(distance) + 1/3* energie/taille * max(taille/energie)

def iter_score(position, vitesse): # Calcule les grandeurs liées au score d'UNE créature
    masse = len(position)   # masse et taille sont identiques ici
    energie = np.sum(np.square(vitesse))*masse*0.5
    distance = np.linalg.norm(centre_de_masse(position,0) - centre_de_masse(position,-1))
    return energie,distance,masse #return energie distance taille

def selection(score_total:np.ndarray,force_total,):

    return None


def check_line_cross(position:np.ndarray,t)->np.ndarray: # Fonction naïve pour empêcher les croisements de segments
    l = len(position)

    # Tableau booléens d'intersection du segment i "[AB]" au segment j "[CD]""
    pt_intersec = np.zeros((l-1,l-1)) 

    # Calcul des droites passant par chaque segment
    for i in range(l-1):
        for j in range(l-1): 
            is_straight_1,is_straight_2 = False, False 
            delta_x_1 = (position[i+1,t,0]-position[i,t,0]) 
            delta_x_2 = (position[j+1,t,0]-position[j,t,0])
            if delta_x_1 <=1e-6:
                is_straight_1 = True
            else:
                coeff_droite_1 = (position[i+1,t,1]-position[i,t,1])/delta_x_1    # Delta y sur delta x pour les coefficients affines 
            if delta_x_2 <=1e-6:
                is_straight_2 = True
            else:
                coeff_droite_2 = (position[j+1,t,1]-position[j,t,1])/delta_x_2   
            if not is_straight_1:    
                ordonnée_origine_1 = position[i,t,1] - coeff_droite_1*position[i,t,1]    # On caclule l'ordonnée à l'origine de chaque droite
            if not is_straight_2:
                ordonnée_origine_2 = position[j,t,1] - coeff_droite_1*position[j,t,1]

        
            # Booléens:
            if is_straight_2:
                above_CD_A = (position[j,t,0] <= position[i,t,0])    # Above = à droite si la ligne est verticale pure
                above_CD_B = (position[j,t,0] <= position[i+1,t,0])
            else: 
                above_CD_A = (coeff_droite_2*position[i,t,0] + ordonnée_origine_2 <= position[i,t,1] )
                above_CD_B = (coeff_droite_2*position[i+1,t,0] + ordonnée_origine_2 <= position[i+1,t,1])
            if is_straight_1:
                above_AB_C = (position[i,t,0] <= position[j,t,0])
                above_AB_D = (position[i,t,0] <= position[j+1,t,0])
            else:
                above_AB_C = (coeff_droite_1*position[j,t,0] + ordonnée_origine_1 <= position[j,t,1])
                above_AB_D = (coeff_droite_1*position[j+1,t,0] + ordonnée_origine_1 <= position[j+1,t,1])

            # Si les segments se croisent chaque point est de part et d'autre des deux droites définies:
            if (above_AB_C != above_AB_D ) and (above_CD_A != above_CD_B):  # Si chaque couple de point est de part et d'autre du segment réciproque, il y a intersection !
                pt_intersec[i,j]=1
        
    return pt_intersec

def see_creatures(event:pygame.event):
    global CURRENT_CREATURE
    if event.key == pygame.K_LEFT:
            if CURRENT_CREATURE!=0:
                CURRENT_CREATURE-=1
    if event.key == pygame.K_RIGHT:
            if CURRENT_CREATURE<len(position_tot)-1:
                CURRENT_CREATURE+=1
    return None

def draw_creature(pos, liste_forces, t, offset):
    """Dessinne une créature à un temps t"""
    liste_forces = liste_forces/3
    for i in range(t):
        pygame.draw.circle(screen,(20,60,120),centre_de_masse(pos,i)+offset,2 )
    for index in range(1,len(pos)):
        pygame.draw.line(screen,(25, 30, 70),pos[index-1,t]+offset,pos[index,t]+offset,4)
        pygame.draw.circle(screen,(100,189,255),pos[index-1,t]+offset,5)
        #draw forces :  

    for index in range(len(pos)):
        colours_force = [(255,0,0),(0,255,0),(0,0,255)]
        for i in range(len(liste_forces)):
            pygame.draw.line(screen,colours_force[i],pos[index-1,t]+offset,pos[index-1,t]+liste_forces[i][index-1,t]*10+offset,2)        
    pygame.draw.circle(screen,(100,189,255),pos[-1,t]+offset,5)
    pygame.draw.circle(screen,(255,0,0),centre_de_masse(pos,0)+offset,3)
    return None
    
def get_offset(barycentre, screen_width, screen_height):
    screen_center = np.array([screen_width // 2, screen_height // 2])
    return screen_center - barycentre

def instantiate_bubbles(N_bubbles,rmax=10):
    bubbles = np.random.rand(N_bubbles,3)
    bubbles[:,0] *= WIDTH
    bubbles[:,1] *= HEIGHT
    bubbles[:,2] *= rmax
    return bubbles

def draw_bubbles(bubbles,offset,barycentre,v_moy,t):
    for index,bubble in enumerate(bubbles):
        pygame.draw.circle(screen,(29,50,140),bubble[:-1]+offset,bubble[2])
    return None





"""
Test créature - la Méduse :
"""
pos = np.array([[100,100], [150,150], [200,100]])
pos2 = np.array([[150,300], [500,300], [600,400]])
pos3 = np.array([[120,120], [150,150], [170,120]]) + np.array([[300, 0], [300, 0], [300, 0]]) # Créature décalée pour le test
matrice_adjacence = np.array([[0,1,0], [1,0,1], [0,1,0]])

#Calcul les longueurs à vide dans une matrice d'adjacence
def neighbors(pos, matrice_adjacence):
    l0 = np.zeros((len(pos), len(pos)))
    for i,point in enumerate(matrice_adjacence):
        for j,voisin in enumerate(point):
            if voisin != 0:
                l0[i,j] = np.linalg.norm(pos[i]-pos[j])
    return l0

def bubulle(centre_masse,v_moy):
    
    return None



meduse = [pos, matrice_adjacence]
med2 = [pos3, matrice_adjacence]




#test de forces aléatoires

#force_initial = [[[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]] , 
              #   [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0][0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]], 
               #   [[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15][0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]]
                 #

force_initial = ([[[10,0]], [[0,0]], [[0,15]]])


meduse = [pos, matrice_adjacence,force_initial]
med2 = [pos3, matrice_adjacence, force_initial]

"""
Test simple - le baton
"""
#pos = np.array([[100,100], [150,150]])
#matrice_adjacence = np.array([[0,1], [1,0]])
baton = [pos, matrice_adjacence, force_initial]


forces = []
pos  = calcul_position(meduse)[1]
force = calcul_position(meduse)[2]
pos2 = calcul_position(med2)[1]
t = 0


#Test bulles
bubbles = instantiate_bubbles(30)
position_tot=np.array([pos,pos2])

while running and t < DUREE_SIM/(1/60):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            see_creatures(event)
            

    screen.fill((0, 35, 120))
    barycentre = centre_de_masse(pos, t)
    offset = get_offset(centre_de_masse(position_tot[CURRENT_CREATURE], t), WIDTH,HEIGHT)
    draw_bubbles(bubbles,offset,barycentre,0,t)
    draw_creature(pos, force,t, offset)
    #draw_creature(pos2,t,offset)
    font=pygame.font.Font(None, 24)
    text = font.render("distance : " + str(distance(pos,t)),1,(255,255,255))
    coulours_force = [(255,0,0),(0,255,0),(0,0,255)]
    force_name = ["force rappel", "force de réaction", "force musculaire"]
    for i in range(len(coulours_force)):
        text_force = font.render(force_name[i],1,coulours_force[i])
        screen.blit(text_force, (10, 30 + i * 20))
    screen.blit(text, (10, 10))
    
    pygame.display.flip()
    clock.tick(60)
    t += 1
    
# Quit Pygame
pygame.quit()