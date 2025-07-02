import pygame
import numpy as np
import json

pygame.init()
# Set up the display
WIDTH, HEIGHT = 800, 600
DUREE_SIM = 100  # Durée de la simulation en secondes
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

"""def frottement_eau(vitesse:np.ndarray,neighbours:np.ndarray,position:np.ndarray,t,alpha:float = 1):  #UNE créature, UNE vitesse associée. Shapes = [N_noeuds,N_t,2]
    #Retourne les forces appliquées à chaque sommet i d'une créature dû à l'eau (NON)
    l=len(position)
    F_visq = np.zeros((l,2))
    v_moy = vitesse_moyenne(vitesse,t)
    

    for i in range(l):
        for index, voisin in enumerate(neighbours[i]):
            if voisin!=0:
                BA = -position[index,t]+position[i,t] # Vecteur BA avec A le premier sommet 
                norm = np.linalg.norm(BA)
                if norm<1e-12:
                    F_visq[i] = np.array([0,0])
                else:
                    cos_theta = np.dot(BA,np.array([1,0]))/np.linalg.norm(BA)
                    sin_theta = np.dot(BA,np.array([0,1]))/np.linalg.norm(BA)
                    u_theta = +cos_theta*np.array([0,1]) - sin_theta*np.array([1,0])
                    v_orad_bout = (np.dot(vitesse[i,t]-v_moy,u_theta))*u_theta                  # Vitesse ortho_radiale du bout du segment
                    F_visq[i] =   - alpha*(np.linalg.norm(v_orad_bout)/4)*v_orad_bout           # Force du point à r/2

    return F_visq"""

"""def frottement_eau_2(vitesse:np.ndarray,neighbours:np.ndarray,position:np.ndarray,t,alpha:float = 1):
    ""ça ne marche pas a l'aide bordel""
    l=len(position)
    F_visq = np.zeros((l,2))
    v_moy = vitesse_moyenne(vitesse,t)

    for i,_ in enumerate(position[:,t]):
        voisins = [index for index, e in enumerate(neighbours[i]) if e != 0]
        section_efficace = np.zeros((l))
        for index in voisins:
            BA = -position[index,t]+position[i,t] # Vecteur BA avec A le premier sommet 
            norm = np.linalg.norm(BA)
            v_reel = (vitesse[i,t]-v_moy)
            if np.linalg.norm(v_reel) >1e-12:
                v_unitaire = v_reel/np.linalg.norm(v_reel)
                if norm>1e-12:
                # Coordonnées locales 
                    cos_theta = np.dot(BA,np.array([1,0]))/np.linalg.norm(BA)
                    sin_theta = np.dot(BA,np.array([0,1]))/np.linalg.norm(BA)
                    normale_locale = +cos_theta*np.array([0,1]) - sin_theta*np.array([1,0])
                    # print(f"{index},vun:{v_unitaire},vre:{v_reel,norm}")
                    section_efficace[index] = (norm*np.abs(np.dot(v_unitaire,normale_locale)))
                    if np.sum(section_efficace)>1e-12:
                        F_visq[index] += -alpha*v_reel*np.linalg.norm(v_reel)*section_efficace[index]/np.sum(section_efficace)    

    return F_visq"""

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
            if np.linalg.norm(v_moy) >1e-6:
                v_unitaire = v_moy/np.linalg.norm(v_moy)
                if norm>1e-6:
                # Coordonnées locales 
                    cos_theta = np.dot(BA,np.array([1,0]))/np.linalg.norm(BA)
                    sin_theta = np.dot(BA,np.array([0,1]))/np.linalg.norm(BA)
                    normale_locale = +cos_theta*np.array([0,1]) - sin_theta*np.array([1,0])
                    # print(f"{index},vun:{v_unitaire},vre:{v_reel,norm}")
                    section_efficace[index] = (norm*np.abs(np.dot(v_unitaire,normale_locale)))
                
                    if np.sum(section_efficace)>1e-6:
                        F_visq[index] += -alpha*v_moy*np.linalg.norm(v_moy)*section_efficace[index]/np.sum(section_efficace)   
                        #print("prour")
                        #print(F_visq[index],np.sum(section_efficace),section_efficace[index],norm,cos_theta) 

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

def force_musc_projetee_copilot(position,neighbours,f_musc,t):
    """
    Calcule la force musculaire projetée sur les normales aux segments des créatures
    position: (n_nodes, n_interval_time, 2) # Positions des noeuds
    neighbours: (n_nodes, n_nodes) # Matrice d'adjacence des voisins
    f_musc: (n_nodes, n_interval_time, 2) # Forces musculaires
    t: int # Temps
    retourne : forces musculaires projetées sur les normales aux segments, shape (n_nodes, 2)
    """
    l = len(position)
    f_musc_proj = np.zeros((l, 2))
    for i in range(l):
        voisins = [index for index, e in enumerate(neighbours[i]) if e != 0]
        for index in voisins:
            BA = -position[index, t] + position[i, t]  # Vecteur BA avec A le premier sommet
            norm = np.linalg.norm(BA)
            if norm > 1e-6:  # Si le segment est minuscule, pas de force
                cos_theta = np.dot(BA, np.array([1, 0])) / np.linalg.norm(BA)
                sin_theta = np.dot(BA, np.array([0, 1])) / np.linalg.norm(BA)
                u_theta = +cos_theta * np.array([0, 1]) - sin_theta * np.array([1, 0])
                f_musc_proj[i] += f_musc[index, t] * np.dot(f_musc[index, t], u_theta) * u_theta
    return f_musc_proj

def force_musc_projetee(position,neighbours,f_musc,t):
    n_nodes = len(position)
    norm_locales = somme_normales_locales(position,neighbours,t)
    f_musc_t = f_musc[:, t]  # Forces musculaires au temps t
    f_musc_proj = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        f_musc_proj[i] = np.dot(f_musc_t[i], norm_locales[i]) * norm_locales[i] / np.linalg.norm(norm_locales[i])
    return f_musc_proj


def somme_normales_locales(position,neighbours,t):
    dico_normales = normales_locales(position, neighbours, t)
    normales_totales = np.zeros((len(position), 2))
    for couple, normale in dico_normales.items():
        normales_totales[couple[0]] += normale
        normales_totales[couple[1]] += normale
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

pos = np.array([[[100,100]], [[150,150]], [[200,100]]])
pos2 = np.array([[150,300], [500,300], [600,400]])
matrice_adjacence = np.array([[0,1,0], [1,0,1], [0,1,0]])
print("normales",normales_locales(pos, matrice_adjacence, 0))  # Affiche les normales locales pour les positions et la matrice d'adjacence données
print("somme_normales", somme_normales_locales(pos, matrice_adjacence, 0))  # Affiche la somme des normales locales


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
    vitesse_moy = np.mean(vitesse[:, t], axis=0)  # liste de 2 éléments : v_moy_x, v_moy_y
    return vitesse_moy

vit = np.array([[[4,8],[2,3]],[[1,2],[3,4]],[[0,0],[1,1]]])  # Exemple de vitesses pour 3 noeuds et 2 temps
#print("vitesse",vitesse_moyenne(vit, 1))  # Affiche la vitesse moyenne au temps t=1

def energie_cinetique(vitesse, t, masse = 1):
    """
    vitesse: (n_nodes, n_interval_time, 2)
    retourne : énergie cinétique de la créature
    """
    vitesse_norm = np.linalg.norm(vitesse[:, int(t)], axis=1)  # norme de la vitesse pour chaque noeud
    energie = 0.5 * masse * np.sum(vitesse_norm**2)  # somme des énergies cinétiques
    return energie

#print("Energie cinétique", energie_cinetique(vit, 1))  # Affiche l'énergie cinétique pour les vitesses données


def distance(position,t):
    return round(np.linalg.norm(centre_de_masse(position,t)-centre_de_masse(position,0)),0)





"""def f_musc_cohérente(neighbours,position,f_musc,t):
    l=len(position)
    f_correction = np.zeros((l,2))
    u_thetas = np.zeros((l,2))
    for i in range(l):
        for index,voisin in enumerate(neighbours[i]):
            if voisin!=0:
                BA = -position[i,t]+position[index,t]
                norm = np.linalg.norm(BA)
                if norm>1e-12:  #Si le segment est minuscule, pas de force
                    cos_theta = np.dot(BA,np.array([1,0]))/norm
                    sin_theta = np.dot(BA,np.array([0,1]))/norm
                    u_theta = +cos_theta*np.array([0,1]) - sin_theta*np.array([1,0])
                    u_thetas.append([i,index,BA,u_theta]))



    return f_correction

def f_réaction(neighbours,position,f_musc,t):
    l=len(position)
    for i in range(l):
        n_vois=np.count_nonzero(neighbours[i])
"""




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

    f_musc_proj = np.zeros((n_nodes, n_interval_time, 2)) #shape = (N_noeuds, N_t, 2)

    #force de viscosité
    #f_visc = np.zeros((n_nodes, n_interval_time, 2)) #shape = (N_noeuds, N_t, 2)

    #force de rappel en chaque sommet
    f_rap = np.zeros((n_nodes, n_interval_time, 2)) #shape = (N_noeuds, N_t, 2)

    #Condition initiale de position
    xy[:,0] = pos_init
    gamma = 1200
    #Calcul itératif des forces/vitesses et positions
    for t in range(1,int(n_interval_time)):

        #calcul de la force de frottement liée à l'eau
        f_eau[:,t] = frottement_eau_globale(v,matrice_adjacence,xy,t-1,1)

        #force de rappel en chacun des sommets
        f_rap[:,t] = force_rappel_amortie(xy, v, l0, t-1)

        f_musc_proj[:,t] = force_musc_projetee(xy, matrice_adjacence, f_musc, t-1) 

        #force musculaire efficace
        #f_musc[:,t] = f_musc_cohérente(matrice_adjacence,xy,f_musc,t-1)
        #Array rassemblant les différentes forces
        #print(np.linalg.norm(f_eau[:,t]),np.linalg.norm(f_rap[:,t]))
        liste_forces = np.array([f_rap, f_eau,f_musc_proj])
        
        #Somme des forces et calcul du PFD au temps t
        a[:,t] = pfd(liste_forces, t)
        
        #Calcul de la vitesse et position au temps t
        v[:, t] = v[:, t-1] + dt * a[:, t-1]
        xy[:, t] = xy[:, t-1] + dt * v[:, t-1]
    
    #Calcul de l'énergie cinétique et de la distance parcourue
    energie = energie_cinetique(v, n_interval_time-1)
    distance_parcourue = distance(xy, n_interval_time-1)
    score = calcul_score(energie, distance_parcourue, n_nodes)
    return (v, xy, score)



#Fonction qui calcule le "score" de chaque créature - A CHANGER.
def calcul_score(energie, distance, taille):
    score = 2/3 * distance + 1/3 * taille / energie
    return 2/3*distance

def iter_score(position, vitesse): # Calcule les grandeurs liées au score d'UNE créature
    masse = len(position)   # masse et taille sont identiques ici
    energie = np.sum(np.square(vitesse))*masse*0.5
    distance = np.linalg.norm(centre_de_masse(position,0) - centre_de_masse(position,-1))
    return energie,distance,masse #return energie distance taille

def selection(score_total:np.ndarray,force_total,N_selected):
    sorted = np.sort(score_total)
    return sorted


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
        CURRENT_CREATURE = (CURRENT_CREATURE - 1)%(len(position_tot))
    if event.key == pygame.K_RIGHT:
        CURRENT_CREATURE = (CURRENT_CREATURE + 1)%(len(position_tot))
    return None

def draw_creature(pos,t, offset):
    """Dessinne une créature à un temps t"""
    for index in range(1,len(pos)):
        pygame.draw.line(screen,(125, 50, 0),pos[index-1,t]+offset,pos[index,t]+offset,10)
        pygame.draw.circle(screen,(255,0,0),pos[index-1,t]+offset,10)
    pygame.draw.circle(screen,(255,255,0),pos[-1,t]+offset,10)
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
        pygame.draw.circle(screen,(109,169,197),bubble[:-1]+offset,bubble[2])
    return None





"""
Test créature - la Méduse :
"""
pos = np.array([[100,100], [150,150], [200,100]])
pos2 = np.array([[150,300], [500,300], [600,400]])
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
med2 = [pos2, matrice_adjacence]




# test de forces aléatoires

#force_initial = [[[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]] , 
              #   [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0][0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]], 
               #   [[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15][0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]]
                 #

force_initial = ([[[0,-15]], [[0,0]], [[0,0]]])
force_initial2 = ([[[0,-15,]],[[0,0]]])


meduse = [pos, matrice_adjacence,force_initial]
med2 = [pos2, matrice_adjacence, force_initial]

"""
Test simple - le baton
"""
pos3 = np.array([[300,100], [250,150]])
matrice_adjacence3 = np.array([[0,1], [1,0]])
baton = [pos3, matrice_adjacence3, force_initial2]


forces = []
pos  = calcul_position(meduse)[1]
pos2 = calcul_position(med2)[1]
pos3 = calcul_position(baton)[1]
t = 0

"""with open("creature_gagnante.json", "r", encoding="utf-8") as f:
    pos = np.array(json.load(f)[1])"""

#Test bulles
bubbles = instantiate_bubbles(30)
position_tot={0:pos,1:pos2,2:pos3}

while running and t < DUREE_SIM/(1/60):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            see_creatures(event)
            

    screen.fill((0, 128, 255))
    barycentre = centre_de_masse(pos, t)
    offset = get_offset(centre_de_masse(position_tot[CURRENT_CREATURE], t), WIDTH,HEIGHT)
    draw_bubbles(bubbles,offset,barycentre,0,t)
    draw_creature(pos,t, offset)
    draw_creature(pos2,t,offset)
    draw_creature(pos3,t,offset)
    font=pygame.font.Font(None, 24)
    text = font.render("N° : " + str(CURRENT_CREATURE)+" distance : " + str(distance(position_tot[CURRENT_CREATURE],t)) ,1,(255,255,255))
    screen.blit(text, (10, 10))
    
    pygame.display.flip()
    clock.tick(60)
    t += 1
    
# Quit Pygame
pygame.quit()