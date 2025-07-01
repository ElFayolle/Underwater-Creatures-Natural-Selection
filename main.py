import pygame
import numpy as np

pygame.init()
# Set up the display
WIDTH, HEIGHT = 800, 600
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


def force_rappel(positions,l0,t):  #Renvoie la force de rappel totale qui s'applique sur chaque noeud d'une créature
    """positions: (n_nodes, t, 2) # Positions des noeuds
    l0 : (n_nodes, n_nodes) # Longueurs de repos des liens entre les noeuds
    retourne : forces de rappel totale qui s'applique sur chaque noeud de la créature, shape (n_nodes, 2)"""
    k = 5e-1 # Constante de raideur du ressort
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













#calcul_position(np.Array()#cycle de forces de la créature, float #pas de temps, float #temps de simul, int #nombre de noeuds) -> vitesse et position 
def calcul_position(creature,f_musc_periode, dt = 1/60, T = 10.):

    n_nodes = len(pos)
    pos_init, matrice_adjacence = creature[0], creature[1]
    l0 = neighbors(pos_init, matrice_adjacence)
    #pos = [[100,100], [100,300]] #test pos initial pour 2 noeuds
    #neigh = [[0,200], [200,0]]   

    #Nombre d'itérations
    n_interval_time = int(T/dt)  
    # Forces qui boucle sur la période cyclique de force donnée
    f_musc = np.array([[f_musc_periode[i][j%len(f_musc_periode[i])] for j in range(n_interval_time)] for i in range(len(f_musc_periode))])  *100
    #f_musc = np.zeros((n_nodes, n_interval_time,2))
    #accéleration en chaque noeud
    a = np.zeros((n_nodes, n_interval_time, 2))     #shape = (N_noeuds, N_t, 2)

    #vitesse en chaque noeud 
    v = np.zeros((n_nodes, n_interval_time, 2))     #shape = (N_noeuds, N_t, 2)

    #position en chaque noeud
    xy = np.zeros((n_nodes, n_interval_time, 2))    #shape = (N_noeuds, N_t, 2)

    #force de l'eau sur chaque sommet
    f_eau = np.zeros((n_nodes, n_interval_time, 2)) #shape = (N_noeuds, N_t, 2)

    #force de rappel en chaque sommet
    f_rap = np.zeros((n_nodes, n_interval_time, 2)) #shape = (N_noeuds, N_t, 2)

    #Condition initiale de position
    xy[:,0] = pos_init

    #Calcul itératif des forces/vitesses et positions
    for t in range(1,int(n_interval_time)):
        #calcul de la force de frottement liée à l'eau
        f_eau[:,t] = frottement_eau(v,matrice_adjacence,xy,t)

        #force de rappel en chacun des sommets
        f_rap[:,t] = force_rappel(xy, l0, t-1) 
        #Array rassemblant les différentes forces
        liste_forces = np.array([f_rap, f_eau,f_musc])
        
        #Somme des forces et calcul du PFD au temps t
        a[:,t] = pfd(liste_forces, t)
        
        #Calcul de la vitesse et position au temps t
        v[:, t] = v[:, t-1] + dt * a[:, t-1]
        xy[:, t] = xy[:, t-1] + dt * v[:, t-1]
    return (v, xy)



#Fonction qui calcule le "score" de chaque créature - amené à changer.
def score(energie, distance, taille):
    score = 2/3*distance/max(distance) + 1/3* energie/taille * max(taille/energie)


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

def see_creatures(event:pygame.event,position_tot):
    i = 0
    if event.key == pygame.K_LEFT:
            if i!=0:
                i-=1
    if event.key == pygame.K_RIGHT:
            if i<position_tot-1:
                i+=1
    screen.fill((0, 128, 255))
    
    pygame.draw.line()
    return None

def draw_creature(pos,t, offset):
    """Dessinne une créature à un temps t"""
    for index in range(1,len(pos)):
        pygame.draw.line(screen,(125, 50, 0),pos[index-1,t]+offset,pos[index,t]+offset,10)
        pygame.draw.circle(screen,(255,0,0),pos[index-1,t]+offset,10)
    pygame.draw.circle(screen,(255,255,0),pos[2,t]+offset,10)
    return None
    
def get_offset(barycentre, screen_width, screen_height):
    screen_center = np.array([screen_width // 2, screen_height // 2])
    return screen_center - barycentre







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

meduse = [pos, matrice_adjacence]
med2 = [pos2, matrice_adjacence]




#test de forces aléatoires
 
force_initial = [[[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]] , 
                 [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]], 
                  [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]] 
                 #[[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]]



forces = []
pos  = calcul_position(meduse, force_initial)[1]
pos2 = calcul_position(med2,force_initial)[1]
t = 0

#fond = pygame.image.load("fond.jpg").convert()
#fond = pygame.transform.scale(fond, (WIDTH, HEIGHT))

while running and t < 10/(1/60):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            see_creatures(event)
            

    screen.fill((0, 128, 255))
    offset = get_offset(centre_de_masse(pos, t), WIDTH, HEIGHT)
    draw_creature(pos,t)
    draw_creature(pos2,t)
    
    pygame.display.flip()
    clock.tick(60)
    t += 1

# Quit Pygame
pygame.quit()