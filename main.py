import pygame
import numpy as np

pygame.init()
# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
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

def nombre_de_voisins(k, i, creatures):
    """Calcule le nombre de voisins du point k de la ième créature"""
    nb = 0
    for voisin in creatures[i][1][k] :
        if voisin != 0 :
            nb += 1
    return nb

def frottement_eau(v_moy,vitesse:np.ndarray,position:np.ndarray,t,alpha:float = 1):  #UNE créature
    """Retourne les forces appliquées à chaque sommet i d'une créature dû à l'eau"""
    l=len(position)
    F_visq = np.zeros(l)
    v_reel = vitesse - v_moy*np.ones(l)
    AB = [0,0]
    for i in range(l):
        if i!=l-1:
            AB = position[i+1,t-1]-position[i,t-1]
        cos_theta = np.dot(AB,[1,0])
        sin_theta = np.sqrt(np.max(0,1-cos_theta^2))
        u_theta = -cos_theta*[1,0] + sin_theta*[0,1]
        v_orad_bout = (np.dot(v_reel[i,t-1],u_theta))*u_theta   # Vitesse ortho_radiale du bout du segment
        F_norm =  alpha*(v_orad_bout/2)^2                        # Force du point à r/2
        F_visq[i][0] = F_norm*cos_theta                 
        F_visq[i][1] = F_norm*sin_theta
    

    return F_visq


def force_rappel(i,j,creature):  #Erronée
    k = 100e10
    mi,mj = creature[i][0], creature[j][0]
    l = ((mi[0] - mj[0])**2 + (mi[1] - mj[1])**2)**0.5
    l0 = creature[i][1 ][j]
    u_ij = np.array((mi - mj)) / l
    return -k * (l - l0) * u_ij


def pfd(liste_force, t, mass=1):
    """
    forces: (n_nodes, n_interval_time, n_forces, 2)
    retourne : accelerations of shape (n_nodes, n_interval_time, 2)
    """
    total_force = np.sum(liste_force[:,t], axis=1)  # shape: (n_nodes, n_interval_time, 2)
    accelerations = total_force / mass
    return accelerations

def vitesse_moyenne(vitesse, t):
    """
    vitesse: (n_nodes, n_interval_time, 2)
    t: float
    retourne : moyenne des vitesses sur le temps t
    """
    vitesse_moy = np.mean(vitesse[:, int(t)], axis=0)  # liste de 2 éléments : v_moy_x, v_moy_y
    vitesse_moy = np.linalg.norm(vitesse_moy)  # norme de la vitesse
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


#test de forces aléatoires
 
force_initial = [[[[15,12],[0,0]],[[7,4],[1,3]],[[0,0],[0,0]]] , [[[-25,-22],[-10,-10]],[[-17,-14],[-1,-3]],[[0,0],[0,0]]]]

#calcul_position(np.Array(), float #pas de temps, float #temps de simul, int #nombre de noeuds)
def calcul_position(f_musc_periode, dt = 1/60, T = 10., n_nodes=2):

    #liste_forces = [ f_eau, f_musc, f_rap ]

    pos = [[100,100], [100,300]] #test pos initial pour 2 noeuds
    neigh = [[0,200], [200,0]]   

    #Nombre d'itérations
    n_interval_time = int(T/dt)  
    # Forces qui boucle sur la période cyclique de force donnée
    f_musc = np.array([[f_musc_periode[i][j%len(f_musc_periode[i])] for j in range(n_interval_time)] for i in range(len(f_musc_periode))])    #CI vitesse 
    v = np.zeros((n_nodes, int(n_interval_time), 2))  #shape = (N_noeuds, N_t, 2)
    xy = np.zeros((n_nodes, int(n_interval_time), 2)) #shape = (N_noeuds, N_t, 2)
    a = np.zeros((n_nodes, int(n_interval_time), 2))
    f_eau = np.zeros((n_nodes, int(n_interval_time), 2))
    f_rap = np.zeros((n_nodes, int(n_interval_time), 2))
    print(np.shape(f_eau))
    print(np.shape(f_rap))
    print(np.shape(f_musc))
    xy[:,0] = pos


    for t in range(1,int(n_interval_time)):
        f_eau[t] = frottement_eau(v[:,t-1], xy[:,t-1], t)# fonction de xy[:,t-1]
        f_rap[t] = force_rappel(1,2,3) # fonction de v[:t-1] et xy[:,t-1] ATTENDRE BASILE
        liste_forces = np.array([f_rap, f_eau,f_musc])
        
        a[:,t] = pfd(liste_forces, t)
        
        v[:, t] = v[:, t-1] + dt * a[:, t-1]
        xy[:, t] = xy[:, t-1] + dt * v[:, t-1]

    return (v, xy)




def score(energie, distance, taille):
    score = 2/3*distance/max(distance) + 1/3* energie/taille * max(taille/energie)


def check_line_cross(creature:np.ndarray)->np.ndarray: # Fonction naïve pour empêcher les croisements de segments
    l = len(creature)

    #Tableau booléens d'intersection du segment i "[AB]" au segment j "[CD]""
    pt_intersec = np.zeros((l-1,l-1)) 

    # Calcul des droites passant par chaque segment
    for i in range(l-1):
        for j in range(l-1):  
            delta_x_1 = (creature[i+1][0]-creature[i][0]) 
            delta_x_2 = (creature[j+1][0]-creature[j][0])
            if delta_x_1 <=1e-6:
                coeff_droite_1 = 0
            else:
                coeff_droite_1 = (creature[i+1][1]-creature[i][1])/delta_x_1    # Delta y sur delta x pour les coefficients affines 
            if delta_x_2 <=1e-6:
                coeff_droite_2 = 0
            else:
                coeff_droite_2 = (creature[j+1][1]-creature[j][1])/delta_x_2
            ordonnée_origine_1 = creature[i][1] - coeff_droite_1*creature[i]    # On caclule l'ordonnée à l'origine de chaque droite
            ordonnée_origine_2 = creature[j][1] - coeff_droite_1*creature[j]

        
        # Booléens:
        above_CD_A = (coeff_droite_2*creature[i][0] + ordonnée_origine_2 <= creature[i][1] )
        above_CD_B = (coeff_droite_2*creature[i+1][0] + ordonnée_origine_2 <= creature[i+1][1])
        above_AB_C = (coeff_droite_1*creature[j][0] + ordonnée_origine_1 <= creature[j][1])
        above_AB_D = (coeff_droite_1*creature[j+1][0] + ordonnée_origine_1 <= creature[j+1][1])

        # Si les segments se croisent chaque point est de part et d'autre des deux droites définies:
        if (above_AB_C != above_AB_D ) and (above_CD_A != above_CD_B):  # Si chaque couple de point est de part et d'autre du segment réciproque, il y a intersection !
            pt_intersec[i][j]=1
        
    return pt_intersec


forces = []
pos  = calcul_position(force_initial)[1]
t = 0


"""while running and t < 10/(1/60):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 128, 255))
    
    # Ligne entre les deux points
    pygame.draw.line(screen, (125, 50, 0), pos[0, t], pos[1, t], 10)
    n_nodes = 2
    # Cercles pour chaque point
    for i in range(n_nodes):
        print()
        print(pos[0])
        pygame.draw.circle(screen, (255, 0, 0), pos[i, t], 20)

    pygame.display.flip()
    clock.tick(60)
    t += 1"""

# Quit Pygame
pygame.quit()