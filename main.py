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
forces = [[(15,12),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(-7,8),(0,0),(0,0)]]
accelerations = []




def force_musculaire(i, k, creatures, forces_creatures_points):
    """Calcule la force musculaire pour les points voisins du point k de la ième créature"""
    nb_vois = nombre_de_voisins(k, i, creatures)
    for index, voisin in enumerate(creatures[i][1][k]) :
        if voisin != 0 :
            forces[i][index][1] += forces_creatures_points[i][k] / nb_vois

def nombre_de_voisins(k, i, creatures):
    """Calcule le nombre de voisins du point k de la ième créature"""
    nb = 0
    for voisin in creatures[i][1][k] :
        if voisin != 0 :
            nb += 1
    return nb

def force_rappel(i,j,creature):
    k = 100e10
    mi,mj = creature[i][0], creature[j][0]
    l = ((mi[0] - mj[0])**2 + (mi[1] - mj[1])**2)**0.5
    l0 = creature[i][1][j]
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
    f_musc = np.array([[f_musc_periode[i][j%len(f_musc_periode[i])] for j in range(n_interval_time)] for i in range(len(f_musc_periode))])
    print(np.shape(forces_entiers))
    #CI vitesse 
    v = np.zeros((n_nodes, int(n_interval_time), 2))  #shape = (N_noeuds, N_t, 2)
    xy = np.zeros((n_nodes, int(n_interval_time), 2)) #shape = (N_noeuds, N_t, 2)
    a = np.zeros((n_nodes, int(n_interval_time), 2))
    f_eau = np.zeros((n_nodes, int(n_interval_time), 2))
    f_rap = np.zeros((n_nodes, int(n_interval_time), 2))

    print(np.shape(v))
    print(np.shape(a))
    xy[:,0] = pos


    for t in range(1,int(n_interval_time)):
        f_eau[t] = 0 # fonction de xy[:,t-1]
        f_rap[t] = 0 # fonction de v[:t-1] et xy[:,t-1]
        liste_forces = [f_rap, f_eau,f_musc]
        
        a[:,t] = pfd(liste_forces, t)
        
        v[:, t] = v[:, t-1] + dt * a[:, t-1]
        xy[:, t] = xy[:, t-1] + dt * v[:, t-1]

    return (v, xy)

forces = []
pos  = calcul_position(force_initial)[1]
t = 0


while running and t < 10/(1/60):
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
    t += 1

# Quit Pygame
pygame.quit()