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

len_nodes = 10
forces = np.zeros((len_nodes, 4))
accelerations = []




def force_musculaire(i, creature, forces_aleatoires):
    nb_vois = nombre_de_voisins(i, creature)
    for voisin, index in enumerate(creature[i][1]) :
        if voisin != 0 :
            forces[index][1] += forces_aleatoires[i] / nb_vois

def nombre_de_voisins(i, creature):
    nb = 0
    for voisin in creature[i][1] :
        nb += 1
    return nb

def force_rappel(i,j,creature):
    k = 0.5
    mi,mj = creature[i][0], creature[j][0]
    l = ((mi[0] - mj[0])**2 + (mi[1] - mj[1])**2)**0.5
    l0 = creature[i][1][j]
    u_ij = np.array((mi - mj)) / l
    return -k * (l - l0) * u_ij


def pfd(forces, mass=1):
    """
    forces: (n_nodes, n_interval_time, n_forces, 2)
    retourne : accelerations of shape (n_nodes, n_interval_time, 2)
    """
    total_force = np.sum(forces, axis=2)  # shape: (n_nodes, n_interval_time, 2)
    accelerations = total_force / mass
    return accelerations




#test de forces aléatoires
 
#calcul_position(np.Array(), float #pas de temps, float #temps de simul, int #nombre de noeuds)
def calcul_position(forces, dt = 1/60, T = 10., n_nodes=2):

    pos = [[100,100], [100,300]] #test pos initial pour 2 noeuds
    neigh = [[0,200], [200,0]]   

    #Nombre d'itérations
    n_interval_time = int(T/dt)  

    # Forces qui boucle sur la période cyclique de force donnée
    forces_entiers = np.array([forces[i%len(forces)] for i in range(n_interval_time)])

    #CI vitesse 
    v = np.zeros((n_nodes, int(n_interval_time), 2))  #shape = (N_noeuds, N_t, 2)
    xy = np.zeros((n_nodes, int(n_interval_time), 2)) #shape = (N_noeuds, N_t, 2)
    a = pfd(forces_entiers)
    

    print(np.shape(v))
    print(np.shape(a))
    xy[:,0] = pos


    for t in range(1,int(n_interval_time)):
        v[:, t] = v[:, t-1] + dt * a[:, t-1]
        pos[:, t] = pos[:, t-1] + dt * v[:, t-1]

    return (v, pos)

forces = []
pos  = calcul_position(forces)
t = 0


while running and t < 10/(1/60):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 128, 255))

    # Ligne entre les deux points
    pygame.draw.line(screen, (125, 50, 0), pos[0, t], pos[1, t], 10)

    # Cercles pour chaque point
    for i in range(n_nodes):
        pygame.draw.circle(screen, (255, 0, 0), pos[i, t].astype(int), 20)

    pygame.display.flip()
    clock.tick(60)
    t += 1
# Quit Pygame
pygame.quit()