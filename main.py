import pygame
import numpy as np

pygame.init()
# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Natural Selection Simulation")
# Set up the clock for frame rate control
clock = pygame.time.Clock()
fond = pygame.image.load("fond.jpg").convert()
fond = pygame.transform.scale(fond, (width, height))
# Main loop
running = True

len_nodes = 10
forces = np.zeros((len_nodes, 4))
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
    k = 0.5
    mi,mj = creature[0][i], creature[0][j]
    l = ((mi[0] - mj[0])**2 + (mi[1] - mj[1])**2)**0.5
    l0 = creature[1][i][j]
    u_ij = np.array((mi - mj)) / l
    return -k * (l - l0) * u_ij

def pfd(forces):
    m = 1
    accelerations_t = np.sum(forces, axis=1)  # Sum forces for each node
    accelerations_t /= m
    return accelerations_t




#test de forces aléatoires
forces 
#calcul_position(np.Array(), float #pas de temps, float #temps de simul, int #nombre de noeuds)
def calcul_position(forces, dt = 1/60, T = 10., n_nodes):

    pos = [[100,100], [100,300]]
    neigh = [[0,200], [200,0]]
    a = pfd(forces)
    n_interval_time = T/dt    
    v = np.zeros(n_nodes, n_interval_time, 2)
    xy = np.zeros(n_nodes, n_interval_time, 2)
    xy[:,0] = pos


    for t in range(1,n_interval_time):
        v[:,t] = (dt*a[-1] + v[-1])
        pos[t] = (dt*v[-1] + pos[-1])

    return(v, pos)



while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with a color (RGB)
    screen.blit(fond, (0,0))

    # Update the display
    

    # Cap the frame rate at 60 FPS
    pos = [[100,100], [100,300]]
    neigh = [[0,200], [200,0]]
    clock.tick(60)
    L = [[(100,100), [0,200]], [(10,300), [200,0]]]



    


    for i, point in enumerate(pos):
        for j, voisin in enumerate(point):
            if voisin!= 0:
                pygame.draw.line(screen, (125,50,0), pos[i], pos[j], 10)
        pygame.draw.circle(screen, (255,0,0), pos[i], 20) 


    pygame.display.flip()
# Quit Pygame
pygame.quit()