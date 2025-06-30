import pygame
import numpy as np
import functions

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
forces_aleatoires = []

def force_musculaire(i, creature, forces_aleatoires):
    nb_vois = nombre_de_voisins(i, creature)
    for index, voisin in enumerate(creature[i][1]) :
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
    l0 = creature[i][1 ][j]
    u_ij = np.array((mi - mj)) / l
    return -k * (l - l0) * u_ij

def pfd(forces):
    m = 1
    accelerations_t = np.sum(forces, axis=1)  # Sum forces for each node
    accelerations_t /= m
    return accelerations_t

def check_line_cross(creature:np.ndarray)->np.ndarray: # Fonction naïve pour empêcher les croisements de segments
    l = len(creature)

    #Tableau booléens d'intersection du segment i "[AB]" au segment j "[CD]""
    pt_intersec = np.zeros((l,l)) 

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




while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with a color (RGB)
    screen.fill((0, 128, 255))

    # Update the display
    

    # Cap the frame rate at 60 FPS
    clock.tick(1)
    L = [[(100,100), [0,200]], [(10,300), [200,0]]]

    accelerations_avant = accelerations.copy()
    accelerations_t = pfd(forces)



    for i in L:
        for index,j in enumerate(i):
            if j!= 0:
                pygame.draw.line(screen, (125,50,0), i[0], L[index][0], 10)
        pygame.draw.circle(screen, (255,0,0), i[0], 20) 
 

    pygame.display.flip()
# Quit Pygame
pygame.quit()