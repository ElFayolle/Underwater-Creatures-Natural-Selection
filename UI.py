import pygame
import numpy as np
from utils import *
from params import *


def see_creatures(position_tot,event:pygame.event):
    global CURRENT_CREATURE
    if event.key == pygame.K_LEFT:
            if CURRENT_CREATURE!=0:
                CURRENT_CREATURE-=1
    if event.key == pygame.K_RIGHT:
            if CURRENT_CREATURE<len(position_tot)-1:
                CURRENT_CREATURE+=1
    return None


# Affichage de la créature et des forces associées
def draw_creature(screen,pos, liste_forces, t, offset):
    """screen: écran pygame
    pos: positions des noeuds de la créature (n_nodes,n_t,2)
    liste_forces: liste des forces appliquées sur chaque noeud (3,n_nodes,n_t, 2)
    t: le temps actuel (int)
    offset: décalage pour centrer la créature sur l'écran (2,)
    return: None
    """
    liste_forces = liste_forces/3 #Modification pour que les forces soient plus visibles sur l'écran

    # Affichage des centres de masses aux temps précédents -> trajet de la créature
    for i in range(t):
        pygame.draw.circle(screen,(20,60,120),centre_de_masse(pos,i)+offset,2 )

    # Affichage des noeuds et segments de la créature
    for index in range(1,len(pos)):
        pygame.draw.line(screen,(25, 30, 70),pos[index-1,t]+offset,pos[index,t]+offset,4)
        pygame.draw.circle(screen,(100,189,255),pos[index-1,t]+offset,5)
    pygame.draw.circle(screen,(100,189,255),pos[-1,t]+offset,5)
        
    # Affichage des forces appliquées sur chaque noeud
    for index in range(len(pos)):
        colours_force = [(255,0,0),(0,255,0),(0,0,255)]
        for i in range(len(liste_forces)):
            pygame.draw.line(screen,colours_force[i],pos[index-1,t]+offset,pos[index-1,t]+liste_forces[i][index-1,t]+offset,2)        

    # Affichage de l'origine de la créature
    pygame.draw.circle(screen,(255,0,0),centre_de_masse(pos,0)+offset,3)
    return None


#fonction qui initialise les bulles
def instantiate_bubbles(N_bubbles,rmax=10):
    """N_bubbles: nombre de bulles à générer
    rmax: rayon maximum des bulles
    return: un tableau de bulles de forme (N_bubbles,3) où chaque bulle est représentée par ses coordonnées (x,y) et son rayon r
    """
    bubbles = np.random.rand(N_bubbles,3)
    bubbles[:,0] *= WIDTH
    bubbles[:,1] *= HEIGHT
    bubbles[:,2] *= rmax
    return bubbles


# Affichage des bulles
def draw_bubbles(screen,bubbles,offset):
    """screen: écran pygame
    bubbles: tableau de bulles de forme (N_bubbles,3) 
    offset: décalage pour centrer les bulles sur l'écran (2,)
    return: None
    """
    for index,bubble in enumerate(bubbles):
        pygame.draw.circle(screen,(29,50,140),bubble[:-1]+offset,bubble[2])
    return None