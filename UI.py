import pygame
import numpy as np

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