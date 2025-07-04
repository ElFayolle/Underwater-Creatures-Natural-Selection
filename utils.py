import pygame
import numpy as np

"""Filtres gaussiens pour transformer les forces d'une génération à l'autre (augmenter un tick de force et les ticks voisins de manière lissée)"""
@np.vectorize
def gaussienne(x,ampl,mu=0,sigma=2):
    return ampl*np.exp(-(x-mu)**2/(2*sigma**2))

#-3 écarts types et +3 écarts types = 99% des valeurs de la gaussienne.
def ampl_plus(signe,ampl,mu=0,sigma=2):
    return signe*(gaussienne(np.arange(-3*sigma,3*sigma),ampl)+np.ones(6*sigma))
def ampl_moins(mu=0,sigma=2):
    return -gaussienne(np.arange(-3*sigma,3*sigma),1)+np.ones(6*sigma)

#force[dot-6:dot+6]*= ampl_moins()     <-- Basile voila comment appliquer en gros


# Calcul la position du centre de masse de la créature à un instant t
def centre_de_masse(position:np.ndarray,t):
    """position: positions des noeuds de la créature (n_nodes, n_t, 2)
    t: le temps actuel (int)
    return : le centre de masse de la créature à l'instant t (2,)"""
    C = np.mean(position[:, t], axis=0) 
    return C



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


# Calcul de la vitesse moyenne de la créature à un instant t
def vitesse_moyenne(vitesse, t):
    """
    vitesse: (n_nodes, n_interval_time, 2)
    t: float
    retourne : moyenne des vitesses sur le temps t (2,)
    """
    vitesse_moy = np.sum(vitesse[:, t], axis=0) 
    return vitesse_moy

# Calcul de l'énergie cinétique de la créature à un instant t
def energie_cinetique(vitesse, t, masse = 1):
    """
    vitesse: (n_nodes, n_interval_time, 2)
    retourne : énergie cinétique de la créature (float)
    """
    vitesse_norm = np.linalg.norm(vitesse[:, int(t)], axis=1)  # norme de la vitesse pour chaque noeud
    energie = 0.5 * masse * np.sum(vitesse_norm**2)  # somme des énergies cinétiques
    return energie

# Calcul de la distance entre le centre de masse à l'instant t et l'origine de la créature (centre de masse à l'instant 0)
def distance(position,t):
    """position: positions des noeuds de la créature (n_nodes, n_t, 2)
    t: le temps actuel (int)
    return : distance entre barycentre et origine (float)
    """
    return round(np.linalg.norm(centre_de_masse(position,t)-centre_de_masse(position,0)),0)


# Calcul du score de la créature (pour l'instant uniquement la distance parcourue)
def calcul_score(energie, distance, taille):
    """energie: énergie cinétique de la créature (float)
    distance: distance parcourue par la créature (float)
    taille: taille de la créature (float)
    retourne : score de la créature (float)"""
    score = 2/3*distance/max(distance) + 1/3* energie/taille * max(taille/energie)
    return score


# Calcul de l'offset pour centrer la créature sur l'écran
def get_offset(barycentre, screen_width, screen_height):
    """barycentre: centre de masse de la créature (2,)
    screen_width: largeur de l'écran (int)
    screen_height: hauteur de l'écran (int)
    return : offset pour centrer la créature sur l'écran (2,)"""

    screen_center = np.array([screen_width // 2, screen_height // 2])
    return screen_center - barycentre


# Calcul des longueurs à vide à partir de la matrice d'adjacence
def neighbors(pos, matrice_adjacence):
    """pos: positions des noeuds de la créature (n_nodes, 2)
    matrice_adjacence: matrice d'adjacence de la créature (n_nodes, n_nodes)
    return : matrice des distances entre les noeuds (n_nodes, n_nodes)"""
    l0 = np.zeros((len(pos), len(pos)))
    for i,point in enumerate(matrice_adjacence):
        for j,voisin in enumerate(point):
            if voisin != 0:
                l0[i,j] = np.linalg.norm(pos[i]-pos[j])
    return l0

# Fonction naïve pour empêcher les croisements de segments - on ne l'utilise pas pour l'instant
def check_line_cross(position:np.ndarray,t)->np.ndarray: 
    
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


