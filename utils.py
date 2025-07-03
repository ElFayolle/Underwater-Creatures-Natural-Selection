import pygame
import numpy as np

def centres_de_masse(positions_tot:np.ndarray,t):

    C_tot = np.zeros((len(positions_tot,2)))
    for index,pos in enumerate(positions_tot): # Boucle for berk mais je ne trouve rien de pratique
        C_tot[index] = centre_de_masse(pos) 
    return C_tot

def centre_de_masse(position:np.ndarray,t):
    """Calcul du centre de masse de chaque créature à un instant t"""
    C = np.mean(position[:, t], axis=0) 
    return C

def somme_normales_locales(position,neighbours,t):
    dico_normales = normales_locales(position, neighbours, t)
    normales_totales = np.zeros((len(position), 2))
    eps=1e-6
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
                cos_theta = np.dot(BA,np.array([1,0]))/np.linalg.norm(BA)
                sin_theta = np.dot(BA,np.array([0,1]))/np.linalg.norm(BA)
                normale_locale = +cos_theta*np.array([0,1]) - sin_theta*np.array([1,0])
                d[(index,i)] = normale_locale
    return d  

def vitesse_moyenne(vitesse, t):
    """
    vitesse: (n_nodes, n_interval_time, 2)
    t: float
    retourne : moyenne des vitesses sur le temps t
    """
    vitesse_moy = np.sum(vitesse[:, t], axis=0)  # liste de 2 éléments : v_moy_x, v_moy_y
    return vitesse_moy


def energie_cinetique(vitesse, t, masse = 1):
    """
    vitesse: (n_nodes, n_interval_time, 2)
    retourne : énergie cinétique de la créature
    """
    vitesse_norm = np.linalg.norm(vitesse[:, int(t)], axis=1)  # norme de la vitesse pour chaque noeud
    energie = 0.5 * masse * np.sum(vitesse_norm**2)  # somme des énergies cinétiques
    return energie



def distance(position,t):
    return round(np.linalg.norm(centre_de_masse(position,t)-centre_de_masse(position,0)),0)

def score(energie, distance, taille):
    score = 2/3*distance/max(distance) + 1/3* energie/taille * max(taille/energie)

def iter_score(position, vitesse): # Calcule les grandeurs liées au score d'UNE créature
    masse = len(position)   # masse et taille sont identiques ici
    energie = np.sum(np.square(vitesse))*masse*0.5
    distance = np.linalg.norm(centre_de_masse(position,0) - centre_de_masse(position,-1))
    return energie,distance,masse #return energie distance taille

def selection(score_total:np.ndarray,force_total,):

    return None

def get_offset(barycentre, screen_width, screen_height):
    screen_center = np.array([screen_width // 2, screen_height // 2])
    return screen_center - barycentre


def neighbors(pos, matrice_adjacence):
    l0 = np.zeros((len(pos), len(pos)))
    for i,point in enumerate(matrice_adjacence):
        for j,voisin in enumerate(point):
            if voisin != 0:
                l0[i,j] = np.linalg.norm(pos[i]-pos[j])
    return l0


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


