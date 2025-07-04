from physics import *
from UI import *
from params import *
import json


#Interface graphique utilisant pygame
pygame.init()

# Initialisation de l'écran
screen = pygame.display.set_mode((WIDTH, HEIGHT))
background = pygame.image.load("fond.jpg").convert()
background = pygame.transform.scale(background, (WIDTH, HEIGHT)) 
pygame.display.set_caption("Natural Selection Simulation")
clock = pygame.time.Clock()
running = True


"""
Test créature - la Méduse :
"""
pos = np.array([[100,100], [150,150], [200,100]])
pos0_5 = np.array([[50,50], [150,150], [250,50]])
pos2 = np.array([[150,300], [500,300], [600,400]])
matrice_adjacence = np.array([[0,1,0], [1,0,1], [0,1,0]])



# test de forces aléatoires

#force_initial = [[[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]] , 
 #                [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]], 
 #                [[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]]
                 #

forces_méduses_cycliques = [[[0, 100],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, -100],[0, -100],[0, -100],[0, -100],[0, -100],[0, -100],[0, -100]],
                            [[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]],
                            [[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, -100],[0, -100],[0, -100],[0, -100],[0, -100],[0, -100],[0, -100]]]


# Test de forces cycliques pour la méduse
force_musc_periode_tst = np.array([[[0,1000*(1-np.exp(-t/100)-np.exp(-5)*100)] for t in range(500)] for c in range (3)])
force_musc_periode_tst[1,:,:] = 0


force_musc_periode2 = np.array([[[1000*(1-np.exp(-((t+250)%500)/100)-np.exp(-5)*100),1000*(1-np.exp(-(t%500)/100)-np.exp(-5)*100)] for t in range(500)] for c in range (3)])
force_musc_periode2[1,:,:] = 0
force_musc_periode2[2,:,0] *= -1


#Force constantes pour la méduse
force_initial = ([[[0,+150]], [[0,0]], [[0,+150]]])
force_initial2 = ([[[0,-15,]],[[0,0]]])


meduse = [pos, matrice_adjacence,force_musc_periode2]
med2 = [pos0_5, matrice_adjacence, force_musc_periode2]

"""
Test simple - le baton
"""
#pos3 = np.array([[300,100], [250,150]])
#matrice_adjacence3 = np.array([[0,1], [1,0]])
#baton = [pos3, matrice_adjacence3, force_initial2]

forces = []



#Calcul des positions et forces
v,pos,liste_forces,score  = calcul_position(meduse,DT,DUREE_SIM)
v2,pos2,liste_forces2,score2 = calcul_position(med2,DT,DUREE_SIM)


# Initialisation du temps
t = 0

def visualisation_creature(i_generation,i_creature=0):
    with open(f"generations/meilleures_creatures_{i_generation}.json", "r", encoding="utf-8") as f:
        creature = json.load(f)[i_creature][1:]  # de la forme [position,matrice_adjacence,forces] 
    pos = calcul_position([np.array(element) for element in creature])[1]
    return pos
visualisation_creature(1,0)
# Ajout de bulles pour une impression visuelle que la créature évolue dans l'eau
bubbles = instantiate_bubbles(30)
position_tot={0:pos,1:pos2}



# Début de la simulation
while running and t < DUREE_SIM/(DT):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Pour pouvoir changer de créature avec les touches flèches
        if event.type == pygame.KEYDOWN:
            CURRENT_CREATURE=see_creatures(position_tot, event)
            

    screen.blit(background, (0, 0))
    #screen.fill((0, 35, 120))

    #Calcul du barycentre de la créature 
    barycentre = centre_de_masse(position_tot[CURRENT_CREATURE], t)

    #offset pour pouvoir centrer la créature sur l'écran
    offset = get_offset(centre_de_masse(position_tot[CURRENT_CREATURE], t), WIDTH,HEIGHT)
    # Ajout des bulles et de la créature
    draw_bubbles(screen,bubbles,offset)
    draw_creature(screen,pos, liste_forces,t, offset)


    # Print de la distance entre l'origine et le centre de masse de la créature
    font=pygame.font.Font(None, 24)
    text = font.render("N° : " + str(CURRENT_CREATURE)+" distance : " + str(distance(position_tot[CURRENT_CREATURE],t)) ,1,(255,255,255))

    # Affichage des forces sur la créature
    force_liste_nom = ["force eau", "force de réaction", "force musculaire","f_rép_noeud"]
    couleurs_force = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
    screen.blit(text, (10, 10))

    for i, force in enumerate(liste_forces):
        text = font.render(force_liste_nom[i], 1, couleurs_force[i])
        screen.blit(text, (10, 30 + i * 20))

    pygame.display.flip()
    clock.tick(60)
    t += 1
    
# Quit Pygame
pygame.quit()