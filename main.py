from physics import *
from UI import *
from params import *
import json

pygame.init()
# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Natural Selection Simulation")
# Set up the clock for frame rate control
clock = pygame.time.Clock()
# Main loop
running = True

n_nodes = 10
forces = np.zeros((n_nodes, 4))
forces = [ [[[15,12],[0,0]],[[7,4],[1,3]],[[0,0],[0,0]]] , [[[25,22],[10,10]],[[17,14],[1,3]],[[0,0],[0,0]]] ]
accelerations = []

"""
Test créature - la Méduse :
"""
pos = np.array([[100,100], [150,150], [200,100]])
pos2 = np.array([[150,300], [500,300], [600,400]])
pos3 = np.array([[120,120], [150,150], [170,120]]) + np.array([[300, 0], [300, 0], [300, 0]]) # Créature décalée pour le test
matrice_adjacence = np.array([[0,1,0], [1,0,1], [0,1,0]])



# test de forces aléatoires

#force_initial = [[[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]] , 
 #                [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]], 
 #                [[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]]
                 #

forces_méduses_cycliques = [[
[0, 100],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, -100],
[0, -100],
[0, -100],
[0, -100],
[0, -100],
[0, -100],
[0, -100]],[
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0]],

[
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 10],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, 0],
[0, -100],
[0, -100],
[0, -100],
[0, -100],
[0, -100],
[0, -100],
[0, -100]]
]

force_musc_periode_tst = np.array([[[0,1000*(1-np.exp(-t/100)-np.exp(-5)*100)] for t in range(500)] for c in range (3)])
force_musc_periode_tst[1,:,:] = 0

force_musc_periode2 = np.array([[[1000*(1-np.exp(-((t+250)%500)/100)-np.exp(-5)*100),1000*(1-np.exp(-(t%500)/100)-np.exp(-5)*100)] for t in range(500)] for c in range (3)])
force_musc_periode2[1,:,:] = 0
force_musc_periode2[2,:,0] *= -1

force_initial = ([[[0,+150]], [[0,0]], [[0,+150]]])
force_initial2 = ([[[0,-15,]],[[0,0]]])


meduse = [pos, matrice_adjacence,force_musc_periode2]
med2 = [pos2, matrice_adjacence, force_initial]

"""
Test simple - le baton
"""
#pos3 = np.array([[300,100], [250,150]])
#matrice_adjacence3 = np.array([[0,1], [1,0]])
#baton = [pos3, matrice_adjacence3, force_initial2]


forces = []
v,pos,liste_forces,score  = calcul_position(meduse,DT,DUREE_SIM)
v2,pos2,liste_forces2,score2 = calcul_position(med2,DT,DUREE_SIM)
# pos3 = calcul_position(baton)[1]
t = 0

def visualisation_creature(i_generation,i_creature=0):
    with open(f"generations/meilleures_creatures_{i_generation}.json", "r", encoding="utf-8") as f:
        creature = json.load(f)[i_creature][1:]  # de la forme [position,matrice_adjacence,forces] 
    calc = calcul_position([np.array(element) for element in creature])
    return calc
v,pos,liste_forces,score = visualisation_creature(33,0)  # Visualisation de la créature 0 de la génération 0
#Test bulles
bubbles = instantiate_bubbles(30)
position_tot={0:pos,1:pos2}

while running and t < DUREE_SIM/(DT):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            see_creatures(position_tot, event)
            

    screen.fill((0, 35, 120))
    barycentre = centre_de_masse(position_tot[CURRENT_CREATURE], t)
    offset = get_offset(centre_de_masse(position_tot[CURRENT_CREATURE], t), WIDTH,HEIGHT)
    draw_bubbles(screen,bubbles,offset,barycentre,0,t)
    draw_creature(screen,pos, liste_forces,t, offset)
    #draw_creature(screen,pos2,liste_forces2,t,offset)
    # draw_creature(screen,pos3,t,offset)



    font=pygame.font.Font(None, 24)
    text = font.render("N° : " + str(CURRENT_CREATURE)+" distance : " + str(distance(position_tot[CURRENT_CREATURE],t)) ,1,(255,255,255))

    force_liste_nom = ["force eau", "force de réaction", "force musculaire"]
    couleurs_force = [(255,0,0),(0,255,0),(0,0,255)]
    screen.blit(text, (10, 10))

    for i, force in enumerate(liste_forces):
        text = font.render(force_liste_nom[i], 1, couleurs_force[i])
        screen.blit(text, (10, 30 + i * 20))

    pygame.display.flip()
    clock.tick(60)
    t += 1
    
# Quit Pygame
pygame.quit()