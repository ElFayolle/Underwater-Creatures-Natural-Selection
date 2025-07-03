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



meduse = [pos, matrice_adjacence]
med2 = [pos3, matrice_adjacence]




# test de forces aléatoires

#force_initial = [[[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]] , 
              #   [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0][0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]], 
               #   [[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15][0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]]
                 #

force_initial = ([[[0,+150]], [[0,0]], [[0,+150]]])
force_initial2 = ([[[0,-15,]],[[0,0]]])


meduse = [pos, matrice_adjacence,force_initial]
med2 = [pos3, matrice_adjacence, force_initial]

"""
Test simple - le baton
"""
pos3 = np.array([[300,100], [250,150]])
matrice_adjacence3 = np.array([[0,1], [1,0]])
baton = [pos3, matrice_adjacence3, force_initial2]


forces = []
pos, liste_forces  = calcul_position(meduse)[1], calcul_position(meduse)[2]
# pos2 = calcul_position(med2)[1]
# pos3 = calcul_position(baton)[1]
t = 0

#with open("creature_gagnante.json", "r", encoding="utf-8") as f:
    #pos = np.array(json.load(f)[1])

#Test bulles
bubbles = instantiate_bubbles(30)
position_tot={0:pos,1:pos2,2:pos3}

while running and t < DUREE_SIM/(1/60):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            see_creatures(position_tot, event)
            

    screen.fill((0, 35, 120))
    barycentre = centre_de_masse(pos, t)
    offset = get_offset(centre_de_masse(position_tot[CURRENT_CREATURE], t), WIDTH,HEIGHT)
    draw_bubbles(screen,bubbles,offset,barycentre,0,t)
    draw_creature(screen,pos, liste_forces,t, offset)
    # draw_creature(screen,pos2,t,offset)
    # draw_creature(screen,pos3,t,offset)
    font=pygame.font.Font(None, 24)
    text = font.render("N° : " + str(CURRENT_CREATURE)+" distance : " + str(distance(position_tot[CURRENT_CREATURE],t)) ,1,(255,255,255))
    screen.blit(text, (10, 10))
    
    pygame.display.flip()
    clock.tick(60)
    t += 1
    
# Quit Pygame
pygame.quit()