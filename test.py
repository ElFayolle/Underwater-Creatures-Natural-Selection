from physics import *
from UI import *

pygame.init()
# Set up the display
WIDTH, HEIGHT = 800, 600
DUREE_SIM = 30  # Durée de la simulation en secondes
CURRENT_CREATURE = 0
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

#Calcul les longueurs à vide dans une matrice d'adjacence
meduse = [pos, matrice_adjacence]
med2 = [pos3, matrice_adjacence]




#test de forces aléatoires

force_initial = [[[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]] , 
                 [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]], 
                  [[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[-15,15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[15,-15],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]]
                 

#force_initial = ([[[10,0]], [[0,0]], [[0,15]]])


meduse = [pos, matrice_adjacence,force_initial]
med2 = [pos3, matrice_adjacence, force_initial]

"""
Test simple - le baton
"""
#pos = np.array([[100,100], [150,150]])
#matrice_adjacence = np.array([[0,1], [1,0]])
baton = [pos, matrice_adjacence, force_initial]


forces = []
pos  = calcul_position(meduse)[1]
force = calcul_position(meduse)[2]
pos2 = calcul_position(med2)[1]
t = 0


#Test bulles
bubbles = instantiate_bubbles(30)
position_tot=np.array([pos,pos2])

while running and t < DUREE_SIM/(1/60):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            see_creatures(event)
            

    screen.fill((0, 35, 120))
    barycentre = centre_de_masse(pos, t)
    offset = get_offset(centre_de_masse(position_tot[CURRENT_CREATURE], t), WIDTH,HEIGHT)
    draw_bubbles(bubbles,offset,barycentre,0,t)
    draw_creature(pos, force,t, offset)
    #draw_creature(pos2,t,offset)
    font=pygame.font.Font(None, 24)
    text = font.render("distance : " + str(distance(pos,t)),1,(255,255,255))
    coulours_force = [(255,0,0),(0,255,0),(0,0,255)]
    force_name = ["force rappel", "force de réaction", "force musculaire"]
    for i in range(len(coulours_force)):
        text_force = font.render(force_name[i],1,coulours_force[i])
        screen.blit(text_force, (10, 30 + i * 20))
    screen.blit(text, (10, 10))
    
    pygame.display.flip()
    clock.tick(60)
    t += 1
    
# Quit Pygame
pygame.quit()