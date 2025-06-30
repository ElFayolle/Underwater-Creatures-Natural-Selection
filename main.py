import pygame

pygame.init()
# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Natural Selection Simulation")
# Set up the clock for frame rate control
clock = pygame.time.Clock()
# Main loop
running = True

forces = []
accelerations = []
a = []
v = []
dx = []


def force_rappel(i,j,creature):
    k = 0.5
    mi,mj = creature[i][0], creature[j][0]
    l = ((mi[0] - mj[0])**2 + (mi[1] - mj[1])**2)**0.5
    l0 = creature[i][1][j]
    u_ij = (mi - mj) / l
    return -k * (l - l0) * u_ij

def pfd(forces):
    m = 1
    accelerations = []
    for i in range(len(forces)):
        forces_i = [0, 0]
        for j in range(len(forces[i])):
            forces_i += forces[i][j]
        accelerations.append(forces_i / m)
    return accelerations
a.append(accelerations)


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
    pos = [[100,100], [100,300]]
    neigh = [[0,200], [200,0]]

    dt = 1/60

    
    accelerations_t = pfd(forces)
    a.append(accelerations) 
    v.append(dt*a[-1] + v[-1])


    


    for i, point in enumerate(pos):
        for j, voisin in enumerate(point):
            if voisin!= 0:
                pygame.draw.line(screen, (125,50,0), pos[i], pos[j], 10)
        pygame.draw.circle(screen, (255,0,0), pos[i], 20) 

    pygame.display.flip()
# Quit Pygame
pygame.quit()