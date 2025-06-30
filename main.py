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

    accelerations = pfd(forces)


    for i in L:
        for index,j in enumerate(i):
            if j!= 0:
                pygame.draw.line(screen, (125,50,0), i[0], L[index][0], 10)
        pygame.draw.circle(screen, (255,0,0), i[0], 20) 


    pygame.display.flip()
# Quit Pygame
pygame.quit()