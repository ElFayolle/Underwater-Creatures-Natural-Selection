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

class Creature:
    nodes = []
    links = []

def force_rappel(i,j,creature):
    k = 0.5
    mi,mj = creature.nodes[i], creature.nodes[j]
    l = ((mi[0] - mj[0])**2 + (mi[1] - mj[1])**2)**0.5
    l0 = creature.links[i][j]
    u_ij = (mi - mj) / l
    return -k * (l - l0) * u_ij

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with a color (RGB)
    screen.fill((0, 128, 255))

    # Update the display
    pygame.display.flip()
    

    # Cap the frame rate at 60 FPS
    clock.tick(60)
    L = [[[1,1], [0,2]], [1,3], [2,0]]
    
# Quit Pygame
pygame.quit()



