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
    L = [[(100.,100.), [0.,200.]], [(100.,300.), [200.,0.]]]

    for i in L:
        for index,j in enumerate(i):
            if j!= 0:
                pygame.draw.line(screen, (125,50,0), i[0], L[index][0], 10)
        pygame.draw.circle(screen, (255,0,0), i[0], 20) 
    


    pygame.display.flip()
# Quit Pygame
pygame.quit()



