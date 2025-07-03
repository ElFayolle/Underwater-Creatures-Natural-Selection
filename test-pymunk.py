import pymunk
import pymunk.pygame_util
import pygame
import math

# Initialisation Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Espace sans gravité
space = pymunk.Space()
space.gravity = (0, 0)

# ==========================
# Création du Soufflet Méduse
# ==========================

def create_soufflet(space, position, arm_length=50, node_mass=1):
    # Corps central
    body_center = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
    body_center.position = position
    shape_center = pymunk.Circle(body_center, 5)
    shape_center.mass = node_mass
    space.add(body_center, shape_center)

    # Deux bras avec masses
    angles = [math.radians(60), math.radians(-60)]
    arms = []
    motors = []

    for angle in angles:
        x = position[0] + arm_length * math.cos(angle)
        y = position[1] + arm_length * math.sin(angle)

        body_end = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
        body_end.position = (x, y)
        shape_end = pymunk.Circle(body_end, 5)
        shape_end.mass = node_mass
        space.add(body_end, shape_end)

        # Fixer la distance avec un bras rigide
        joint = pymunk.PinJoint(body_center, body_end, (0,0), (0,0))
        space.add(joint)

        # Moteur pour contrôler l'angle
        motor = pymunk.SimpleMotor(body_center, body_end, 0)
        motor.max_force = 100000
        space.add(motor)

        arms.append(body_end)
        motors.append(motor)

    return body_center, arms, motors

# Création de la méduse soufflet
center, arm_bodies, arm_motors = create_soufflet(space, (400, 300))

# Oscillation paramétrique
time = 0
frequency = 2  # Battements par seconde
amplitude = 4  # Vitesse angulaire

# ==========================
# Boucle principale
# ==========================

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    time += 1/60.0

    # Oscillation symétrique
    speed = amplitude * math.sin(2 * math.pi * frequency * time)
    arm_motors[0].rate = speed
    arm_motors[1].rate = -speed

    # Impulsion vers le haut à chaque fermeture pour simuler propulsion
    if math.sin(2 * math.pi * frequency * time) > 0.9:
        center.apply_impulse_at_local_point((0, -100))

    screen.fill((255, 255, 255))
    space.debug_draw(draw_options)
    space.step(1/60.0)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
