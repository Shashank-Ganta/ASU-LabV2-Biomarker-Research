import pygame
import random

# initialize Pygame
pygame.init()

# set screen dimensions
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512

# create game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# define game variables
bird_position = [50, 200]
bird_movement = 0
gravity = 0.25
pipe_list = []
SPAWN_PIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWN_PIPE, 1200)
pipe_speed = 3
base_position = 0
score = 0
font = pygame.font.Font(None, 30)

# define game functions
def draw_bird():
    pygame.draw.circle(screen, (255, 255, 255), bird_position, 15)

def move_bird():
    global bird_movement
    bird_movement += gravity
    bird_position[1] += bird_movement

def draw_pipe(pipe_position):
    bottom_pipe = pygame.Rect(pipe_position[0], pipe_position[1], 52, 320)
    top_pipe = pygame.Rect(pipe_position[0], pipe_position[1] - 420, 52, 320)
    pygame.draw.rect(screen, (0, 255, 0), bottom_pipe)
    pygame.draw.rect(screen, (0, 255, 0), top_pipe)

def move_pipes():
    for pipe in pipe_list:
        pipe[0] -= pipe_speed

def spawn_pipe():
    random_height = random.randint(-200, 0)
    return [SCREEN_WIDTH, random_height]

def draw_base():
    base = pygame.Rect(base_position, 450, SCREEN_WIDTH, 62)
    pygame.draw.rect(screen, (150, 150, 150), base)

def move_base():
    global base_position
    base_position -= pipe_speed
    if base_position < -SCREEN_WIDTH:
        base_position = 0

def detect_collision():
    for pipe in pipe_list:
        bottom_pipe = pygame.Rect(pipe[0], pipe[1], 52, 320)
        top_pipe = pygame.Rect(pipe[0], pipe[1] - 420, 52, 320)
        if bottom_pipe.colliderect(bird_image_rect) or top_pipe.colliderect(bird_image_rect):
            return True
    if bird_position[1] >= 450 or bird_position[1] <= 0:
        return True
    return False

# game loop
running = True
while running:
    # event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bird_movement = 0
                bird_movement -= 8
        if event.type == SPAWN_PIPE:
            pipe_list.append(spawn_pipe())

    # draw game elements
    screen.fill((0, 0, 0))
    draw_bird()
    for pipe in pipe_list:
        draw_pipe(pipe)
    draw_base()

    # move game elements
    move_bird()
    move_pipes()
    move_base()

    # detect collision
    bird_image_rect = pygame.Rect(bird_position[0] - 15, bird_position[1] - 15, 30, 30)
    if detect_collision():
        running = False

    # update score
    for pipe in pipe_list:
        if pipe[0] == bird_position[0]:
            score += 1

    # display score
    score_surface = font.render(str(score),
    True, (255, 255, 255))
    score_rect = score_surface.get_rect(center=(SCREEN_WIDTH//2, 50))
    screen.blit(score_surface, score_rect)

    # update display
    pygame.display.update()

    # set frame rate
    clock = pygame.time.Clock()
    clock.tick(60)

