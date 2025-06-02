import os
import torch
import pygame
import time
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE
from sprite_strip_anim import SpriteStripAnim
import numpy as np
import math

# ==== CONFIGURAÇÃO ====
SCORE_FILE = "score/scores.txt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 480
MIDDLE_X = 280
GROUND_Y = 200
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GAME_TITLE = 'Horse Racing RL - AI Only'
FOLDER_PREFIX = 'images/'
BACKGROUND_PATH = FOLDER_PREFIX + 'background.png'
HORSE_RIDER_SS_PATH = FOLDER_PREFIX + 'horse_rider_spritesheet.png'
DOG_SS_PATH = FOLDER_PREFIX + 'SleepDog.png'
PIG_SS_PATH = FOLDER_PREFIX + 'PigIdle.png'
WOLF_SS_PATH = FOLDER_PREFIX + 'TimberWolf.png'
TCHICK_SS_PATH = FOLDER_PREFIX + 'TinyChick.png'
PIPE_SS_PATH = FOLDER_PREFIX + 'Pipe.png'
ROCK_SS_PATH = FOLDER_PREFIX + 'rock.png'
FPS = 30
TICK_FRAMES = 4
GRAVITY = 3
HORSE_RIDER_SCALE = (130, 130)
ANIMAL_SCALE = (64, 64)
PIPE_SCALE = (75, 350)
MEDIA_PREFIX = 'media/'
MAX_SCROLL_SPEED = 12
BASE_ACCELERATION = 0.02
SPEED_MULTIPLIER = 1.0
SPEED_INCREASE_INTERVAL = 2000
ANIMAL_SHOW_FLAGS = [0, 1]
ANIMAL_SHOW_WEIGHTS = [0.40, 0.60]
ROCK_SHOW_FLAGS = [0, 1]
ROCK_SHOW_WEIGHTS = [1.0, 0.0]
ROCK_HEIGHTS = [GROUND_Y - 25, GROUND_Y - 100]
MIN_DISTANCE = 80
MIN_SPAWN_DISTANCE = 400
SONAR_RANGE = 1500
STATE_BUCKETS = [5, 5, 2, 5, 3]
ACTIONS = [0, 1, 2]
PIPE_INDEX = 3

class QNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(8, 8)
        self.fc2 = torch.nn.Linear(8, 8)
        self.fc3 = torch.nn.Linear(8, action_size)
    def forward(self, x):
        if x.shape[1] != 8:
            x_pad = torch.zeros((x.shape[0], 8), device=x.device, dtype=x.dtype)
            x_pad[:, :x.shape[1]] = x
            x = x_pad
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    def act(self, state_tensor):
        with torch.no_grad():
            q_values = self.forward(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        return action

def discretize(value, buckets, min_val, max_val):
    if value <= min_val:
        return 0
    if value >= max_val:
        return buckets - 1
    ratio = (value - min_val) / (max_val - min_val)
    return int(ratio * (buckets - 1))

def get_sonar_distance(game_state, horse_x, horse_y):
    horse_center_x = horse_x + HORSE_RIDER_SCALE[0] // 2
    horse_center_y = horse_y + HORSE_RIDER_SCALE[1] // 2
    min_dist = float('inf')
    if game_state['is_animal_running']:
        if game_state['current_animal_anim'] == PIPE_INDEX:
            animal_center = (game_state['animal_x'] + PIPE_SCALE[0] // 2, game_state['animal_y'] + PIPE_SCALE[1] // 2)
        else:
            animal_center = (game_state['animal_x'] + ANIMAL_SCALE[0] // 2, game_state['animal_y'] + ANIMAL_SCALE[1] // 2)
        dist_to_animal = math.hypot(animal_center[0] - horse_center_x, animal_center[1] - horse_center_y)
        if dist_to_animal < min_dist:
            min_dist = dist_to_animal
    if min_dist == float('inf'):
        return -1
    return min_dist

def load_best_model():
    best_model_path = None
    for i in range(1000):
        path = f"treinamento/model/best_model_{i}.pth"
        if os.path.exists(path):
            best_model_path = path
    if best_model_path is None:
        raise RuntimeError("Nenhum modelo treinado encontrado!")
    state_size = 8
    model = QNetwork(state_size, len(ACTIONS)).to(DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()
    print(f"Usando modelo: {best_model_path}")
    return model

def main():
    pygame.init()
    display_surf = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(GAME_TITLE)
    frame_per_sec = pygame.time.Clock()
    background_image = pygame.image.load(BACKGROUND_PATH).convert()
    from sprite_strip_anim import SpriteStripAnim
    rider_run_anim = SpriteStripAnim(HORSE_RIDER_SS_PATH, (0,130,64,64), 3, -1, True, TICK_FRAMES, HORSE_RIDER_SCALE)
    rider_crouch_anim = SpriteStripAnim(HORSE_RIDER_SS_PATH, (0,0,64,64), 3, -1, True, TICK_FRAMES, HORSE_RIDER_SCALE)
    animal_animations = [
        SpriteStripAnim(WOLF_SS_PATH, (0,0,16,16), 4, -1, True, TICK_FRAMES, ANIMAL_SCALE),
        SpriteStripAnim(TCHICK_SS_PATH, (0,0,16,16), 4, -1, True, TICK_FRAMES, ANIMAL_SCALE),
        SpriteStripAnim(PIG_SS_PATH, (0,0,64,64), 4, -1, True, TICK_FRAMES, ANIMAL_SCALE),
        SpriteStripAnim(PIPE_SS_PATH, (0,0,75,219), 1, -1, True, TICK_FRAMES, PIPE_SCALE),
    ]
    model = load_best_model()
    # Estado inicial
    game_state = {
        'horse_rider_x': MIDDLE_X,
        'horse_rider_y': GROUND_Y,
        'rider_y_speed': 0,
        'is_ducking': False,
        'duck_timer': 0,
        'scroll_speed': 10.0,
        'background_x': 0,
        'animal_x': SCREEN_WIDTH,
        'animal_y': GROUND_Y + 50,
        'rock_x': SCREEN_WIDTH,
        'rock_y': GROUND_Y - 25,
        'is_animal_running': False,
        'is_rock_running': False,
        'current_animal_anim': None,
        'score': 0
    }
    running = True
    last_speed_increase = time.time() * 1000
    speed_multiplier = SPEED_MULTIPLIER
    font = pygame.font.SysFont('Arial', 40)
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
        # Atualiza ambiente
        now = time.time() * 1000
        if now - last_speed_increase > SPEED_INCREASE_INTERVAL:
            speed_multiplier *= 1.05
            last_speed_increase = now
        game_state['scroll_speed'] = min(game_state['scroll_speed'] + BASE_ACCELERATION * speed_multiplier, MAX_SCROLL_SPEED)
        game_state['background_x'] -= int(game_state['scroll_speed'] * speed_multiplier)
        if game_state['background_x'] <= -background_image.get_width():
            game_state['background_x'] = 0
        current_scroll = abs(game_state['background_x'])
        if not game_state['is_animal_running'] and current_scroll >= MIN_SPAWN_DISTANCE:
            game_state['is_animal_running'] = True
            game_state['animal_x'] = SCREEN_WIDTH
            game_state['current_animal_anim'] = np.random.choice([0,1,2,3], p=[0.2,0.2,0.2,0.4])
            if game_state['current_animal_anim'] == PIPE_INDEX:
                game_state['animal_y'] = GROUND_Y - 315
            else:
                game_state['animal_y'] = GROUND_Y + 50
        game_state['animal_x'] -= int(game_state['scroll_speed'] * speed_multiplier)
        if game_state['animal_x'] < -ANIMAL_SCALE[0]:
            game_state['is_animal_running'] = False
        # Estado para IA
        horse_height = discretize(game_state['horse_rider_y'], STATE_BUCKETS[0], 0, SCREEN_HEIGHT)
        dist_to_animal = discretize(game_state['animal_x'] - game_state['horse_rider_x'], STATE_BUCKETS[1], 0, SCREEN_WIDTH) if game_state['is_animal_running'] else STATE_BUCKETS[1] - 1
        animal_active = 1 if game_state['is_animal_running'] else 0
        dist_to_rock = STATE_BUCKETS[3] - 1
        rock_active = 0
        sonar_dist = get_sonar_distance(game_state, game_state['horse_rider_x'], game_state['horse_rider_y'])
        sonar_norm = sonar_dist / SONAR_RANGE if sonar_dist >= 0 else -1
        is_pipe = 1 if game_state['is_animal_running'] and game_state['current_animal_anim'] == PIPE_INDEX else 0
        close_pipe = 1 if is_pipe and 0 < (game_state['animal_x'] - game_state['horse_rider_x']) < 200 else 0
        state_vector = [horse_height, dist_to_animal, animal_active, dist_to_rock, rock_active, sonar_norm, is_pipe, close_pipe]
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(DEVICE)
        action = model.act(state_tensor)
        # Executa ação
        if action == 1 and game_state['horse_rider_y'] == GROUND_Y:
            game_state['rider_y_speed'] = 30
            game_state['is_ducking'] = False
            game_state['duck_timer'] = 0
        elif action == 2 and game_state['horse_rider_y'] == GROUND_Y:
            if not game_state['is_ducking']:
                game_state['is_ducking'] = True
                game_state['duck_timer'] = 1
        else:
            if game_state['is_ducking']:
                game_state['duck_timer'] += 1
                if game_state['duck_timer'] > 30:
                    game_state['is_ducking'] = False
                    game_state['duck_timer'] = 0
        game_state['horse_rider_y'] -= game_state['rider_y_speed']
        game_state['rider_y_speed'] -= GRAVITY
        if game_state['horse_rider_y'] > GROUND_Y:
            game_state['horse_rider_y'] = GROUND_Y
            game_state['rider_y_speed'] = 0
        game_state['score'] += int(game_state['scroll_speed'] * speed_multiplier)
        # --- Colisão ---
        if game_state['is_ducking']:
            rider_rect = pygame.Rect(game_state['horse_rider_x'], game_state['horse_rider_y'] + 40, 100, 60)
        else:
            rider_rect = pygame.Rect(game_state['horse_rider_x'], game_state['horse_rider_y'], 100, 100)
        if game_state['is_animal_running'] and game_state['current_animal_anim'] is not None:
            if game_state['current_animal_anim'] == PIPE_INDEX:
                animal_rect = pygame.Rect(game_state['animal_x'], game_state['animal_y'], PIPE_SCALE[0], PIPE_SCALE[1])
            else:
                animal_rect = pygame.Rect(game_state['animal_x'], game_state['animal_y'], ANIMAL_SCALE[0], ANIMAL_SCALE[1])
        else:
            animal_rect = pygame.Rect(game_state['animal_x'], game_state['animal_y'], ANIMAL_SCALE[0], ANIMAL_SCALE[1])
        collision = rider_rect.colliderect(animal_rect)
        if collision:
            # Exibe mensagem de game over e pausa
            text_surface = font.render('GAME OVER!', True, RED)
            display_surf.blit(text_surface, (SCREEN_WIDTH//2-120, SCREEN_HEIGHT//2-40))
            pygame.display.update()
            time.sleep(1)
            # Reinicia o estado do jogo automaticamente
            game_state = {
                'horse_rider_x': MIDDLE_X,
                'horse_rider_y': GROUND_Y,
                'rider_y_speed': 0,
                'is_ducking': False,
                'duck_timer': 0,
                'scroll_speed': 10.0,
                'background_x': 0,
                'animal_x': SCREEN_WIDTH,
                'animal_y': GROUND_Y + 50,
                'rock_x': SCREEN_WIDTH,
                'rock_y': GROUND_Y - 25,
                'is_animal_running': False,
                'is_rock_running': False,
                'current_animal_anim': None,
                'score': 0
            }
            last_speed_increase = time.time() * 1000
            speed_multiplier = SPEED_MULTIPLIER
            continue
        # Renderização
        display_surf.fill(BLACK)
        display_surf.blit(background_image, (game_state['background_x'], 0))
        display_surf.blit(background_image, (game_state['background_x'] + background_image.get_width(), 0))
        if game_state['is_animal_running'] and game_state['current_animal_anim'] is not None:
            display_surf.blit(animal_animations[game_state['current_animal_anim']].next(), (game_state['animal_x'], game_state['animal_y']))
        if game_state['is_ducking']:
            display_surf.blit(rider_crouch_anim.next(), (game_state['horse_rider_x'], game_state['horse_rider_y']))
        else:
            display_surf.blit(rider_run_anim.next(), (game_state['horse_rider_x'], game_state['horse_rider_y']))
        font = pygame.font.SysFont('Arial', 40)
        text_surface = font.render(f'Score: {game_state["score"]}', True, WHITE)
        display_surf.blit(text_surface, (20, 20))
        # Mostrar scroll speed
        speed_surface = font.render(f'Scroll Speed: {game_state["scroll_speed"]:.2f}', True, YELLOW)
        display_surf.blit(speed_surface, (20, 70))
        pygame.display.update()
        frame_per_sec.tick(FPS)
    pygame.quit()

if __name__ == "__main__":
    main()
