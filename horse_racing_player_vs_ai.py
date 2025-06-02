from random import choices, randint, choice
import time
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_s, K_h, K_UP, K_DOWN, K_SPACE
from pygame import mixer
import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import collections
import pygame

# ==== CONFIGURAÇÃO ====
ENABLE_NICKNAME = True
SCORE_FILE = "score/scores.txt"

# Torch device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 960
SPLIT_SCREEN = True  # Dividir a tela em duas partes (Jogador vs IA)
TOP_BOUNDARY = SCREEN_HEIGHT // 2  # Limite da tela dividida horizontalmente

MIDDLE_X_PLAYER = 280
MIDDLE_X_AI = 280  # Ambos na mesma posição X
GROUND_Y = 200
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
GAME_TITLE = 'Horse Racing - Player vs AI'
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
PIPE_SCALE = (75, 350)  # Pipe scale - wider and much taller to extend vertically
ROCK_SCALE = (50, 50)
MEDIA_PREFIX = 'media/'
MAX_SCROLL_SPEED = 13
BASE_ACCELERATION = 0.02
SPEED_MULTIPLIER = 1.2
SPEED_INCREASE_INTERVAL = 2000
ANIMAL_SHOW_FLAGS = [0, 1]
ANIMAL_SHOW_WEIGHTS = [0.40, 0.60]  # Probabilidade de animais aparecerem
ROCK_SHOW_FLAGS = [0, 1]
ROCK_SHOW_WEIGHTS = [1.0, 0.0]  # Pedras nunca vão aparecer (100% chance de ser 0)
ROCK_HEIGHTS = [GROUND_Y - 25, GROUND_Y - 100]
MIN_DISTANCE = 80
MIN_SPAWN_DISTANCE = 400  # Nova constante
SONAR_RANGE = 1500  # raio de detecção do sonar em pixels

# RL parameters
STATE_BUCKETS = [5, 5, 2, 5, 3]
# Estado discretizado:
# 0: altura cavalo (5 buckets)
# 1: distância para animal (5 buckets)
# 2: animal ativo (2 buckets)
# 3: distância para rocha (5 buckets)
# 4: rocha ativa (3 buckets)

# Novas ações: 0 = idle, 1 = pular, 2 = abaixar
ACTIONS = [0, 1, 2]
GAMMA = 0.95
ALPHA = 0.1

# Definindo a rede neural
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # O modelo treinado espera entrada de tamanho 8
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, action_size)

    def forward(self, x):
        # Garante que a entrada tenha shape (batch, 8)
        if x.shape[1] != 8:
            # Preenche com zeros extras se necessário
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

def get_top_scores(limit=5):
    scores = []
    if os.path.exists(SCORE_FILE):
        with open(SCORE_FILE, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    scores.append((parts[0], int(parts[1])))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:limit]

def save_score(nick, pontos):
    try:
        os.makedirs(os.path.dirname(SCORE_FILE), exist_ok=True)
        with open(SCORE_FILE, "a") as f:
            f.write(f"{nick} {pontos}\n")
    except Exception as e:
        print(f"Erro ao salvar pontuação: {e}")

# Inicialização do Pygame
pygame.init()
DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(GAME_TITLE)
frame_per_sec = pygame.time.Clock()

# Carregar assets
background_image = pygame.image.load(BACKGROUND_PATH).convert()
mixer.init()
# Música do menu
mixer.music.load(MEDIA_PREFIX + 'intro.mp3')
mixer.music.play(loops=-1)
mixer.music.set_volume(0.05)

# Animações
from sprite_strip_anim import SpriteStripAnim
rider_run_anim = SpriteStripAnim(HORSE_RIDER_SS_PATH, (0,130,64,64), 3, -1, True, TICK_FRAMES, HORSE_RIDER_SCALE)
rider_crouch_anim = SpriteStripAnim(HORSE_RIDER_SS_PATH, (0,0,64,64), 3, -1, True, TICK_FRAMES, HORSE_RIDER_SCALE)
rock_anim = SpriteStripAnim(ROCK_SS_PATH, (0,0,50,50), 1, -1, True, TICK_FRAMES, ROCK_SCALE)

# Animais aleatórios
animal_animations = [
    SpriteStripAnim(WOLF_SS_PATH, (0,0,16,16), 4, -1, True, TICK_FRAMES, ANIMAL_SCALE),  # índice 0: lobo
    SpriteStripAnim(TCHICK_SS_PATH, (0,0,16,16), 4, -1, True, TICK_FRAMES, ANIMAL_SCALE),  # índice 1: pintinho
    SpriteStripAnim(PIG_SS_PATH, (0,0,64,64), 4, -1, True, TICK_FRAMES, ANIMAL_SCALE),  # índice 2: porco
    SpriteStripAnim(PIPE_SS_PATH, (0,0,75,219), 1, -1, True, TICK_FRAMES, PIPE_SCALE),  # índice 3: tubo (pipe)
]

# Índice do tubo (pipe) na lista de animações
PIPE_INDEX = 3

# Sistema de fontes
pygame.font.init()
main_font = pygame.font.SysFont('Arial', 40)
small_font = pygame.font.SysFont('Arial', 30)
medium_font = pygame.font.SysFont('Arial', 36)

class GameState:
    def __init__(self, side="player"):
        self.side = side  # 'player' ou 'ai'
        self.reset()
        
    def reset(self):
        if self.side == "player":
            self.horse_rider_x = MIDDLE_X_PLAYER
        else:
            self.horse_rider_x = MIDDLE_X_AI
        
        self.horse_rider_y = GROUND_Y
        self.rider_y_speed = 0
        self.is_ducking = False
        self.duck_timer = 0  # tempo em frames abaixado
        self.scroll_speed = 11.0
        self.background_x = 0
        self.animal_x = SCREEN_WIDTH
        self.animal_y = GROUND_Y + 50
        self.rock_x = SCREEN_WIDTH
        self.rock_y = choice(ROCK_HEIGHTS)
        self.start_time = time.time()
        self.last_speed_increase = time.time()*1000
        self.game_over = False
        self.score = 0
        self.speed_multiplier = SPEED_MULTIPLIER
        self.current_high_score = get_top_scores(1)[0][1] if get_top_scores(1) else 0
        self.nickname = "PLAYER" if self.side == "player" else "AI"
        self.is_animal_running = False
        self.is_rock_running = False  # Garantimos que as pedras não comecem ativas
        self.current_animal_anim = None
        self.last_animal_spawn_x = -MIN_SPAWN_DISTANCE  # Controla spawn animal separadamente
        self.last_rock_spawn_x = -MIN_SPAWN_DISTANCE  # Mantemos essa variável para compatibilidade

def draw_text(text, font, color, x, y):
    text_surface = font.render(text, True, color)
    DISPLAYSURF.blit(text_surface, (x, y))

def check_collision(rider_rect, animal_rect):
    # Use proper rectangle collision detection
    animal_collision = rider_rect.colliderect(animal_rect)
    return animal_collision

def discretize(value, buckets, min_val, max_val):
    """Discretiza o valor em buckets"""
    if value <= min_val:
        return 0
    if value >= max_val:
        return buckets - 1
    ratio = (value - min_val) / (max_val - min_val)
    return int(ratio * (buckets - 1))

def get_sonar_distance(game_state, horse_x, horse_y):
    # Retorna a menor distância detectada pelo sonar, ou -1 se nada detectado
    horse_center_x = horse_x + HORSE_RIDER_SCALE[0] // 2
    horse_center_y = horse_y + HORSE_RIDER_SCALE[1] // 2
    min_dist = float('inf')
    
    # Animal
    if game_state.is_animal_running:
        # Usar as dimensões corretas para calcular o centro baseado no tipo
        if game_state.current_animal_anim and animal_animations.index(game_state.current_animal_anim) == PIPE_INDEX:
            # Se for pipe, usar PIPE_SCALE
            animal_center = (game_state.animal_x + PIPE_SCALE[0] // 2, game_state.animal_y + PIPE_SCALE[1] // 2)
        else:
            # Se for animal normal, usar ANIMAL_SCALE
            animal_center = (game_state.animal_x + ANIMAL_SCALE[0] // 2, game_state.animal_y + ANIMAL_SCALE[1] // 2)
        
        dist_to_animal = math.hypot(animal_center[0] - horse_center_x, animal_center[1] - horse_center_y)
        if dist_to_animal < min_dist:
            min_dist = dist_to_animal
    
    if min_dist == float('inf'):
        return -1
    return min_dist

def update_environment(game_state):
    # Atualiza o ambiente de jogo (obstáculos, fundo, etc.)
    game_state.scroll_speed = min(game_state.scroll_speed + BASE_ACCELERATION * game_state.speed_multiplier, MAX_SCROLL_SPEED)
    current_time = time.time() * 1000
    
    if current_time - game_state.last_speed_increase > SPEED_INCREASE_INTERVAL:
        game_state.speed_multiplier *= 1.05
        game_state.last_speed_increase = current_time
    
    game_state.background_x -= int(game_state.scroll_speed * game_state.speed_multiplier)
    if game_state.background_x <= -background_image.get_width():
        game_state.background_x = 0
        
    current_scroll = abs(game_state.background_x)
    
    # Spawn de animais
    if not game_state.is_animal_running and (current_scroll - game_state.last_animal_spawn_x >= MIN_SPAWN_DISTANCE):
        game_state.is_animal_running = choices(ANIMAL_SHOW_FLAGS, ANIMAL_SHOW_WEIGHTS)[0] == 1
        if game_state.is_animal_running:
            game_state.animal_x = SCREEN_WIDTH  # Sempre começa fora da tela à direita
            game_state.current_animal_anim = choice(animal_animations)
            # Verifica se o animal escolhido é o tubo (índice 3)
            if animal_animations.index(game_state.current_animal_anim) == PIPE_INDEX:
                game_state.animal_y = GROUND_Y - 315  # Tubo alto - estende do chão para cima
            else:
                game_state.animal_y = GROUND_Y + 50  # Animais no chão
            game_state.last_animal_spawn_x = current_scroll
            
    # Não há spawn de pedras neste jogo
    game_state.is_rock_running = False
            
    # Movimento dos obstáculos
    game_state.animal_x -= int(game_state.scroll_speed * game_state.speed_multiplier)
    
    # Verifica se saiu da tela (resetando)
    if game_state.animal_x < -ANIMAL_SCALE[0]:
        game_state.is_animal_running = False

    # Garante que obstáculos sejam spawnados regularmente
    if not game_state.is_animal_running:
        spawn_chance = randint(0, 100)
        if spawn_chance < 10:  # 10% de chance de spawnar um novo obstáculo se não houver nenhum
            game_state.is_animal_running = True
            game_state.animal_x = SCREEN_WIDTH  # Sempre começa fora da tela à direita
            game_state.current_animal_anim = choice(animal_animations)
            # Verifica se o animal escolhido é o tubo (índice 3)
            if animal_animations.index(game_state.current_animal_anim) == PIPE_INDEX:
                game_state.animal_y = GROUND_Y - 315  # Tubo alto - estende do chão para cima
            else:
                game_state.animal_y = GROUND_Y + 50  # Animais no chão
            game_state.last_animal_spawn_x = current_scroll

def update_player(game_state, keys):
    # Atualiza o estado do jogador com base nas teclas pressionadas
    if not game_state.game_over:
        if (keys[K_UP] or keys[K_SPACE]) and game_state.horse_rider_y == GROUND_Y:
            game_state.rider_y_speed = 30
            game_state.is_ducking = False
            game_state.duck_timer = 0
        elif keys[K_DOWN] and game_state.horse_rider_y == GROUND_Y:
            if not game_state.is_ducking:
                game_state.is_ducking = True
                game_state.duck_timer = 1
        else:
            if game_state.is_ducking:
                game_state.duck_timer += 1
                if game_state.duck_timer > 30 and not keys[K_DOWN]:
                    game_state.is_ducking = False
                    game_state.duck_timer = 0
        
        # Aplicar física
        game_state.horse_rider_y -= game_state.rider_y_speed
        game_state.rider_y_speed -= GRAVITY
        if game_state.horse_rider_y > GROUND_Y:
            game_state.horse_rider_y = GROUND_Y
            game_state.rider_y_speed = 0
            
        # Aumentar score
        game_state.score += int(game_state.scroll_speed * game_state.speed_multiplier)

def update_ai(game_state, ai_model):
    # Atualiza a IA com base no modelo treinado
    if not game_state.game_over:
        # Construir vetor de estado para a IA (igual ao horse_racing_ai_only)
        horse_height = discretize(game_state.horse_rider_y, STATE_BUCKETS[0], 0, SCREEN_HEIGHT)
        dist_to_animal = discretize(game_state.animal_x - game_state.horse_rider_x, STATE_BUCKETS[1], 0, SCREEN_WIDTH) if game_state.is_animal_running else STATE_BUCKETS[1] - 1
        animal_active = 1 if game_state.is_animal_running else 0
        dist_to_rock = STATE_BUCKETS[3] - 1
        rock_active = 0
        sonar_dist = get_sonar_distance(game_state, game_state.horse_rider_x, game_state.horse_rider_y)
        sonar_norm = sonar_dist / SONAR_RANGE if sonar_dist >= 0 else -1
        is_pipe = 1 if game_state.is_animal_running and game_state.current_animal_anim and animal_animations.index(game_state.current_animal_anim) == PIPE_INDEX else 0
        close_pipe = 1 if is_pipe and 0 < (game_state.animal_x - game_state.horse_rider_x) < 200 else 0
        state_vector = [horse_height, dist_to_animal, animal_active, dist_to_rock, rock_active, sonar_norm, is_pipe, close_pipe]
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(DEVICE)
        action = ai_model.act(state_tensor)
        # Executar ação da IA (igual ao ai_only)
        if action == 1 and game_state.horse_rider_y == GROUND_Y:
            game_state.rider_y_speed = 30
            game_state.is_ducking = False
            game_state.duck_timer = 0
        elif action == 2 and game_state.horse_rider_y == GROUND_Y:
            if not game_state.is_ducking:
                game_state.is_ducking = True
                game_state.duck_timer = 1
        else:
            if game_state.is_ducking:
                game_state.duck_timer += 1
                if game_state.duck_timer > 30:
                    game_state.is_ducking = False
                    game_state.duck_timer = 0
        
        # Aplicar física
        game_state.horse_rider_y -= game_state.rider_y_speed
        game_state.rider_y_speed -= GRAVITY
        if game_state.horse_rider_y > GROUND_Y:
            game_state.horse_rider_y = GROUND_Y
            game_state.rider_y_speed = 0
            
        # Aumentar score
        game_state.score += int(game_state.scroll_speed * game_state.speed_multiplier)

def check_game_state_collision(game_state):
    # Verifica colisões e atualiza o estado do jogo
    y_offset = 0
    if game_state.side == "ai":
        y_offset = TOP_BOUNDARY
    if game_state.is_ducking:
        rider_rect = pygame.Rect(game_state.horse_rider_x, game_state.horse_rider_y + 40 + y_offset, 100, 60)  # mais baixo
    else:
        rider_rect = pygame.Rect(game_state.horse_rider_x, game_state.horse_rider_y + y_offset, 100, 100)
    # Determinar as dimensões corretas do animal baseado no tipo
    if game_state.is_animal_running and game_state.current_animal_anim:
        if animal_animations.index(game_state.current_animal_anim) == PIPE_INDEX:
            animal_rect = pygame.Rect(game_state.animal_x, game_state.animal_y + y_offset, PIPE_SCALE[0], PIPE_SCALE[1])
        else:
            animal_rect = pygame.Rect(game_state.animal_x, game_state.animal_y + y_offset, ANIMAL_SCALE[0], ANIMAL_SCALE[1])
    else:
        animal_rect = pygame.Rect(game_state.animal_x, game_state.animal_y + y_offset, ANIMAL_SCALE[0], ANIMAL_SCALE[1])
    collision = check_collision(rider_rect, animal_rect)
    if collision:
        game_state.game_over = True
        save_score(game_state.nickname, game_state.score)
        return True
    return False

def draw_environment(game_state):
    # Desenha o ambiente do jogo (fundo, obstáculos, etc.)
    # Se for o lado do jogador, desenha na metade superior
    # Se for o lado da IA, desenha na metade inferior
    y_offset = 0
    height_limit = SCREEN_HEIGHT // 2
    if game_state.side == "ai":
        y_offset = TOP_BOUNDARY
    
    # Draw background
    DISPLAYSURF.blit(background_image, (game_state.background_x, y_offset), (0, 0, background_image.get_width(), height_limit))
    DISPLAYSURF.blit(background_image, (game_state.background_x + background_image.get_width(), y_offset), (0, 0, background_image.get_width(), height_limit))
    
    # Draw obstacles
    if game_state.is_animal_running and game_state.current_animal_anim:
        DISPLAYSURF.blit(game_state.current_animal_anim.next(), (game_state.animal_x, game_state.animal_y + y_offset))

def draw_player(game_state):
    # Desenha o cavalo do jogador
    y_offset = 0
    if game_state.side == "ai":
        y_offset = TOP_BOUNDARY
    
    if game_state.is_ducking:
        DISPLAYSURF.blit(rider_crouch_anim.next(), (game_state.horse_rider_x, game_state.horse_rider_y + y_offset))
    else:
        DISPLAYSURF.blit(rider_run_anim.next(), (game_state.horse_rider_x, game_state.horse_rider_y + y_offset))
    
    # Se for game over, desenha uma cruz sobre o cavalo
    if game_state.game_over:
        pygame.draw.line(DISPLAYSURF, RED, (game_state.horse_rider_x, game_state.horse_rider_y + y_offset), 
                        (game_state.horse_rider_x + HORSE_RIDER_SCALE[0], game_state.horse_rider_y + HORSE_RIDER_SCALE[1] + y_offset), 4)
        pygame.draw.line(DISPLAYSURF, RED, (game_state.horse_rider_x + HORSE_RIDER_SCALE[0], game_state.horse_rider_y + y_offset), 
                        (game_state.horse_rider_x, game_state.horse_rider_y + HORSE_RIDER_SCALE[1] + y_offset), 4)

def draw_ui(player_state, ai_state):
    # Desenha a interface do usuário (pontuações, divisão de tela, etc.)
    # Divisória horizontal
    pygame.draw.line(DISPLAYSURF, WHITE, (0, TOP_BOUNDARY), (SCREEN_WIDTH, TOP_BOUNDARY), 4)
    
    # Títulos
    draw_text(player_state.nickname, medium_font, RED, 20, 20)
    draw_text("AI", medium_font, GREEN, 20, TOP_BOUNDARY + 20)
    
    # Pontuações
    draw_text(f"Score: {player_state.score}", small_font, WHITE, 20, 60)
    draw_text(f"Score: {ai_state.score}", small_font, WHITE, 20, TOP_BOUNDARY + 60)
    
    # Instruções
    draw_text("Setas ↑/↓ ou Espaço", small_font, WHITE, 20, TOP_BOUNDARY - 40)
    
    # Game over
    if player_state.game_over:
        draw_text("GAME OVER!", main_font, RED, 120, 140)
    if ai_state.game_over:
        draw_text("GAME OVER!", main_font, RED, 120, TOP_BOUNDARY + 140)
    
    # Resultado final
    if player_state.game_over and ai_state.game_over:
        # Tela preta de resultado final
        DISPLAYSURF.fill(BLACK)
        # Pega o top score e nick
        top_scores = get_top_scores(1)
        if top_scores:
            top_nick, top_score = top_scores[0]
        else:
            top_nick, top_score = "-", 0
        draw_text("RESULTADO FINAL", main_font, YELLOW, SCREEN_WIDTH // 2 - 150, TOP_BOUNDARY - 100)
        draw_text(f"Score Jogador: {player_state.nickname} - {player_state.score}", main_font, CYAN, SCREEN_WIDTH // 2 - 250, TOP_BOUNDARY - 40)
        draw_text(f"Score IA: {ai_state.score}", main_font, GREEN, SCREEN_WIDTH // 2 - 250, TOP_BOUNDARY + 20)
        draw_text(f"Recorde: {top_nick} - {top_score}", main_font, WHITE, SCREEN_WIDTH // 2 - 250, TOP_BOUNDARY + 80)
        if player_state.score > ai_state.score:
            draw_text("JOGADOR VENCEU!", main_font, CYAN, SCREEN_WIDTH // 2 - 150, TOP_BOUNDARY + 140)
        elif ai_state.score > player_state.score:
            draw_text("IA VENCEU!", main_font, GREEN, SCREEN_WIDTH // 2 - 120, TOP_BOUNDARY + 140)
        else:
            draw_text("EMPATE!", main_font, YELLOW, SCREEN_WIDTH // 2 - 80, TOP_BOUNDARY + 140)
        draw_text("Pressione ENTER para jogar novamente ou ESC para sair", small_font, WHITE, SCREEN_WIDTH // 2 - 250, TOP_BOUNDARY + 200)
        pygame.display.update()
        # Espera ENTER ou ESC
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        player_state.reset()
                        ai_state.reset()
                        player_state.nickname = player_state.nickname  # mantém nick
                        waiting = False
                        return
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        exit()
        return

def main():
    # Carregando o melhor modelo treinado
    best_model_path = None

    path = f"treinamento/model/best_model_18.pth"
    best_model_path = path
    
    if best_model_path is None:
        print("Nenhum modelo treinado encontrado!")
        return
    
    print(f"Usando modelo: {best_model_path}")
    
    # Carregar modelo
    state_size = len(STATE_BUCKETS) + 1  # +1 para o sonar
    ai_model = QNetwork(state_size, len(ACTIONS)).to(DEVICE)
    ai_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    ai_model.eval()  # Modo de avaliação
    
    # Criação dos estados do jogo
    player_state = GameState(side="player")
    ai_state = GameState(side="ai")
    
    running = True
    show_hitbox_flag = False
    
    # Iniciar o jogo
    nickname = "PLAYER"
    if ENABLE_NICKNAME:
        pygame.display.update()
        input_box = pygame.Rect(SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2 - 30, 400, 60)
        color_inactive = pygame.Color('lightskyblue3')
        color_active = pygame.Color('dodgerblue2')
        color = color_inactive
        active = True
        text = 'PLAYER'
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if input_box.collidepoint(event.pos):
                        active = True
                    else:
                        active = False
                    color = color_active if active else color_inactive
                if event.type == pygame.KEYDOWN:
                    if active:
                        if event.key == pygame.K_RETURN:
                            nickname = text
                            done = True
                        elif event.key == pygame.K_BACKSPACE:
                            text = text[:-1]
                        else:
                            text += event.unicode
            
            DISPLAYSURF.fill(BLACK)
            draw_text("HORSE RACING - PLAYER VS AI", main_font, YELLOW, SCREEN_WIDTH // 2 - 240, 80)
            draw_text("Digite seu nickname:", small_font, WHITE, SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2 - 70)
            
            # Render the current text
            txt_surface = pygame.font.SysFont('Arial', 32).render(text, True, WHITE)
            
            # Resize the box if the text is too long
            width = max(400, txt_surface.get_width() + 10)
            input_box.w = width
            
            # Blit the text
            DISPLAYSURF.blit(txt_surface, (input_box.x + 5, input_box.y + 10))
            
            # Blit the input_box rect
            pygame.draw.rect(DISPLAYSURF, color, input_box, 2)
            
            # Adicionar instrução
            draw_text("Pressione ENTER para começar", small_font, WHITE, 
                    SCREEN_WIDTH // 2 - 180, SCREEN_HEIGHT // 2 + 60)
            
            pygame.display.update()
            frame_per_sec.tick(FPS)
    
    player_state.nickname = nickname
    
    # Ao sair do menu, troca para música do jogo
    mixer.music.load(MEDIA_PREFIX + 'game.mp3')
    mixer.music.play(loops=-1)
    mixer.music.set_volume(0.2)
    
    # Loop principal do jogo
    while running:
        # Capturar entrada do jogador
        keys = pygame.key.get_pressed()
        
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
            if event.type == KEYDOWN:
                if event.key == K_h:
                    show_hitbox_flag = not show_hitbox_flag
        
        # Atualizar ambiente
        if not player_state.game_over:
            update_environment(player_state)
        if not ai_state.game_over:
            update_environment(ai_state)
        
        # Atualizar jogador
        update_player(player_state, keys)
        
        # Atualizar IA
        update_ai(ai_state, ai_model)
        
        # Verificar colisões
        check_game_state_collision(player_state)
        check_game_state_collision(ai_state)
        
        # Renderização
        DISPLAYSURF.fill(BLACK)
        
        # Desenhar ambiente
        draw_environment(player_state)
        draw_environment(ai_state)
        
        # Desenhar cavalos
        draw_player(player_state)
        draw_player(ai_state)
        
        # Desenhar UI
        draw_ui(player_state, ai_state)
        
        # Mostrar hitboxes se ativado
        if show_hitbox_flag:
            player_y_offset = 0
            ai_y_offset = TOP_BOUNDARY
            # Hitboxes para o jogador
            if player_state.is_ducking:
                player_hitbox = pygame.Rect(player_state.horse_rider_x, player_state.horse_rider_y + 40 + player_y_offset, 100, 60)
            else:
                player_hitbox = pygame.Rect(player_state.horse_rider_x, player_state.horse_rider_y + player_y_offset, 100, 100)
            pygame.draw.rect(DISPLAYSURF, CYAN, player_hitbox, 2)
            # Hitbox para animal do jogador
            if player_state.is_animal_running and player_state.current_animal_anim:
                if animal_animations.index(player_state.current_animal_anim) == PIPE_INDEX:
                    animal_hitbox = pygame.Rect(player_state.animal_x, player_state.animal_y + player_y_offset, PIPE_SCALE[0], PIPE_SCALE[1])
                else:
                    animal_hitbox = pygame.Rect(player_state.animal_x, player_state.animal_y + player_y_offset, ANIMAL_SCALE[0], ANIMAL_SCALE[1])
                pygame.draw.rect(DISPLAYSURF, YELLOW, animal_hitbox, 2)
            # Hitboxes para a IA
            if ai_state.is_ducking:
                ai_hitbox = pygame.Rect(ai_state.horse_rider_x, ai_state.horse_rider_y + 40 + ai_y_offset, 100, 60)
            else:
                ai_hitbox = pygame.Rect(ai_state.horse_rider_x, ai_state.horse_rider_y + ai_y_offset, 100, 100)
            pygame.draw.rect(DISPLAYSURF, GREEN, ai_hitbox, 2)
            # Hitbox para animal da IA
            if ai_state.is_animal_running and ai_state.current_animal_anim:
                if animal_animations.index(ai_state.current_animal_anim) == PIPE_INDEX:
                    ai_animal_hitbox = pygame.Rect(ai_state.animal_x, ai_state.animal_y + ai_y_offset, PIPE_SCALE[0], PIPE_SCALE[1])
                else:
                    ai_animal_hitbox = pygame.Rect(ai_state.animal_x, ai_state.animal_y + ai_y_offset, ANIMAL_SCALE[0], ANIMAL_SCALE[1])
                pygame.draw.rect(DISPLAYSURF, YELLOW, ai_animal_hitbox, 2)
        
        pygame.display.update()
        frame_per_sec.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()
