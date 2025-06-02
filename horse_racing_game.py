from random import choices, randint, choice
import time
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_s, K_h
from pygame import mixer
import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import collections
import pygame
import imageio
import collections
import imageio
import collections
import pygame
import imageio

# ==== CONFIGURAÇÃO ====
ENABLE_NICKNAME = False
SCORE_FILE = "score/scores.txt"

# Torch device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 480
MIDDLE_X = 280
GROUND_Y = 200
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GAME_TITLE = 'Horse Racing RL'
FOLDER_PREFIX = 'images/'
BACKGROUND_PATH = FOLDER_PREFIX + 'background.png'
HORSE_RIDER_SS_PATH = FOLDER_PREFIX + 'horse_rider_spritesheet.png'
DOG_SS_PATH = FOLDER_PREFIX + 'SleepDog.png'
PIG_SS_PATH = FOLDER_PREFIX + 'PigIdle.png'
WOLF_SS_PATH = FOLDER_PREFIX + 'TimberWolf.png'
TCHICK_SS_PATH = FOLDER_PREFIX + 'TinyChick.png'
PIPE_SS_PATH = FOLDER_PREFIX + 'Pipe.png'  # Renamed from BIRD_SS_PATH to PIPE_SS_PATH
ROCK_SS_PATH = FOLDER_PREFIX + 'rock.png'
FPS = 30
TICK_FRAMES = 4
GRAVITY = 3
HORSE_RIDER_SCALE = (130, 130)
ANIMAL_SCALE = (64, 64)
PIPE_SCALE = (75, 350)  # Pipe scale - wider and much taller to extend vertically
ROCK_SCALE = (50, 50)
MEDIA_PREFIX = 'media/'
MAX_SCROLL_SPEED = 12
BASE_ACCELERATION = 0.02
SPEED_MULTIPLIER = 1.0
SPEED_INCREASE_INTERVAL = 2000
ANIMAL_SHOW_FLAGS = [0, 1]
ANIMAL_SHOW_WEIGHTS = [0.40, 0.60]  # Aumentei a probabilidade de animais aparecerem
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
TRAINING_EPISODES = 50
NUM_AGENTS = 20  # Número de agentes para treinamento
RENDER_TRAINING = True  # Flag para renderização
REALTIME_TRAINING_VIS = True  # Flag para ativar/desativar visualização do treinamento em tempo real

REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64

# Definindo a rede neural
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 8)  # state_size agora é dinâmico
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def act(self, state_tensor):
        with torch.no_grad():
            q_values = self.forward(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        return action

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(tuple(args))
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

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
        with open(SCORE_FILE, "a") as f:
            f.write(f"{nick} {pontos}\n")
    except Exception as e:
        print(f"Erro ao salvar pontuação: {e}")

def load_count():
    try:
        with open("count.txt", "r") as f:
            return int(f.read().strip())
    except:
        return 0

def save_count(count):
    with open("count.txt", "w") as f:
        f.write(str(count))

# Inicialização do Pygame
pygame.init()
DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(GAME_TITLE)
frame_per_sec = pygame.time.Clock()

# Carregar assets
background_image = pygame.image.load(BACKGROUND_PATH).convert()
mixer.init()
mixer.music.load(MEDIA_PREFIX + 'dark-happy-world.ogg')
mixer.music.play(loops=-1)
mixer.music.set_volume(0.0)

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

class GameState:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.horse_rider_x = MIDDLE_X
        self.horse_rider_y = GROUND_Y
        self.rider_y_speed = 0
        # Adicionado para abaixar
        self.is_ducking = False
        self.duck_timer = 0  # tempo em frames abaixado
        self.scroll_speed = 10.0
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
        self.nickname = "RL_AGENT"
        self.is_animal_running = False
        self.is_rock_running = False  # Garantimos que as pedras não comecem ativas
        self.current_animal_anim = None
        self.last_animal_spawn_x = -MIN_SPAWN_DISTANCE  # Controla spawn animal separadamente
        self.last_rock_spawn_x = -MIN_SPAWN_DISTANCE  # Mantemos essa variável para compatibilidade

state = GameState()

def draw_text(text, font, color, x, y):
    text_surface = font.render(text, True, color)
    DISPLAYSURF.blit(text_surface, (x, y))

def check_collision(rider_rect, animal_rect, rock_rect, is_ducking):
    # Use proper rectangle collision detection instead of center distance
    # This will properly detect collisions with the entire pipe area
    animal_collision = rider_rect.colliderect(animal_rect)
    rock_collision = False  # Pedras não causam mais colisão
    
    # Se estiver abaixado, pode evitar certas colisões com rochas altas (simulação simples)
    # Supondo que abaixar evita rocha alta (rock_y == ROCK_HEIGHTS[1])
    # avoids_rock = is_ducking and rock_collision and rock_rect.top < GROUND_Y - 125  # rocha alta
    # if avoids_rock:
    #     rock_collision = False

    return animal_collision # or rock_collision (removido colisão com pedras)

def discretize(value, buckets, min_val, max_val):
    """Discretiza o valor em buckets"""
    if value <= min_val:
        return 0
    if value >= max_val:
        return buckets - 1
    ratio = (value - min_val) / (max_val - min_val)
    return int(ratio * (buckets - 1))

import random

class MultiAgent:
    def __init__(self, num_agents):
        state_size = 8  # Corrigido: agora o vetor de estado tem 8 entradas
        self.agents = [QNetwork(state_size, len(ACTIONS)).to(DEVICE) for _ in range(num_agents)]
        self.optimizers = [optim.Adam(agent.parameters(), lr=0.001) for agent in self.agents]
        self.replay_buffers = [ReplayBuffer(REPLAY_BUFFER_SIZE) for _ in range(num_agents)]

    def act(self, state, agent_index):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        return self.agents[agent_index].act(state_tensor)

    def store(self, agent_index, state, action, reward, next_state, done):
        self.replay_buffers[agent_index].push(state, action, reward, next_state, done)

    def learn(self, agent_index):
        buffer = self.replay_buffers[agent_index]
        if len(buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
        # Corrige: garante que todos os arrays tenham o mesmo tamanho (BATCH_SIZE)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        # Garante que todos os arrays tenham o mesmo tamanho
        min_len = min(states.shape[0], actions.shape[0], rewards.shape[0], next_states.shape[0], dones.shape[0])
        states = states[:min_len]
        actions = actions[:min_len]
        rewards = rewards[:min_len]
        next_states = next_states[:min_len]
        dones = dones[:min_len]
        if states.shape[1] != 8:
            states = states.reshape(-1, 8)
        if next_states.shape[1] != 8:
            next_states = next_states.reshape(-1, 8)
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        # Garante que todos os tensores tenham o mesmo tamanho na dimensão 0
        min_len_torch = min(states.shape[0], actions.shape[0], rewards.shape[0], next_states.shape[0], dones.shape[0])
        states = states[:min_len_torch]
        actions = actions[:min_len_torch]
        rewards = rewards[:min_len_torch]
        next_states = next_states[:min_len_torch]
        dones = dones[:min_len_torch]
        q_values = self.agents[agent_index](states)
        next_q_values = self.agents[agent_index](next_states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q = rewards + (1 - dones) * GAMMA * next_q_value
        loss = nn.MSELoss()(q_value, expected_q)
        self.optimizers[agent_index].zero_grad()
        loss.backward()
        self.optimizers[agent_index].step()

multi_agent = MultiAgent(NUM_AGENTS)

def calculate_reward(game_state, done, action):
    """Recompensa elaborada:
       - Penaliza colisão fortemente
       - Recompensa sobrevivência
       - Penaliza pular sem necessidade (gastar impulso)
       - Recompensa pular perto de obstáculo
       - Recompensa por esperar próximo do obstáculo antes de pular (ação idle com obstáculo perto)
       - Penaliza abaixar desnecessariamente, recompensa abaixar correto para rocha baixa
    """
    horse_x = game_state.horse_rider_x
    if done:
        # Penalidade maior se morreu pelo pipe
        morreu_pelo_pipe = False
        if game_state.is_animal_running and game_state.current_animal_anim:
            is_pipe = animal_animations.index(game_state.current_animal_anim) == PIPE_INDEX
            # Considera morte pelo pipe se obstáculo estava próximo
            dist_animal = game_state.animal_x - horse_x
            if is_pipe and dist_animal is not None and 0 < dist_animal < 100:
                morreu_pelo_pipe = True
        if morreu_pelo_pipe:
            return -200  # penalidade maior
        return -100

    reward = 0
    horse_x = game_state.horse_rider_x

    dist_animal = game_state.animal_x - horse_x if game_state.is_animal_running else None
    dist_rock = None  # Pedras foram removidas, então sempre None

    close_animal = dist_animal is not None and 0 < dist_animal < 200
    close_rock = False  # Sempre falso porque não há mais pedras

    # Sobreviver ao frame
    reward += 1

    if action == 1:  # pular
        # Só ganha 40 se pular animal (não pipe)
        is_pipe = False
        if game_state.is_animal_running and game_state.current_animal_anim:
            is_pipe = animal_animations.index(game_state.current_animal_anim) == PIPE_INDEX
        if close_animal and not is_pipe:
            reward += 40  # pular animal (recompensa ainda mais forte)
        elif close_animal and is_pipe:
            reward -= 10  # pular pipe é ruim
        else:
            reward -= 10  # pular sem necessidade
    elif action == 0:  # idle
        if close_animal:
            reward += 2
        else:
            reward += 0
    elif action == 2:  # abaixar
        is_pipe = False
        if game_state.is_animal_running and game_state.current_animal_anim:
            is_pipe = animal_animations.index(game_state.current_animal_anim) == PIPE_INDEX
        if close_animal and is_pipe:
            reward += 10  # abaixar perto do pipe (recompensa reforçada)
        else:
            reward -= 1

    return reward

def sonar_detect(game_state):
    horse_center_x = game_state.horse_rider_x + HORSE_RIDER_SCALE[0]//2
    horse_center_y = game_state.horse_rider_y + HORSE_RIDER_SCALE[1]//2

    detected_objects = []

    if game_state.is_animal_running:
        animal_center = (game_state.animal_x + ANIMAL_SCALE[0]//2, game_state.animal_y + ANIMAL_SCALE[1]//2)
        dist_to_animal = math.hypot(animal_center[0] - horse_center_x, animal_center[1] - horse_center_y)
        if dist_to_animal <= SONAR_RANGE:
            detected_objects.append(('animal', dist_to_animal, animal_center))

    # Pedras foram removidas do jogo
    # if game_state.is_rock_running:
    #     rock_center = (game_state.rock_x + ROCK_SCALE[0]//2, game_state.rock_y + ROCK_SCALE[1]//2)
    #     dist_to_rock = math.hypot(rock_center[0] - horse_center_x, rock_center[1] - horse_center_y)
    #     if dist_to_rock <= SONAR_RANGE:
    #         detected_objects.append(('rock', dist_to_rock, rock_center))

    return detected_objects

def get_sonar_distance(game_state, horse_x, horse_y):
    # Retorna a menor distância detectada pelo sonar, ou -1 se nada detectado
    horse_center_x = horse_x + HORSE_RIDER_SCALE[0] // 2
    horse_center_y = horse_y + HORSE_RIDER_SCALE[1] // 2
    min_dist = float('inf')    # Animal
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
    # Pedras foram removidas, então não há detecção para elas
    # if game_state.is_rock_running:
    #     rock_center = (game_state.rock_x + ROCK_SCALE[0] // 2, game_state.rock_y + ROCK_SCALE[1] // 2)
    #     dist_to_rock = math.hypot(rock_center[0] - horse_center_x, rock_center[1] - horse_center_y)
    #     if dist_to_rock < min_dist:
    #         min_dist = dist_to_rock
    if min_dist == float('inf'):
        return -1
    return min_dist

# Colors for each agent
AGENT_COLORS = [
    (0, 200, 255),  # cyan
    (255, 100, 100),  # red
    (100, 255, 100),  # green
    (255, 255, 100),  # yellow
    (255, 100, 255),  # magenta
]

class AgentHorseState:
    def __init__(self):
        self.x = MIDDLE_X
        self.y = GROUND_Y
        self.y_speed = 0
        self.is_ducking = False
        self.duck_timer = 0
        self.game_over = False
        self.score = 0
        self.nickname = "RL_AGENT"

    def reset(self):
        self.x = MIDDLE_X
        self.y = GROUND_Y
        self.y_speed = 0
        self.is_ducking = False
        self.duck_timer = 0
        self.game_over = False
        self.score = 0

def main():
    count = load_count()
    print(f"COUNT ATUAL: {count}")
    # # Carrega pesos individuais de cada agente, se existirem
    # for agent_index in range(NUM_AGENTS):
    #     agent_path = f"treinamento/agent_{agent_index}_model.pth"
    #     if os.path.exists(agent_path):
    #         multi_agent.agents[agent_index].load_state_dict(torch.load(agent_path, map_location=DEVICE))
    #         print(f"Pesos carregados para agente {agent_index} de {agent_path}")
    
    # Se count > 0, carrega pesos anteriores para os agentes
    if count > 0:
        distilled_path = f"treinamento/model/best_model_{count-1}.pth"
        if os.path.exists(distilled_path):
            for agent_index in range(NUM_AGENTS):
                multi_agent.agents[agent_index].load_state_dict(torch.load(distilled_path, map_location=DEVICE))
            print(f"Pesos do modelo destilado {distilled_path} carregados para todos os agentes.")
        else:
            print(f"Modelo destilado {distilled_path} não encontrado. Treinamento começará do zero.")
    
    running = True    
    episode = 0
    max_episodes = TRAINING_EPISODES
    show_sonar_flag = False
    show_agent_colors = True
    show_hitbox_flag = False
    
    # Shared environment state
    env_state = GameState()
    # Each agent has its own horse state
    agent_horses = [AgentHorseState() for _ in range(NUM_AGENTS)]
    agent_gameover = [False for _ in range(NUM_AGENTS)]
    agent_scores = [0 for _ in range(NUM_AGENTS)]
    agent_score_history = [[] for _ in range(NUM_AGENTS)]  # Histórico de scores por agente

    # Configuração para salvar frames diretamente no disco
    temp_frames_dir = None
    video_writer = None
    frames_count = 0
    
    if RENDER_TRAINING:
        display_surf = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Criar diretório temporário para salvar frames
        temp_frames_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "treinamento", "temp_frames")
        if os.path.exists(temp_frames_dir):
            # Limpar frames anteriores se existirem
            for f in os.listdir(temp_frames_dir):
                os.remove(os.path.join(temp_frames_dir, f))
        else:
                os.makedirs(temp_frames_dir, exist_ok=True)
    else:
        display_surf = DISPLAYSURF

    while running and episode < max_episodes:
        # Reset environment and agents
        env_state.reset()
        for agent in agent_horses:
            agent.reset()
        agent_gameover = [False for _ in range(NUM_AGENTS)]
        agent_scores = [0 for _ in range(NUM_AGENTS)]
        steps = 0
        while not all(agent_gameover):
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    return
                if event.type == KEYDOWN:
                    if event.key == K_s:
                        show_sonar_flag = not show_sonar_flag
                    if event.key == pygame.K_x:
                        show_agent_colors = not show_agent_colors
                    if event.key == K_h:
                        show_hitbox_flag = not show_hitbox_flag
            # Update environment (obstacles, background, etc.)
            if not all(agent_gameover):
                env_state.scroll_speed = min(env_state.scroll_speed + BASE_ACCELERATION * env_state.speed_multiplier, MAX_SCROLL_SPEED)
                current_time = time.time() * 1000
                if current_time - env_state.last_speed_increase > SPEED_INCREASE_INTERVAL:
                    env_state.speed_multiplier *= 1.05
                    env_state.last_speed_increase = current_time
                
                env_state.background_x -= int(env_state.scroll_speed * env_state.speed_multiplier)
                if env_state.background_x <= -background_image.get_width():
                    env_state.background_x = 0
                current_scroll = abs(env_state.background_x)
                if not env_state.is_animal_running and (current_scroll - env_state.last_animal_spawn_x >= MIN_SPAWN_DISTANCE):
                    env_state.is_animal_running = choices(ANIMAL_SHOW_FLAGS, ANIMAL_SHOW_WEIGHTS)[0] == 1
                    if env_state.is_animal_running:
                        env_state.animal_x = SCREEN_WIDTH
                        env_state.current_animal_anim = choice(animal_animations)
                        # Verifica se o animal escolhido é o tubo (índice 3)
                        if animal_animations.index(env_state.current_animal_anim) == PIPE_INDEX:
                            env_state.animal_y = GROUND_Y - 315  # Tubo alto - estende do chão para cima
                        else:
                            env_state.animal_y = GROUND_Y + 50  # Animais no chão
                        env_state.last_animal_spawn_x = current_scroll
                # Desabilitamos o spawn de pedras
                if not env_state.is_rock_running and (current_scroll - env_state.last_rock_spawn_x >= MIN_SPAWN_DISTANCE):
                    env_state.is_rock_running = False  # Garantir que sempre seja False
                    env_state.last_rock_spawn_x = current_scroll
                
                env_state.animal_x -= int(env_state.scroll_speed * env_state.speed_multiplier)
                env_state.rock_x -= int(env_state.scroll_speed * env_state.speed_multiplier)
                if env_state.animal_x < -ANIMAL_SCALE[0]:
                    env_state.is_animal_running = False
                if env_state.rock_x < -ROCK_SCALE[0]:
                    env_state.is_rock_running = False

            # Each agent acts in the shared environment
            for agent_index, agent in enumerate(agent_horses):
                if agent_gameover[agent_index]:
                    continue
                # Build a state vector for this agent (horse position, env obstacles, sonar)
                horse_height = discretize(agent.y, STATE_BUCKETS[0], 0, SCREEN_HEIGHT)
                dist_to_animal = discretize(env_state.animal_x - agent.x, STATE_BUCKETS[1], 0, SCREEN_WIDTH) if env_state.is_animal_running else STATE_BUCKETS[1] - 1
                animal_active = 1 if env_state.is_animal_running else 0
                dist_to_rock = discretize(env_state.rock_x - agent.x, STATE_BUCKETS[3], 0, SCREEN_WIDTH) if env_state.is_rock_running else STATE_BUCKETS[3] - 1
                rock_active = 1 if env_state.is_rock_running else 0
                sonar_dist = get_sonar_distance(env_state, agent.x, agent.y)
                sonar_norm = sonar_dist / SONAR_RANGE if sonar_dist >= 0 else -1
                # --- FEATURE ENGENEERING: Adiciona flag se o obstáculo é o pipe e está próximo ---
                is_pipe = 0
                close_pipe = 0
                if env_state.is_animal_running and env_state.current_animal_anim:
                    is_pipe = int(animal_animations.index(env_state.current_animal_anim) == PIPE_INDEX)
                    dist_pipe = env_state.animal_x - agent.x
                    close_pipe = int(is_pipe and 0 < dist_pipe < 200)
                # Inclui as flags no vetor de estado
                state_vector = [horse_height, dist_to_animal, animal_active, dist_to_rock, rock_active, sonar_norm, is_pipe, close_pipe]
                action = multi_agent.act(state_vector, agent_index)

                # Agent action logic
                if not agent.game_over:
                    if action == 1 and agent.y == GROUND_Y:
                        agent.y_speed = 30
                        agent.is_ducking = False
                        agent.duck_timer = 0
                    elif action == 2 and agent.y == GROUND_Y:
                        if not agent.is_ducking:
                            agent.is_ducking = True
                            agent.duck_timer = 1
                    else:
                        if agent.is_ducking:
                            agent.duck_timer += 1
                            if agent.duck_timer > 30:
                                agent.is_ducking = False
                                agent.duck_timer = 0
                    agent.y -= agent.y_speed
                    agent.y_speed -= GRAVITY
                    if agent.y > GROUND_Y:
                        agent.y = GROUND_Y
                        agent.y_speed = 0
                    if action == 0:
                        agent.score += int(env_state.scroll_speed * env_state.speed_multiplier)
                
                # Collision detection for this agent
                if agent.is_ducking:
                    rider_rect = pygame.Rect(agent.x, agent.y + 40, 100, 60)  # mais baixo
                else:
                    rider_rect = pygame.Rect(agent.x, agent.y, 100, 100)
                
                # Determinar as dimensões corretas do animal baseado no tipo
                if env_state.is_animal_running and env_state.current_animal_anim:
                    if animal_animations.index(env_state.current_animal_anim) == PIPE_INDEX:
                        # Se for pipe, usar PIPE_SCALE
                        animal_rect = pygame.Rect(env_state.animal_x, env_state.animal_y, PIPE_SCALE[0], PIPE_SCALE[1])
                    else:
                        # Se for animal normal, usar ANIMAL_SCALE
                        animal_rect = pygame.Rect(env_state.animal_x, env_state.animal_y, ANIMAL_SCALE[0], ANIMAL_SCALE[1])
                else:
                    animal_rect = pygame.Rect(env_state.animal_x, env_state.animal_y, ANIMAL_SCALE[0], ANIMAL_SCALE[1])
                
                rock_rect = pygame.Rect(env_state.rock_x, env_state.rock_y, ROCK_SCALE[0], ROCK_SCALE[1])
                collision = check_collision(rider_rect, animal_rect, rock_rect, agent.is_ducking)
                if collision:
                    agent.game_over = True
                    agent_gameover[agent_index] = True
                    save_score(agent.nickname + f'_A{agent_index}', agent.score)

                # Next state and reward
                next_horse_height = discretize(agent.y, STATE_BUCKETS[0], 0, SCREEN_HEIGHT)
                next_dist_to_animal = discretize(env_state.animal_x - agent.x, STATE_BUCKETS[1], 0, SCREEN_WIDTH) if env_state.is_animal_running else STATE_BUCKETS[1] - 1
                next_animal_active = 1 if env_state.is_animal_running else 0
                next_dist_to_rock = discretize(env_state.rock_x - agent.x, STATE_BUCKETS[3], 0, SCREEN_WIDTH) if env_state.is_rock_running else STATE_BUCKETS[3] - 1
                next_rock_active = 1 if env_state.is_rock_running else 0
                next_sonar_dist = get_sonar_distance(env_state, agent.x, agent.y)
                next_sonar_norm = next_sonar_dist / SONAR_RANGE if next_sonar_dist >= 0 else -1
                next_state_vector = [next_horse_height, next_dist_to_animal, next_animal_active, next_dist_to_rock, next_rock_active, next_sonar_norm]
                reward = calculate_reward(env_state, agent.game_over, action)
                done = float(agent.game_over)
                multi_agent.store(agent_index, state_vector, action, reward, next_state_vector, done)
                multi_agent.learn(agent_index)
                agent_scores[agent_index] = agent.score
                if REALTIME_TRAINING_VIS:
                    if len(agent_score_history[agent_index]) == 0 or agent_score_history[agent_index][-1] != agent.score:
                        agent_score_history[agent_index].append(agent.score)

            steps += 1

            # --- Garantir spawn de obstáculos mesmo com múltiplos agentes ---
            env_state.scroll_speed = min(env_state.scroll_speed + BASE_ACCELERATION * env_state.speed_multiplier, MAX_SCROLL_SPEED)
            current_time = time.time() * 1000
            if current_time - env_state.last_speed_increase > SPEED_INCREASE_INTERVAL:
                env_state.speed_multiplier *= 1.05
                env_state.last_speed_increase = current_time
            env_state.background_x -= int(env_state.scroll_speed * env_state.speed_multiplier)
            if env_state.background_x <= -background_image.get_width():
                env_state.background_x = 0
            current_scroll = abs(env_state.background_x)

            # --- Lógica de spawn robusta ---
            # Se nenhum obstáculo está ativo, força o spawn de pelo menos um
            if not env_state.is_animal_running and not env_state.is_rock_running:
                # Apenas animais serão spawnados, removendo completamente as pedras
                env_state.is_animal_running = True
                env_state.animal_x = SCREEN_WIDTH
                env_state.current_animal_anim = choice(animal_animations)                # Verifica se o animal escolhido é o tubo (índice 3)
                if animal_animations.index(env_state.current_animal_anim) == PIPE_INDEX:
                    env_state.animal_y = GROUND_Y - 315  # Tubo no chão
                else:
                    env_state.animal_y = GROUND_Y + 50  # Animais no chão
                env_state.last_animal_spawn_x = current_scroll
            else:
                if not env_state.is_animal_running and (current_scroll - env_state.last_animal_spawn_x >= MIN_SPAWN_DISTANCE):
                    env_state.is_animal_running = choices(ANIMAL_SHOW_FLAGS, ANIMAL_SHOW_WEIGHTS)[0] == 1
                    if env_state.is_animal_running:
                        env_state.animal_x = SCREEN_WIDTH
                        env_state.current_animal_anim = choice(animal_animations)
                        # Se for o tubo (pipe), posicionar mais alto
                        if animal_animations.index(env_state.current_animal_anim) == PIPE_INDEX:
                            env_state.animal_y = GROUND_Y - 315  # Tubo no chão
                        else:
                            env_state.animal_y = GROUND_Y + 50  # Animais no chão
                        env_state.last_animal_spawn_x = current_scroll
                if not env_state.is_rock_running and (current_scroll - env_state.last_rock_spawn_x >= MIN_SPAWN_DISTANCE):
                    env_state.is_rock_running = False  # Garantir que sempre seja False
                    env_state.last_rock_spawn_x = current_scroll
            env_state.animal_x -= int(env_state.scroll_speed * env_state.speed_multiplier)
            env_state.rock_x -= int(env_state.scroll_speed * env_state.speed_multiplier)
            if env_state.animal_x < -ANIMAL_SCALE[0]:
                env_state.is_animal_running = False
            if env_state.rock_x < -ROCK_SCALE[0]:
                env_state.is_rock_running = False

            # Visualization
            if RENDER_TRAINING:
                display_surf.fill(BLACK)
                # Draw background
                display_surf.blit(background_image, (env_state.background_x, 0))
                display_surf.blit(background_image, (env_state.background_x + background_image.get_width(), 0))
                # Draw obstacles - apenas os animais serão desenhados
                if env_state.is_animal_running and env_state.current_animal_anim:
                    display_surf.blit(env_state.current_animal_anim.next(), (env_state.animal_x, env_state.animal_y))
                # Removendo renderização das pedras
                # if env_state.is_rock_running:
                #     display_surf.blit(rock_anim.next(), (env_state.rock_x, env_state.rock_y))
                # Draw all agents
                for agent_index, agent in enumerate(agent_horses):
                    if agent.game_over:
                        continue  # Não desenha nem processa agente morto
                    color = AGENT_COLORS[agent_index % len(AGENT_COLORS)] if show_agent_colors else (0, 0, 0, 0)
                    # Draw horse sprite (abaixado ou normal)
                    if agent.is_ducking:
                        display_surf.blit(rider_crouch_anim.next(), (agent.x, agent.y))
                    else:
                        display_surf.blit(rider_run_anim.next(), (agent.x, agent.y))
                    # Overlay colorido semi-transparente
                    if show_agent_colors:
                        overlay = pygame.Surface((HORSE_RIDER_SCALE[0], HORSE_RIDER_SCALE[1]), pygame.SRCALPHA)
                        overlay.fill((*color, 80))
                        display_surf.blit(overlay, (agent.x, agent.y))
                    # Cruz de game over
                    if agent.game_over:
                        pygame.draw.line(display_surf, RED, (agent.x, agent.y), (agent.x + HORSE_RIDER_SCALE[0], agent.y + HORSE_RIDER_SCALE[1]), 4)
                        pygame.draw.line(display_surf, RED, (agent.x + HORSE_RIDER_SCALE[0], agent.y), (agent.x, agent.y + HORSE_RIDER_SCALE[1]), 4)
                    draw_text(f'A{agent_index}: {agent.score}', small_font, color, agent.x, agent.y - 30)
                    # Sonar/Colisão
                    if show_sonar_flag:
                        horse_center = (agent.x + HORSE_RIDER_SCALE[0] // 2, agent.y + HORSE_RIDER_SCALE[1] // 2)
                        pygame.draw.circle(display_surf, GREEN, horse_center, SONAR_RANGE, 2)                        # Detecção de obstáculos
                        detected = []
                        if env_state.is_animal_running:
                            # Determinar as dimensões corretas para o sonar baseado no tipo de animal/pipe
                            if env_state.current_animal_anim and animal_animations.index(env_state.current_animal_anim) == PIPE_INDEX:
                                # Se for pipe, usar PIPE_SCALE
                                animal_center = (env_state.animal_x + PIPE_SCALE[0] // 2, env_state.animal_y + PIPE_SCALE[1] // 2)
                            else:
                                # Se for animal normal, usar ANIMAL_SCALE
                                animal_center = (env_state.animal_x + ANIMAL_SCALE[0] // 2, env_state.animal_y + ANIMAL_SCALE[1] // 2)
                            
                            dist_to_animal = math.hypot(animal_center[0] - horse_center[0], animal_center[1] - horse_center[1])
                            if dist_to_animal <= SONAR_RANGE:
                                pygame.draw.circle(display_surf, YELLOW, animal_center, 30, 2)
                                pygame.draw.line(display_surf, YELLOW, horse_center, animal_center)
                        # Removendo renderização de sonar para pedras
                        # if env_state.is_rock_running:
                        #     rock_center = (env_state.rock_x + ROCK_SCALE[0] // 2, env_state.rock_y + ROCK_SCALE[1] // 2)
                        #     dist_to_rock = math.hypot(rock_center[0] - horse_center[0], rock_center[1] - horse_center[1])
                        #     if dist_to_rock <= SONAR_RANGE:
                        #         pygame.draw.circle(display_surf, RED, rock_center, 30, 2)
                        #         pygame.draw.line(display_surf, RED, horse_center, rock_center)                        # Retângulo de colisão
                        rider_rect = pygame.Rect(agent.x, agent.y, 100, 100)
                        if agent.game_over:
                            pygame.draw.rect(display_surf, RED, rider_rect, 4)
                        if env_state.is_animal_running:
                            # Usar as dimensões corretas para o retângulo de colisão
                            if env_state.current_animal_anim and animal_animations.index(env_state.current_animal_anim) == PIPE_INDEX:
                                # Se for pipe, usar PIPE_SCALE
                                animal_rect = pygame.Rect(env_state.animal_x, env_state.animal_y, PIPE_SCALE[0], PIPE_SCALE[1])
                            else:
                                # Se for animal normal, usar ANIMAL_SCALE
                                animal_rect = pygame.Rect(env_state.animal_x, env_state.animal_y, ANIMAL_SCALE[0], ANIMAL_SCALE[1])
                            pygame.draw.rect(display_surf, YELLOW, animal_rect, 2)
                            pygame.draw.rect(display_surf, YELLOW, animal_rect, 2)
                        # Removendo renderização de retângulo de colisão para pedras
                        # if env_state.is_rock_running:
                        #     rock_rect = pygame.Rect(env_state.rock_x, env_state.rock_y, ROCK_SCALE[0], ROCK_SCALE[1])
                        #     pygame.draw.rect(display_surf, RED, rock_rect, 2)
                    # Hitbox visualization
                    if show_hitbox_flag:
                        # Rider hitbox
                        if agent.is_ducking:
                            rider_hitbox = pygame.Rect(agent.x, agent.y + 40, 100, 60)
                        else:
                            rider_hitbox = pygame.Rect(agent.x, agent.y, 100, 100)
                        pygame.draw.rect(display_surf, (0, 255, 255), rider_hitbox, 3)
                        # Animal/Pipe hitbox
                        if env_state.is_animal_running and env_state.current_animal_anim:
                            if animal_animations.index(env_state.current_animal_anim) == PIPE_INDEX:
                                pipe_hitbox = pygame.Rect(env_state.animal_x, env_state.animal_y, PIPE_SCALE[0], PIPE_SCALE[1])
                                pygame.draw.rect(display_surf, (0, 255, 0), pipe_hitbox, 3)
                            else:
                                animal_hitbox = pygame.Rect(env_state.animal_x, env_state.animal_y, ANIMAL_SCALE[0], ANIMAL_SCALE[1])
                                pygame.draw.rect(display_surf, (255, 0, 255), animal_hitbox, 3)
                    # Draw info
                draw_text(f'Episódio: {episode + 1}/{max_episodes}  |  Treinamento: {count}', small_font, WHITE, 20, 20)
                draw_text(f'Passos: {steps}', small_font, WHITE, 20, 60)
                draw_text('Pressione S para alternar Sonar/Colisão', small_font, WHITE, 20, SCREEN_HEIGHT - 40)
                draw_text('Pressione X para alternar cores dos agentes', small_font, WHITE, 20, SCREEN_HEIGHT - 80)
                draw_text('Pressione H para alternar Hitboxes', small_font, WHITE, 20, SCREEN_HEIGHT - 120)
                # Real-Time Training Visualization
                if REALTIME_TRAINING_VIS:
                    # Parâmetros do gráfico
                    graph_x = SCREEN_WIDTH - 320
                    graph_y = 40
                    graph_w = 300
                    graph_h = 200
                    pygame.draw.rect(display_surf, (30,30,30), (graph_x-5, graph_y-5, graph_w+10, graph_h+10), border_radius=8)
                    pygame.draw.rect(display_surf, (60,60,60), (graph_x, graph_y, graph_w, graph_h), border_radius=6)
                    max_score = max([max(hist) if hist else 1 for hist in agent_score_history])
                    max_score = max(max_score, 10)
                    for idx, hist in enumerate(agent_score_history):
                        if len(hist) < 2:
                            continue
                        color = AGENT_COLORS[idx % len(AGENT_COLORS)]
                        points = []
                        for i, s in enumerate(hist[-graph_w:]):
                            px = graph_x + i
                            py = (graph_y + graph_h - int((s / max_score) * (graph_h-10)))/1.25
                            points.append((px, py))
                        if len(points) > 1:
                            pygame.draw.lines(display_surf, color, False, points, 2)
                    draw_text('Score (real-time)', small_font, WHITE, graph_x, graph_y-40)
                    # --- Detecta estagnação ---
                    STUCK_WINDOW = 100  # passos para considerar janela
                    STUCK_TOL = 5       # tolerância de variação
                    stuck = False
                    if all(len(hist) > STUCK_WINDOW for hist in agent_score_history):
                        stuck = True
                        for hist in agent_score_history:
                            window = hist[-STUCK_WINDOW:]
                            if max(window) - min(window) > STUCK_TOL:
                                stuck = False
                                break
                    if stuck:
                        pygame.draw.rect(display_surf, (255, 80, 80), (graph_x, graph_y+graph_h//2-20, graph_w, 40), border_radius=8)
                        draw_text('REDE ESTAGNADA!', small_font, (0,0,0), graph_x+20, graph_y+graph_h//2-10)
                        print(f"Rede Estagnada! {episode + 1} - Training {count}")                # Record frame - salvando direto para arquivo temporário em vez de memória
                if RENDER_TRAINING:
                    frame = pygame.surfarray.array3d(display_surf)
                    frame = np.transpose(frame, (1, 0, 2))
                    # Salvar frame como imagem PNG
                    frame_path = os.path.join(temp_frames_dir, f"frame_{frames_count:06d}.png")
                    imageio.imwrite(frame_path, frame)
                    frames_count += 1
                pygame.display.update()
                frame_per_sec.tick(FPS)
            else:
                pygame.display.update()
                frame_per_sec.tick(FPS)

        print(f"Episódio {episode + 1} finalizado com pontuação: {agent_scores}")
        episode += 1
        time.sleep(1)    # Gravar vídeo ao final do treinamento
    if RENDER_TRAINING:
        video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "treinamento", "videos", f"training_video_{count}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        try:
            print(f"Compilando vídeo a partir de {frames_count} frames...")
            
            # Criar vídeo a partir dos frames salvos no disco
            with imageio.get_writer(video_path, fps=FPS) as video:
                for i in range(frames_count):
                    frame_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.png")
                    if os.path.exists(frame_path):
                        frame = imageio.imread(frame_path)
                        video.append_data(frame)
                        
                        # Opcional: remover o frame depois de adicioná-lo ao vídeo para economizar espaço
                        os.remove(frame_path)
            
            print(f"Vídeo de treinamento salvo em {video_path}")
            
            # Limpar diretório temporário de frames
            if os.path.exists(temp_frames_dir):
                try:
                    for f in os.listdir(temp_frames_dir):
                        os.remove(os.path.join(temp_frames_dir, f))
                    os.rmdir(temp_frames_dir)
                except Exception as e:
                    print(f"Aviso: Não foi possível remover todos os arquivos temporários: {e}")
                    
        except Exception as e:
            print(f"Erro ao criar vídeo: {e}")

    # # Salvar o modelo treinado individual de cada agente
    # for agent_index in range(NUM_AGENTS):
    #     torch.save(multi_agent.agents[agent_index].state_dict(), f"treinamento/agent_{agent_index}_model.pth")

    # Seleção do Melhor Agente
    best_agent = int(np.argmax(agent_scores))
    best_model_path = f"treinamento/model/best_model_{count}.pth"
    torch.save(multi_agent.agents[best_agent].state_dict(), best_model_path)
    print(f"Melhor agente: {best_agent} salvo como {best_model_path} (score: {agent_scores[best_agent]})")

    # Ensemble por Votação
    def ensemble_act(state):
        votes = [agent.act(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)) for agent in multi_agent.agents]
        return max(set(votes), key=votes.count)
    print("Função ensemble_act(state) disponível para inferência por votação.")

    # Incrementa o count.txt
    save_count(count + 1)

    print("Treinamento finalizado e modelos salvos.")
    

if __name__ == "__main__":
    for i in range(30):
        main()
        time.sleep(150)
    pygame.quit()


