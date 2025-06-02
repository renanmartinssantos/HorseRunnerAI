import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# --- Parâmetros ---
STATE_SIZE = 3    # [distancia, colisao, score]
ACTION_SIZE = 2   # pular ou abaixar
HIDDEN_SIZE = 64
LR = 1e-3
GAMMA = 0.99
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Rede Neural ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.out = nn.Linear(HIDDEN_SIZE, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# --- Memória de Replay ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# --- Ambiente Simplificado (Interface RL) ---
class GameEnvRL:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Resetar seu estado do jogo aqui
        self.score = 0
        self.done = False
        # Posição inicial, etc
        self.horse_y = GROUND_Y
        self.rider_y_speed = 0
        self.abaixado = False
        # Simule um obstáculo a uma certa distância inicial (exemplo)
        self.obstacle_dist = 1.0  # 1.0 = distância normalizada, perto é 0.0
        self.collision = 0
        return self._get_state()
    
    def _get_state(self):
        # Retorna vetor de estado normalizado entre 0 e 1
        dist_norm = self.obstacle_dist  # já normalizado (ex: distância / max_dist)
        col = float(self.collision)
        score_norm = self.score / 10000  # exemplo para normalizar o score
        return np.array([dist_norm, col, score_norm], dtype=np.float32)
    
    def step(self, action):
        # action: 0 = pular (cima), 1 = abaixar (baixo)
        if self.done:
            return self._get_state(), 0, self.done
        
        # Atualizar estado do jogo simulando a física simplificada e obstáculos
        # Exemplo muito simples:
        if action == 0:  # pular
            self.horse_y += 20
        elif action == 1:  # abaixar
            self.abaixado = True
        
        # Atualiza distância do obstáculo (diminuir)
        self.obstacle_dist -= 0.05  # simula a aproximação
        if self.obstacle_dist < 0:
            self.obstacle_dist = 1.0  # novo obstáculo aparece
        
        # Detectar colisão simplificada
        # Exemplo: colisão se distância < 0.1 e não abaixado ou pulo
        if self.obstacle_dist < 0.1 and not self.abaixado and self.horse_y <= GROUND_Y + 10:
            self.collision = 1
            self.done = True
        
        # Recompensa
        reward = 0.001  # por distância percorrida
        if self.done:
            reward = -1
        
        self.score += 1
        state = self._get_state()
        return state, reward, self.done
