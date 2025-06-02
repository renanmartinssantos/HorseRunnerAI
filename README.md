# Horse Racing - Player vs AI

Um jogo de corrida de cavalos em Python, onde você compete contra uma IA treinada com Deep Q-Learning (PyTorch). O jogo é exibido em tela dividida, mostrando o progresso do jogador e da IA simultaneamente.

> **Este projeto foi baseado no repositório original [py_horse_racing do TekiChan](https://github.com/tekichan/py_horse_racing/).**

## Funcionalidades
- **Tela dividida:** Jogador e IA correm em pistas separadas, exibidas na mesma janela.
- **Obstáculos dinâmicos:** Animais e tubos aparecem aleatoriamente como obstáculos.
- **IA com PyTorch:** A IA utiliza uma rede neural treinada e executa em GPU se disponível.
- **Nickname e placar:** O jogador pode inserir seu nome e as pontuações são salvas.
- **Visualização de hitboxes:** Ative/desative pressionando `H`.
- **Música e animações:** Inclui trilha sonora e sprites animados.

## Como Jogar

1. **Requisitos**
   - Python 3.8+
   - Pygame
   - PyTorch
   - numpy

   Instale as dependências:
   ```sh
   pip install pygame torch numpy
   ```

2. **Recursos**
   - Certifique-se de que as pastas `images/` e `media/` estejam presentes com os sprites e músicas necessários.
   - O modelo treinado da IA deve estar em `treinamento/model/best_model_18.pth`.

3. **Executando**
   ```sh
   python horse_racing_player_vs_ai.py
   ```

4. **Controles**
   - **Cima ou Espaço:** Pular
   - **Baixo:** Abaixar
   - **H:** Mostrar/ocultar hitboxes
   - **ESC:** Sair do jogo

5. **Fluxo**
   - Digite seu nickname e pressione ENTER para começar.
   - O jogo termina quando ambos (jogador e IA) colidem com um obstáculo.
   - O resultado final e o recorde são exibidos ao final da partida.

## Estrutura do Código
- **QNetwork:** Rede neural da IA (PyTorch).
- **GameState:** Estado do jogador ou IA.
- **Funções principais:** Atualização do ambiente, detecção de colisão, renderização e lógica de controle.
- **main():** Loop principal do jogo, inicialização e gerenciamento de eventos.

## Observações
- O jogo utiliza GPU automaticamente se disponível (`torch.device('cuda')`).
- O placar é salvo em `score/scores.txt`.
- Para treinar ou atualizar o modelo da IA, utilize o script de treinamento correspondente.
