import numpy as np
import random

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)  
        self.current_player = 1

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        return self.get_state(), self.current_player

    def get_state(self):
      
        return tuple(self.board)

    def available_actions(self):
        return [i for i, val in enumerate(self.board) if val == 0]

    def step(self, action, player):
        if self.board[action] != 0:
            return self.get_state(), -10, True, False 
        
        self.board[action] = player

        # التحقق من الفوز أو التعادل
        winner = self.check_winner()
        if winner is not None:
            if winner == 1:
                return self.get_state(), 1, True, False  
            elif winner == -1:
                return self.get_state(), -1, True, False
            else:
                return self.get_state(), 0.5, True, False 

        return self.get_state(), 0, False, False

    def check_winner(self):
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], 
            [0, 3, 6], [1, 4, 7], [2, 5, 8], 
            [0, 4, 8], [2, 4, 6]          
        ]
        for combo in winning_combos:
            values = self.board[combo]
            if np.all(values == 1):
                return 1
            if np.all(values == -1):
                return -1
        if len(self.available_actions()) == 0:
            return 0 
        return None

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
        self.q_table = {}
        self.alpha = alpha      
        self.gamma = gamma       
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        
        q_values = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_values)
        
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_available_actions):
        current_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a) for a in next_available_actions], default=0.0)
        
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


env = TicTacToeEnv()
agent1 = QLearningAgent()
agent2 = QLearningAgent() 

episodes = 50000
for episode in range(episodes):
    state, player = env.reset()
    done = False

    while not done:
        actions = env.available_actions()
        if not actions:
            break
        
       
        action1 = agent1.choose_action(state, actions)
        next_state, reward1, done, _ = env.step(action1, 1)
        
        if done:
            agent1.update(state, action1, reward1, next_state, [])
            break

        next_actions = env.available_actions()
       
        action2 = random.choice(next_actions)
        final_state, reward2, done, _ = env.step(action2, -1)
        
        reward1_total = reward1 + (-reward2) 
        agent1.update(state, action1, reward1_total, final_state, env.available_actions())
        
        state = final_state
        if not done:
            actions = env.available_actions()

    agent1.decay_epsilon()

