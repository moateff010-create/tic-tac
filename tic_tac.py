import random
import numpy as np


class TicTacToeEnv:
    def __init__(self):
        # 1 = X, -1 = O, 0 = empty
        self.board = np.zeros(9, dtype=int)

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        return tuple(self.board)

    def available_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def check_winner(self):
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6],
        ]
        for a, b, c in lines:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        if 0 not in self.board:
            return 0
        return None

    def step(self, action, player):
        self.board[action] = player
        winner = self.check_winner()
        done = winner is not None
        return tuple(self.board), winner, done


class QAgent:
    def __init__(self, player, alpha=0.2, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.9995, min_epsilon=0.05, rewards=None):
        self.player = player  # 1 for X, -1 for O
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q = {}
        self.rewards = rewards or {
            "win": 1.0,
            "loss": -1.0,
            "draw": 0.2,
            "step": 0.0,
        }

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, state, actions, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.choice(actions)
        vals = [self.get_q(state, a) for a in actions]
        max_v = max(vals)
        best = [a for a, v in zip(actions, vals) if v == max_v]
        return random.choice(best)

    def update(self, state, action, reward, next_state, next_actions, done):
        old_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            next_max = max([self.get_q(next_state, a) for a in next_actions], default=0.0)
            target = reward + self.gamma * next_max
        self.q[(state, action)] = old_q + self.alpha * (target - old_q)

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def reward_for_player(player, winner, draw_reward):
    if winner == 0:
        return draw_reward
    if winner == player:
        return 1.0
    return -1.0  # خسارة لأي لاعب (بما فيهم X) تعطي -1


def train_self_play(episodes=40000, alpha=0.2, gamma=0.9, draw_reward=0.2, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    env = TicTacToeEnv()
    rewards_cfg = {"win": 1.0, "loss": -1.0, "draw": draw_reward, "step": 0.0}
    agent_x = QAgent(1, alpha=alpha, gamma=gamma, rewards=rewards_cfg)
    agent_o = QAgent(-1, alpha=alpha, gamma=gamma, rewards=rewards_cfg)

    for _ in range(episodes):
        state = env.reset()
        done = False
        last_x = None  # (state, action)
        last_o = None  # (state, action)
        turn = 1

        while not done:
            actions = env.available_actions()
            if turn == 1:
                action = agent_x.choose_action(state, actions)
                next_state, winner, done = env.step(action, 1)
                if done:
                    r_x = reward_for_player(1, winner, draw_reward)
                    agent_x.update(state, action, r_x, next_state, [], True)
                    if last_o is not None:
                        r_o = reward_for_player(-1, winner, draw_reward)
                        agent_o.update(last_o[0], last_o[1], r_o, next_state, [], True)
                else:
                    if last_o is not None:
                        agent_o.update(last_o[0], last_o[1], 0.0, next_state, env.available_actions(), False)
                    last_x = (state, action)
                state = next_state
                turn = -1
            else:
                action = agent_o.choose_action(state, actions)
                next_state, winner, done = env.step(action, -1)
                if done:
                    r_o = reward_for_player(-1, winner, draw_reward)
                    agent_o.update(state, action, r_o, next_state, [], True)
                    if last_x is not None:
                        r_x = reward_for_player(1, winner, draw_reward)
                        agent_x.update(last_x[0], last_x[1], r_x, next_state, [], True)
                else:
                    if last_x is not None:
                        agent_x.update(last_x[0], last_x[1], 0.0, next_state, env.available_actions(), False)
                    last_o = (state, action)
                state = next_state
                turn = 1

        agent_x.decay()
        agent_o.decay()

    return agent_x, agent_o


def minimax(board_arr, is_maximizing_x):
    env = TicTacToeEnv()
    env.board = board_arr.copy()
    w = env.check_winner()
    if w == 1:
        return 1
    if w == -1:
        return -1
    if w == 0:
        return 0

    actions = env.available_actions()
    if is_maximizing_x:
        best = -999
        for a in actions:
            env.board[a] = 1
            best = max(best, minimax(env.board, False))
            env.board[a] = 0
        return best
    else:
        best = 999
        for a in actions:
            env.board[a] = -1
            best = min(best, minimax(env.board, True))
            env.board[a] = 0
        return best


def best_minimax_move(board_arr, player):
    env = TicTacToeEnv()
    env.board = board_arr.copy()
    actions = env.available_actions()
    if player == 1:
        best_val = -999
        best_moves = []
        for a in actions:
            env.board[a] = 1
            val = minimax(env.board, False)
            env.board[a] = 0
            if val > best_val:
                best_val = val
                best_moves = [a]
            elif val == best_val:
                best_moves.append(a)
        return random.choice(best_moves)
    else:
        best_val = 999
        best_moves = []
        for a in actions:
            env.board[a] = -1
            val = minimax(env.board, True)
            env.board[a] = 0
            if val < best_val:
                best_val = val
                best_moves = [a]
            elif val == best_val:
                best_moves.append(a)
        return random.choice(best_moves)


def evaluate(agent_x, opponent_type="random", games=500):
    env = TicTacToeEnv()
    wins = draws = losses = 0

    for _ in range(games):
        state = env.reset()
        done = False
        turn = 1
        while not done:
            actions = env.available_actions()
            if turn == 1:
                action = agent_x.choose_action(state, actions, greedy=True)
            else:
                if opponent_type == "random":
                    action = random.choice(actions)
                else:
                    action = best_minimax_move(env.board, -1)
            state, winner, done = env.step(action, turn)
            turn *= -1

        if winner == 1:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return {
        "win_rate": wins / games,
        "draw_rate": draws / games,
        "loss_rate": losses / games,
    }


if __name__ == "__main__":
    agent_x, _ = train_self_play(episodes=40000, alpha=0.2, gamma=0.9, draw_reward=0.2)
    print("Baseline vs Random:", evaluate(agent_x, "random", games=500))
    print("Baseline vs Minimax:", evaluate(agent_x, "minimax", games=200))

    print("\n--- Experiments on alpha, gamma, and draw_reward ---")
    configs = [
        {"name": "low_alpha (0.05)", "alpha": 0.05, "gamma": 0.9, "draw_reward": 0.2},
        {"name": "high_alpha (0.5)", "alpha": 0.5, "gamma": 0.9, "draw_reward": 0.2},
        {"name": "low_gamma (0.3)", "alpha": 0.2, "gamma": 0.3, "draw_reward": 0.2},
        {"name": "high_gamma (0.99)", "alpha": 0.2, "gamma": 0.99, "draw_reward": 0.2},
        {"name": "draw_0.0", "alpha": 0.2, "gamma": 0.9, "draw_reward": 0.0},
        {"name": "draw_0.5", "alpha": 0.2, "gamma": 0.9, "draw_reward": 0.5},
    ]

    for cfg in configs:
        ax, _ = train_self_play(
            episodes=30000,
            alpha=cfg["alpha"],
            gamma=cfg["gamma"],
            draw_reward=cfg["draw_reward"],
        )
        r_res = evaluate(ax, "random", games=300)
        m_res = evaluate(ax, "minimax", games=100)
        print(f"{cfg['name']:20s} | vs Random: {r_res} | vs Minimax: {m_res}")
        
### Experiments & Behavior Analysis
#Reward Sign & Symmetry:** Losing always returns `-1` for the losing side (including `X`), winning returns `+1` for the winner, and draws return the same draw reward regardless of which player made the final move.
#True Self-Play:** Both `X` and `O` are `QAgent` instances updating their own Q-tables from gameplay outcomes rather than `O` playing purely at random.
#Alpha (Learning Rate):** Low `alpha` makes learning slower and smoother across noisy self-play updates; high `alpha` adapts faster to recent outcomes but can oscillate more if updates are noisy.
#Gamma (Discount Factor):** Higher `gamma` values future states leading to wins/draws more strongly, whereas lower `gamma` discounts long-horizon outcomes more heavily.
#Draw Reward:** A moderate positive draw reward encourages safe/defensive play against strong opponents (like Minimax) when a win is blocked, while `0.0` treats draws neutrally compared to non-terminal steps.        

