import pickle
import math
import time
import copy
import random

from collections import defaultdict


class TabularQLearningAgent:
    def __init__(
        self,
        player,
        opponent,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
    ):
        self.player = player
        self.opponent = opponent
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = defaultdict(lambda: defaultdict(float))

        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []
        self.epsilon_history = []

    def get_state(self, game):
        return tuple(cell for row in game.board_tracker for cell in row)

    def get_q_value(self, state, action):
        return self.q_table[state][action]

    def get_best_action(self, state, available_moves):
        best_value = -math.inf
        best_actions = []

        for action in available_moves:
            q = self.get_q_value(state, action)
            if q > best_value:
                best_value = q
                best_actions = [action]
            elif q == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def choose_action(self, game, training=True):
        available = game.get_available_moves()

        if not available:
            return None

        if not training:
            state = self.get_state(game)
            return self.get_best_action(state, available)

        if random.random() < self.epsilon:
            return random.choice(available)
        else:
            state = self.get_state(game)
            return self.get_best_action(state, available)

    def _apply_move(self, game, move, player):
        if hasattr(game, "rows"):
            col_idx = move - 1
            row = game.get_next_open_row(col_idx)
            if row is not None:
                game.board_tracker[row][col_idx] = player

        else:
            row, col = game.position_map[move]
            game.board_tracker[row][col] = player

    def get_reward(self, game):
        if game.check_winner(self.player):
            return 1.0
        if game.check_winner(self.opponent):
            return -1.0
        if game.check_draw():
            return 0.1
        return 0.0

    def update(self, state, action, reward, next_state, next_available):
        current_q = self.get_q_value(state, action)

        if not next_available:
            target = reward
        else:
            future_q = max(self.get_q_value(next_state, a) for a in next_available)
            target = reward + self.gamma * future_q

        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def take_turn(self, game):
        action = self.choose_action(game, training=False)
        if action is not None:
            self._apply_move(game, action, self.player)

    def _evaluate(self, game, opponent, n_games=200):
        wins = 0
        eval_game = copy.deepcopy(game)
        for _ in range(n_games):
            eval_game.reset()
            turn = "X"
            while not eval_game.is_game_over():
                if turn == self.player:
                    action = self.choose_action(eval_game, training=False)
                    if action:
                        self._apply_move(eval_game, action, self.player)

                else:
                    opponent.take_turn(eval_game)
                turn = "O" if turn == "X" else "X"
            if eval_game.check_winner(self.player):
                wins += 1

        return wins / n_games

    def train(
        self,
        game,
        opponent,
        n_episodes=50000,
        eval_interval=1000,
        eval_games=1000,
        verbose=True,
    ):
        wins = 0
        losses = 0
        draws = 0

        start_time = time.time()
        initial_win_rate = self._evaluate(game, opponent, eval_games)
        self.win_rates.append(initial_win_rate)

        for episode in range(1, n_episodes + 1):
            game.reset()
            episode_reward = 0
            episode_length = 0
            turn = "X"

            last_state = None
            last_action = None

            while not game.is_game_over():
                if turn == self.player:
                    state = self.get_state(game)
                    action = self.choose_action(game, training=True)

                    if action is None:
                        break

                    self._apply_move(game, action, self.player)
                    episode_length += 1

                    reward = self.get_reward(game)
                    next_state = self.get_state(game)
                    next_available = game.get_available_moves()

                    if game.is_game_over():
                        self.update(state, action, reward, next_state, next_available)
                        episode_reward += reward
                        if reward == 1.0:
                            wins += 1
                        elif reward == 0.1:
                            draws += 1
                        break

                    # self.update(state, action, 0.0, next_state, next_available)
                    # episode_reward += 0.0

                    last_state = state
                    last_action = action
                else:
                    opponent.take_turn(game)
                    episode_length += 1

                    reward = self.get_reward(game)

                    if last_state is not None and last_action is not None:
                        next_state = self.get_state(game)
                        next_available = game.get_available_moves()
                        self.update(
                            last_state, last_action, reward, next_state, next_available
                        )
                        episode_reward += reward
                    if game.is_game_over():
                        if reward == -1.0:
                            losses += 1
                        elif reward == 0.1:
                            draws += 1
                        break
                turn = "O" if turn == "X" else "X"

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.epsilon_history.append(self.epsilon)

            if episode % eval_interval == 0:
                win_rate = self._evaluate(game, opponent, eval_games)
                self.win_rates.append(win_rate)
                elapsed = time.time() - start_time

                if verbose:
                    print(
                        f"Episode {episode:>6}/{n_episodes} | "
                        f"Epsilon: {self.epsilon:.4f} | "
                        f"Win rate: {win_rate*100:.1f}% | "
                        f"Q-table size: {len(self.q_table):,} | "
                        f"Elapsed: {elapsed:.0f}s"
                    )

        if verbose:
            total_time = time.time() - start_time
            print(f"\nTraining complete.")
            print(f"Total episodes: {n_episodes:,}")
            print(f"Q-table states: {len(self.q_table):,}")
            print(f"Final epsilon: {self.epsilon:.4f}")
            print(f"Total time: {total_time:.1f}s")

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(dict(self.q_table), f)
