import time
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_available):
        self.buffer.append((state, action, reward, next_state, done, next_available))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_availables = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            next_availables,
        )

    def __len__(self):
        return len(self.buffer)


class DeepQNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepQNetwork, self).__init__()

        layers = []
        prev_size = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

        self._initialise_weights()

    def _initialise_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        player,
        opponent,
        input_size,
        output_size,
        hidden_sizes=(128, 128),
        alpha=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
        batch_size=64,
        buffer_capacity=50000,
        target_update_freq=500,
    ):
        self.player = player
        self.opponent = opponent
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        # self.device = torch.device("mps")
        print(f"DQNAgent using device: {self.device}")

        self.online_net = DeepQNetwork(input_size, hidden_sizes, output_size).to(
            self.device
        )

        self.target_net = DeepQNetwork(input_size, hidden_sizes, output_size).to(
            self.device
        )

        self.update_target_network()
        self.target_net.eval()

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=alpha)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.win_rates = []
        self.epsilon_history = []

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_state(self, game):
        encoding = []
        for row in game.board_tracker:
            for cell in row:
                if cell == self.player:
                    encoding.append(1.0)
                elif cell == self.opponent:
                    encoding.append(-1.0)
                else:
                    encoding.append(0.0)
        return np.array(encoding, dtype=np.float32)

    def _greedy_action(self, game, available):
        state = self.get_state(game)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_tensor).squeeze(0)
        self.online_net.train()

        mask = torch.full((self.output_size,), float("-inf"), device=self.device)
        for move in available:
            mask[move - 1] = 0.0

        masked_q = q_values + mask
        action_idx = masked_q.argmax().item()
        return action_idx + 1

    def choose_action(self, game, training=True):
        available = game.get_available_moves()
        if not available:
            return None

        if not training:
            return self._greedy_action(game, available)

        if random.random() < self.epsilon:
            return random.choice(available)
        return self._greedy_action(game, available)

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

    def store_transition(self, state, action, reward, next_state, done, next_available):
        self.replay_buffer.push(
            state, action - 1, reward, next_state, float(done), next_available
        )

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_availables = (
            self.replay_buffer.sample(self.batch_size)
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            raw_next_q = self.target_net(next_states)

            mask = torch.full_like(raw_next_q, float("-inf"))
            for i, available in enumerate(next_availables):
                if available:
                    for move in available:
                        mask[i, move - 1] = 0.0
                else:
                    mask[i] = 0.0

            next_q = (raw_next_q + mask).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimiser.step()
        self.steps_done += 1

        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def take_turn(self, game):
        action = self.choose_action(game, training=False)
        self._apply_move(game, action, self.player)

    def _evaluate(self, game, opponent, n_games=200):
        wins = 0
        for _ in range(n_games):
            game.reset()
            turn = "X"
            while not game.is_game_over():
                if turn == self.player:
                    action = self.choose_action(game, training=False)
                    if action:
                        self._apply_move(game, action, self.player)
                else:
                    opponent.take_turn(game)
                turn = "O" if turn == "X" else "X"
            if game.check_winner(self.player):
                wins += 1
        return wins / n_games

    def save(self, filepath):
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimiser_state": self.optimiser.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
                "input_size": self.input_size,
                "output_size": self.output_size,
            },
            filepath,
        )
        print(f"DQN model saved to {filepath}")

    def train(
        self,
        game,
        opponent,
        n_episodes=50000,
        eval_interval=1000,
        eval_games=20,
        verbose=True,
    ):
        start_time = time.time()

        for episode in range(1, n_episodes + 1):
            game.reset()
            episode_reward = 0
            episode_length = 0
            episode_loss = []
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
                    done = game.is_game_over()

                    if done:
                        self.store_transition(
                            state,
                            action,
                            reward,
                            next_state,
                            done,
                            game.get_available_moves(),
                        )
                        loss = self.train_step()
                        if loss is not None:
                            episode_loss.append(loss)
                        episode_reward += reward
                        break

                    last_state = state
                    last_action = action
                else:
                    opponent.take_turn(game)
                    episode_length += 1

                    reward = self.get_reward(game)
                    done = game.is_game_over()

                    if last_state is not None and last_action is not None:
                        next_state = self.get_state(game)
                        next_available = game.get_available_moves()
                        self.store_transition(
                            last_state,
                            last_action,
                            reward,
                            next_state,
                            done,
                            next_available,
                        )
                        loss = self.train_step()
                        if loss is not None:
                            episode_loss.append(loss)
                        episode_reward += reward

                    if done:
                        break
                turn = "O" if turn == "X" else "X"

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.epsilon_history.append(self.epsilon)

            avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0.0
            self.losses.append(avg_loss)

            if episode % eval_interval == 0:
                win_rate = self._evaluate(game, opponent, eval_games)
                self.win_rates.append(win_rate)
                elapsed = time.time() - start_time

                if verbose:
                    print(
                        f"Episode {episode:>6}/{n_episodes} | "
                        f"Epsilon: {self.epsilon:.4f} | "
                        f"Win rate: {win_rate*100:.1f}% | "
                        f"Avg loss: {avg_loss:.4f} | "
                        f"Buffer: {len(self.replay_buffer):,} | "
                        f"Steps: {self.steps_done:,} | "
                        f"Elapsed: {elapsed:.0f}s"
                    )

        if verbose:
            print(f"\nTraining complete.")
            print(f"Total episodes: {n_episodes:,}")
            print(f"Total steps: {self.steps_done:,}")
            print(f"Final epsilon: {self.epsilon:.4f}")
            print(f"Total time: {time.time() - start_time:.1f}s")
