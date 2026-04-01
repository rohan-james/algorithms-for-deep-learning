import random
import pickle
import torch
from collections import defaultdict

from games.TicTacToe import TicTacToe
from games.ConnectFour import ConnectFour

from agents.DefaultOpponent import DefaultOpponent
from agents.DefaultOpponentConnectFour import DefaultOpponentConnectFour
from agents.MiniMax import MiniMaxAgent
from agents.QLearning import TabularQLearningAgent
from agents.DQN import DQNAgent

from experiments.DQNTests import DeepQNetworkTest

random.seed(42)


def load_dqn(game, player, opponent):
    """Load a saved DQNAgent (online + target nets) from a checkpoint."""
    print(game)
    if game == "tictactoe":
        input_size = 9
        output_size = 9
    else:
        input_size = 42
        output_size = 7
    path = f"models/dqn_{game}.pkl"
    agent = DQNAgent(
        player=player,
        opponent=opponent,
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=(64, 64),
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    agent.online_net.load_state_dict(ckpt["online_state_dict"])
    agent.target_net.load_state_dict(ckpt["target_state_dict"])
    agent.epsilon = 0.0
    agent.online_net.eval()
    return agent


def load_qlearning(game, player, opponent):

    path = f"models/qtable_{game}.pkl"
    agent = TabularQLearningAgent(player=player, opponent=opponent)
    with open(path, "rb") as f:
        raw = pickle.load(f)
    # re-wrap as defaultdict so missing states still return 0
    agent.q_table = defaultdict(
        lambda: defaultdict(float), {k: defaultdict(float, v) for k, v in raw.items()}
    )
    agent.epsilon = 0.0
    return agent


def run_matchup(label_x, agent_x, label_o, agent_o, game, n_games=2):

    tester = DeepQNetworkTest(
        game=game,
        agent1=agent_x,
        agent2=agent_o,
        n_games=n_games,
        results_path=f"results_test/{label_x}_vs_{label_o}",
        per_game_folder_path=f"results_test/{label_x}_vs_{label_o}",
        agent_x_label=label_x,
        agent_o_label=label_o,
    )
    return tester.run_match()


GAME_OPTIONS = [
    ("1. Tic-Tac-Toe", TicTacToe(), "tictactoe"),
    ("2. Connect Four", ConnectFour(), "connectfour"),
]


def build_agent_options(is_connect_four, rl_game):

    mm_depth = 5 if is_connect_four else 9
    mm_ab_depth = 5 if is_connect_four else 9

    default_opponent_o = (
        DefaultOpponentConnectFour(player="O", opponent="X")
        if is_connect_four
        else DefaultOpponent(player="O", opponent="X")
    )

    agent1_options = [
        (
            "1. Minimax",
            MiniMaxAgent(
                player="X", opponent="O", depth_limit=mm_depth, use_alpha_beta=False
            ),
            "minimax",
        ),
        (
            "2. Minimax With AB Pruning",
            MiniMaxAgent(
                player="X", opponent="O", depth_limit=mm_ab_depth, use_alpha_beta=True
            ),
            "minimax_ab",
        ),
        (
            "3. Q-Learning",
            load_qlearning(rl_game, player="X", opponent="O"),
            "q_learning",
        ),
        ("4. DQN", load_dqn(rl_game, player="X", opponent="O"), "dqn"),
    ]

    agent2_options = [
        (
            "1. Minimax",
            MiniMaxAgent(
                player="O", opponent="X", depth_limit=mm_depth, use_alpha_beta=False
            ),
            "minimax",
        ),
        (
            "2. Minimax With AB Pruning",
            MiniMaxAgent(
                player="O", opponent="X", depth_limit=mm_ab_depth, use_alpha_beta=True
            ),
            "minimax_ab",
        ),
        (
            "3. Q-Learning",
            load_qlearning(rl_game, player="O", opponent="X"),
            "q_learning",
        ),
        ("4. Default Opponent", default_opponent_o, "default"),
    ]

    return agent1_options, agent2_options


def prompt():
    for item in GAME_OPTIONS:
        print(item[0])
    game_idx = int(input("Enter the index of the game to trial: "))
    print("Please wait, loading game models ...")

    selected = GAME_OPTIONS[game_idx - 1]
    game = selected[1]
    rl_game = selected[2]
    is_connect_four = rl_game == "connectfour"

    agent1_options, agent2_options = build_agent_options(is_connect_four, rl_game)

    for item in agent1_options:
        print(item[0])
    player_1_idx = int(input("Select player 1: "))
    agent_1 = agent1_options[player_1_idx - 1][1]

    for item in agent2_options:
        print(item[0])
    player_2_idx = int(input("Select player 2: "))
    agent_2 = agent2_options[player_2_idx - 1][1]

    run_matchup(
        agent1_options[player_1_idx - 1][2],
        agent_1,
        agent2_options[player_2_idx - 1][2],
        agent_2,
        game,
        n_games=10,
    )


def main():
    prompt()


if __name__ == "__main__":
    main()
