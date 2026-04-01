import pandas as pd
import time

from collections import defaultdict


class DeepQNetworkTest:
    def __init__(
        self,
        game,
        agent1,
        agent2,
        n_games,
        results_path,
        per_game_folder_path,
        agent_x_label,
        agent_o_label,
    ):
        self.game = game
        self.agent_x = agent1
        self.agent_o = agent2
        self.results_path = results_path
        self.per_game_folder_path = per_game_folder_path
        self.n_games = n_games
        self.agent_x_label = agent_x_label
        self.agent_o_label = agent_o_label

    def run_single_game(self, verbose=False):
        self.game.reset()
        agents = {"X": self.agent_x, "O": self.agent_o}
        turn = "X"
        nodes_x = 0
        nodes_y = 0

        while not self.game.is_game_over():
            if hasattr(agents[turn], "nodes_visited"):
                agents[turn].nodes_visited = 0

            agents[turn].take_turn(self.game)
            print(self.game.construct_grid())
            print("X count:", sum(row.count("X") for row in self.game.board_tracker))
            print("O count:", sum(row.count("O") for row in self.game.board_tracker))

            if turn == "X" and hasattr(self.agent_x, "nodes_visited"):
                nodes_x += self.agent_x.nodes_visited
            elif turn == "O" and hasattr(self.agent_o, "nodes_visited"):
                nodes_y += self.agent_o.nodes_visited

            if verbose:
                print(f"--- {turn}'s move ---")
                print(self.game.construct_grid())

            if self.game.check_winner(turn):
                return turn, nodes_x, nodes_y
            if self.game.check_draw():
                return "Draw", nodes_x, nodes_y

            turn = "O" if turn == "X" else "X"

        if self.game.check_winner("X"):
            return "X", nodes_x, nodes_y
        if self.game.check_winner("O"):
            return "O", nodes_x, nodes_y
        return "Draw", nodes_x, nodes_y

    def run_match(
        self,
        verbose=False,
    ):
        results = defaultdict(int)
        total_nodes_x = 0
        total_nodes_o = 0
        total_time = 0
        per_game_rows = []

        for i in range(self.n_games):
            start = time.time()
            outcome, nodes_x, nodes_o = self.run_single_game(verbose=verbose)
            elapsed = time.time() - start
            total_time += elapsed
            results[outcome] += 1
            total_nodes_x += nodes_x
            total_nodes_o += nodes_o

            per_game_rows.append(
                {
                    "game": i + 1,
                    "outcome": outcome,
                    f"{self.agent_x_label}_won": int(outcome == "X"),
                    f"{self.agent_o_label}_won": int(outcome == "O"),
                    "draw": int(outcome == "Draw"),
                    "time_s": round(elapsed, 4),
                    f"nodes_{self.agent_x_label}": (
                        nodes_x if hasattr(self.agent_x, "nodes_visited") else None
                    ),
                    f"nodes_{self.agent_o_label}": (
                        nodes_o if hasattr(self.agent_o, "nodes_visited") else None
                    ),
                }
            )

            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{self.n_games} games...")

        avg_nodes_x = (
            total_nodes_x / self.n_games
            if hasattr(self.agent_x, "nodes_visited")
            else None
        )
        avg_nodes_o = (
            total_nodes_o / self.n_games
            if hasattr(self.agent_o, "nodes_visited")
            else None
        )

        stats = {
            "label_x": self.agent_x_label,
            "label_o": self.agent_o_label,
            "n_games": self.n_games,
            "x_wins": results["X"],
            "o_wins": results["O"],
            "draws": results["Draw"],
            "x_win_rate": results["X"] / self.n_games,
            "o_win_rate": results["O"] / self.n_games,
            "draw_rate": results["Draw"] / self.n_games,
            "avg_time_per_game": total_time / self.n_games,
            "avg_nodes_x": avg_nodes_x,
            "avg_nodes_o": avg_nodes_o,
        }

        self._print_match_summary(stats)
        self._create_per_game_csv(self.per_game_folder_path, per_game_rows)
        return stats

    def _print_match_summary(self, stats):

        print(f"\n{'='*50}")
        print(f"  {stats['label_x']}  vs  {stats['label_o']}")
        print(f"  Games played : {stats['n_games']}")
        print(
            f"  {stats['label_x']} wins : {stats['x_wins']} "
            f"({stats['x_win_rate']*100:.1f}%)"
        )
        print(
            f"  {stats['label_o']} wins : {stats['o_wins']} "
            f"({stats['o_win_rate']*100:.1f}%)"
        )
        print(f"  Draws        : {stats['draws']} " f"({stats['draw_rate']*100:.1f}%)")
        print(f"  Avg time/game: {stats['avg_time_per_game']:.4f}s")
        if stats["avg_nodes_x"]:
            print(f"  Avg nodes ({stats['label_x']}): {stats['avg_nodes_x']:,.0f}")
        if stats["avg_nodes_o"]:
            print(f"  Avg nodes ({stats['label_o']}): {stats['avg_nodes_o']:,.0f}")
        print(f"{'='*50}\n")

    def _create_per_game_csv(self, save_path, per_game_rows):

        per_game_path = save_path.replace(".csv", "_per_game.csv")
        df = pd.DataFrame(per_game_rows)
        df.to_csv(per_game_path, index=False)

    def save_ql_match_summary(self, stats):

        result = {
            "competitors": f"{stats['label_x']} vs {stats['label_o']}",
            "games played": stats["n_games"],
            f"{stats['label_x']} wins": stats["x_wins"],
            f"{stats['label_x']} win_rate": f"({stats['x_win_rate']*100:.1f}%)",
            f"{stats['label_o']} wins": stats["o_wins"],
            f"{stats['label_o']} win_rate": f"({stats['o_win_rate']*100:.1f}%)",
            "draws": stats["draws"],
            "draw_rate": f"({stats['draw_rate']*100:.1f}%)",
            "avg time/game": f"{stats['avg_time_per_game']:.4f}s",
        }
        df = pd.DataFrame([result])
        df.to_csv(f"{self.results_path}.csv", index=False)
        print(f"Match summary saved to {self.results_path}.csv")

    def run_test(self):
        print("=" * 50)
        print("Training Q-Learning agent on Tic Tac Toe...")
        print("=" * 50)

        stats = self.run_match()
        self.save_ql_match_summary(stats)
