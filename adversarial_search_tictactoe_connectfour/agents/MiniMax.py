import math


class MiniMaxAgent:
    def __init__(self, player, opponent, depth_limit=None, use_alpha_beta=False):
        self.player = player
        self.opponent = opponent
        self.depth_limit = depth_limit
        self.use_alpha_beta = use_alpha_beta
        self.nodes_visited = 0

    def _make_move(self, game, move, player):
        if hasattr(game, "rows"):
            col_idx = move - 1
            row = game.get_next_open_row(col_idx)
            if row is not None:
                game.board_tracker[row][col_idx] = player
        else:
            row, col = game.position_map[move]
            game.board_tracker[row][col] = player

    def _get_terminal_score(self, game, depth):
        if game.check_winner(self.player):
            return 1000 + depth
        if game.check_winner(self.opponent):
            return -(1000 + depth)
        if game.check_draw():
            return 0

        if depth == 0:
            if hasattr(game, "rows"):
                return self.evaluate(game)
            return 0

        return None

    def _undo_move(self, game, move):
        if hasattr(game, "rows"):
            col_idx = move - 1
            for row in range(game.rows):
                if game.board_tracker[row][col_idx] != " ":
                    game.board_tracker[row][col_idx] = " "
                    break
        else:
            row, col = game.position_map[move]
            game.board_tracker[row][col] = " "

    def minimax(self, game, depth, is_maximising):
        self.nodes_visited += 1

        terminal_score = self._get_terminal_score(game, depth)
        if terminal_score is not None:
            return terminal_score

        available = game.get_available_moves()

        if is_maximising:
            best_score = -math.inf
            for move in available:
                self._make_move(game, move, self.player)
                score = self.minimax(game, depth - 1, is_maximising=False)
                self._undo_move(game, move)
                best_score = max(best_score, score)
            return best_score

        else:
            best_score = math.inf
            for move in available:
                self._make_move(game, move, self.opponent)
                score = self.minimax(game, depth - 1, is_maximising=True)
                self._undo_move(game, move)
                best_score = min(best_score, score)
            return best_score

    def minimax_ab(self, game, depth, alpha, beta, is_maximising):
        self.nodes_visited += 1

        terminal_score = self._get_terminal_score(game, depth)
        if terminal_score is not None:
            return terminal_score

        available = game.get_available_moves()

        if is_maximising:
            best_score = -math.inf
            for move in available:
                self._make_move(game, move, self.player)
                score = self.minimax_ab(
                    game, depth - 1, alpha, beta, is_maximising=False
                )
                self._undo_move(game, move)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)

                if beta <= alpha:
                    break

            return best_score
        else:
            best_score = math.inf
            for move in available:
                self._make_move(game, move, self.opponent)
                score = self.minimax_ab(
                    game, depth - 1, alpha, beta, is_maximising=True
                )
                self._undo_move(game, move)
                best_score = min(best_score, score)
                beta = min(beta, best_score)

                if beta <= alpha:
                    break

            return best_score

    def choose_move(self, game):
        self.nodes_visited = 0
        available = game.get_available_moves()

        if not available:
            return None

        best_move = None
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf

        depth = self.depth_limit if self.depth_limit is not None else len(available)

        for move in available:
            self._make_move(game, move, self.player)

            if self.use_alpha_beta:
                score = self.minimax_ab(
                    game, depth - 1, alpha, beta, is_maximising=False
                )
            else:
                score = self.minimax(game, depth - 1, is_maximising=False)

            self._undo_move(game, move)
            if score > best_score:
                best_score = score
                best_move = move

            if self.use_alpha_beta:
                alpha = max(alpha, best_score)

        return best_move

    def evaluate(self, game):
        score = 0
        b = game.board_tracker
        rows = game.rows
        cols = game.cols

        center_col = cols // 2
        center_array = [b[r][center_col] for r in range(rows)]
        score += center_array.count(self.player) * 3

        windows = []

        for r in range(rows):
            for c in range(cols - 3):
                windows.append([b[r][c + k] for k in range(4)])

        for r in range(rows - 3):
            for c in range(cols):
                windows.append([b[r + k][c] for k in range(4)])

        for r in range(rows - 3):
            for c in range(cols - 3):
                windows.append([b[r + k][c + k] for k in range(4)])

        for r in range(3, rows):
            for c in range(cols - 3):
                windows.append([b[r - k][c + k] for k in range(4)])

        for window in windows:
            agent_count = window.count(self.player)
            opponent_count = window.count(self.opponent)
            empty_count = window.count(" ")

            if opponent_count == 0:
                if agent_count == 4:
                    score += 100
                elif agent_count == 3 and empty_count == 1:
                    score += 5
                elif agent_count == 2 and empty_count == 2:
                    score += 2

            if agent_count == 0:
                if opponent_count == 4:
                    score -= 100
                elif opponent_count == 3 and empty_count == 1:
                    score -= 4
                elif opponent_count == 2 and empty_count == 2:
                    score -= 1
        return score

    def take_turn(self, game):
        move = self.choose_move(game)
        if move is not None:
            self._make_move(game, move, self.player)
            print(game.construct_grid())
        return move
