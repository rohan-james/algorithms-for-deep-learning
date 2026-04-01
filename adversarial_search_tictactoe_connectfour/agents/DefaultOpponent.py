import random


class DefaultOpponent:
    def __init__(self, player="O", opponent="X"):
        self.player = player
        self.opponent = opponent

    def find_winning_move(self, game, symbol):
        for move in game.get_available_moves():
            row, col = game.position_map[move]
            game.board_tracker[row][col] = symbol
            if game.check_winner(symbol):
                game.board_tracker[row][col] = " "
                return move
            game.board_tracker[row][col] = " "
        return None

    def choose_move(self, game):
        available = game.get_available_moves()
        if not available:
            return None

        win_move = self.find_winning_move(game, self.player)
        if win_move is not None:
            print(f"Opponent playing winning move at {win_move}")
            return win_move

        block_move = self.find_winning_move(game, self.opponent)
        if block_move is not None:
            print(f"Opponent blocking opponent at {block_move}")
            return block_move

        chosen = random.choice(available)
        print(f"Opponent playing randomly at {chosen}")
        return chosen

    def take_turn(self, game):
        move = self.choose_move(game)
        if move is not None:
            game.mark_choice(move, self.player)
            print(game.construct_grid())
