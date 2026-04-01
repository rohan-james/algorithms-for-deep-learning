import random


class DefaultOpponentConnectFour:
    def __init__(self, player="O", opponent="X"):
        self.player = player
        self.opponent = opponent
        self.column_preference = [4, 3, 5, 2, 6, 1, 7]

    def find_winning_move(self, game, symbol):
        for col in game.get_available_moves():
            row = game.get_next_open_row(col - 1)
            if row is None:
                continue
            game.board_tracker[row][col - 1] = symbol
            if game.check_winner(symbol):
                game.board_tracker[row][col - 1] = " "
                return col
            game.board_tracker[row][col - 1] = " "
        return None

    def choose_move(self, game):
        available = game.get_available_moves()
        if not available:
            return None

        win_move = self.find_winning_move(game, self.player)
        if win_move is not None:
            print(f"opponent playing winning move at column {win_move}")
            return win_move

        block_move = self.find_winning_move(game, self.opponent)
        if block_move is not None:
            print(f"oppnent blocking opponent at column {block_move}")
            return block_move

        for col in self.column_preference:
            if col in available:
                print(f"opponent playing positionally at column {col}")
                return col

        chosen = random.choice(available)
        print(f"opponent playing randomly at column {chosen}")
        return chosen

    def take_turn(self, game):
        move = self.choose_move(game)
        if move is not None:
            game.mark_choice(move, self.player)
            print(game.construct_grid())
