class TicTacToe:
    def __init__(self):
        self.user_inputs = set()
        self.board_tracker = [[" " for _ in range(3)] for _ in range(3)]

        self.position_map = {
            1: (0, 0),
            2: (0, 1),
            3: (0, 2),
            4: (1, 0),
            5: (1, 1),
            6: (1, 2),
            7: (2, 0),
            8: (2, 1),
            9: (2, 2),
        }

    def construct_grid(self):
        rows = []
        for i, row in enumerate(self.board_tracker):
            rows.append(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                rows.append("---+---+---")

        return "\n".join(rows)

    def mark_choice(self, position, player):
        if position not in self.position_map:
            print("invalid position. Choose between 1 and 9")
            return False
        row, col = self.position_map[position]
        if self.board_tracker[row][col] != " ":
            print("position already taken. Choose another")
            return False
        self.board_tracker[row][col] = player
        self.user_inputs.add(position)
        return True

    def get_available_moves(self):
        return [
            pos
            for pos, (r, c) in self.position_map.items()
            if self.board_tracker[r][c] == " "
        ]

    def check_winner(self, player):
        b = self.board_tracker
        for i in range(3):
            if all(b[i][j] == player for j in range(3)):
                return True
            if all(b[j][i] == player for j in range(3)):
                return True

        if all(b[i][i] == player for i in range(3)):
            return True
        if all(b[i][2 - i] == player for i in range(3)):
            return True
        return False

    def check_draw(self):
        return not self.get_available_moves()

    def is_game_over(self):
        return self.check_winner("X") or self.check_winner("O") or self.check_draw()

    def reset(self):
        self.board_tracker = [[" " for _ in range(3)] for _ in range(3)]
        self.user_inputs = set()

    def take_user_input(self, player="X"):
        available = self.get_available_moves()

        while True:
            try:
                user_input = int(input(f"Player {player}, enter your position (1-9): "))
                if self.mark_choice(user_input, player):
                    break
            except ValueError:
                print("please enter a valid range")
        print(self.construct_grid())
