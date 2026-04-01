class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board_tracker = [[" " for _ in range(self.cols)] for _ in range(self.rows)]
        self.user_inputs = []

        self.position_map = {}
        self.construct_position_map()

    def construct_position_map(self):
        for i in range(self.cols):
            self.position_map[i + 1] = (0, i)

    def get_next_open_row(self, col):
        for row in range(self.rows - 1, -1, -1):
            if self.board_tracker[row][col] == " ":
                return row
        return None

    def get_available_moves(self):
        return [
            col + 1
            for col in range(self.cols)
            if self.get_next_open_row(col) is not None
        ]

    def check_winner(self, player):
        b = self.board_tracker

        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(b[row][col + k] == player for k in range(4)):
                    return True

        for row in range(self.rows - 3):
            for col in range(self.cols):
                if all(b[row + k][col] == player for k in range(4)):
                    return True

        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if all(b[row - k][col + k] == player for k in range(4)):
                    return True

        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(b[row + k][col + k] == player for k in range(4)):
                    return True

        return False

    def reset(self):
        self.board_tracker = [[" " for _ in range(self.cols)] for _ in range(self.rows)]
        self.user_inputs = []

    def check_draw(self):
        return not self.get_available_moves()

    def mark_choice(self, col_choice, player):
        if col_choice not in self.position_map:
            print(f"invalid column. Please choose between 1 and {self.cols}")
            return False

        col_idx = col_choice - 1
        row = self.get_next_open_row(col_idx)

        if row is None:
            print(f"column {col_choice} is full. Choose another column")
            return False

        self.board_tracker[row][col_idx] = player
        self.user_inputs.append(col_choice)

        return True

    def is_game_over(self):
        return self.check_winner("X") or self.check_winner("O") or self.check_draw()

    def construct_grid(self):
        header = "  1   2   3   4   5   6   7"
        rows = []
        for row in self.board_tracker:
            rows.append("| " + " | ".join(row) + " |")
            rows.append("-" * 29)

        rows.append(header)

        return "\n".join(rows)

    def take_user_input(self, player="X"):
        available = self.get_available_moves()
        print(f"Available columns: {available}")
        while True:
            try:
                col_choice = int(input(f"player {player}, choose a column (1-7): "))
                if self.mark_choice(col_choice, player):
                    break
            except ValueError:
                print("Please enter a valid integer.")
        print(self.construct_grid())
