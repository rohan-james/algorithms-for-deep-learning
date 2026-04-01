# Maze Solver — Setup & Guide

## Environment Setup

This project requires Python 3.9 and a compatible Torch installation for DQN. Follow the steps below to set up the environment using Conda.

```bash
python3 -m venv .venv/
source .venv/bin/activate
pip3 install -r requirements.txt
```

---

## Running the Code

Once the environment is active, run the main file:

```bash
python3 main.py
```

The program will guide you through a series of prompts

## Interactive Prompts

### 1. Game Choice

```
1. Tic-Tac-Toe
2. Connect Four
Enter the index of the game to trial: 
```

### 2. Agent 1 Selection
Choose Player 1 of the Selected Game

```
1. Minimax
2. Minimax With AB Pruning
3. Q-Learning
4. DQN
Select player 1: 
```

### 3. Agent 1 Selection
Choose Player 2 of the Selected Game

```
1. Minimax
2. Minimax With AB Pruning
3. Q-Learning
4. Default Opponent
Select player 2: 
```

## Example Output

After completing the prompts, the program prints the results:

```
==================================================
  dqn  vs  q_learning
  Games played : 10
  dqn wins : 10 (100.0%)
  q_learning wins : 0 (0.0%)
  Draws        : 0 (0.0%)
  Avg time/game: 0.0133s
==================================================
```