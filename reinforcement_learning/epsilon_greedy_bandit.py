import numpy as np


class EpsilonGreedyBandit:
    def __init__(self, k=10, epsilon=0.1, steps=1000):
        self.k = k
        self.epsilon = epsilon  # Exploration probability
        self.steps = steps
        self.q_true = np.random.normal(0, 1, k)  # True reward distribition
        self.q_estimates = np.zeros(k)  # Estimates action values
        self.action_counts = np.zeros(k)  # Count of times each action was taken

    def select_action(self):
        """
        Select an action using epsilon-greedy strategy
        """
        if np.random.randn() < self.epsilon:
            return np.random.randint(self.k)  # Explore
        return np.argmax(self.q_estimates)  # Exploit

    def update(self, action, reward):
        """
        Update the action-value estimate using incremental mean
        """
        self.action_counts[action] += 1
        self.q_estimates[action] += (
            reward - self.q_estimates[action]
        ) / self.action_counts[action]

    def run(self):
        rewards = []
        for _ in range(self.steps):
            action = self.select_action()
            reward = np.random.normal(
                self.q_true[action], 1
            )  # Sample reward from normal distribution
            self.update(action, reward)
            rewards.append(reward)
        return rewards, self.q_estimates


bandit = EpsilonGreedyBandit()
rewards, estimates = bandit.run()
print("final Q-value estimates: ", estimates)
